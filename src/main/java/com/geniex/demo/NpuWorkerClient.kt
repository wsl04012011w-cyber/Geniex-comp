package com.geniex.demo

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.Message
import android.os.Messenger
import android.os.RemoteException
import com.geniex.sdk.bean.ChatMessage
import com.geniex.sdk.bean.LlmStreamResult
import com.geniex.sdk.bean.ProfilingData
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout

/**
 * Messenger client for NpuWorkerService. All GGML/Hexagon JNI calls stay in
 * the worker process; the UI process receives tokens and profiling records.
 */
class NpuWorkerClient(context: Context) {
    private val appContext = context.applicationContext
    private val lock = Any()
    private val nextRequestId = AtomicInteger(1)
    private val pending = ConcurrentHashMap<Int, CompletableDeferred<Bundle>>()
    private val streams = ConcurrentHashMap<Int, Channel<LlmStreamResult>>()

    @Volatile
    private var remote: Messenger? = null

    @Volatile
    private var connectDeferred: CompletableDeferred<Messenger>? = null

    @Volatile
    private var bound = false

    private val replyHandler = object : Handler(Looper.getMainLooper()) {
        override fun handleMessage(message: Message) {
            val id = message.arg1
            when (message.what) {
                NpuWorkerService.MSG_RESPONSE -> {
                    val result = pending.remove(id)
                    result?.complete(message.data ?: Bundle())
                }

                NpuWorkerService.MSG_TOKEN -> {
                    streams[id]?.trySend(
                        LlmStreamResult.Token(message.data?.getString(NpuWorkerService.KEY_TOKEN).orEmpty()),
                    )
                }

                NpuWorkerService.MSG_COMPLETE -> {
                    val data = message.data ?: Bundle()
                    val profile = ProfilingData(
                        ttftMs = data.getDouble(NpuWorkerService.KEY_TTFT),
                        mediaMs = data.getDouble(NpuWorkerService.KEY_MEDIA_MS),
                        promptTimeMs = data.getDouble(NpuWorkerService.KEY_PROMPT_MS),
                        decodeTimeMs = data.getDouble(NpuWorkerService.KEY_DECODE_MS),
                        promptTokens = data.getLong(NpuWorkerService.KEY_PROMPT_TOKENS),
                        generatedTokens = data.getLong(NpuWorkerService.KEY_GENERATED_TOKENS),
                        prefillSpeed = data.getDouble(NpuWorkerService.KEY_PREFILL_SPEED),
                        decodingSpeed = data.getDouble(NpuWorkerService.KEY_DECODE_SPEED),
                        draftNTotal = data.getLong(NpuWorkerService.KEY_DRAFT_TOTAL),
                        draftNAccepted = data.getLong(NpuWorkerService.KEY_DRAFT_ACCEPTED),
                        stopReason = data.getString(NpuWorkerService.KEY_STOP_REASON).orEmpty(),
                    )
                    streams.remove(id)?.let { channel ->
                        channel.trySend(LlmStreamResult.Completed(profile))
                        channel.close()
                    }
                }

                NpuWorkerService.MSG_ERROR -> {
                    val detail = message.data?.getString(NpuWorkerService.KEY_DETAIL) ?: "NPU worker error"
                    streams.remove(id)?.let { channel ->
                        channel.trySend(LlmStreamResult.Error(IllegalStateException(detail)))
                        channel.close()
                    }
                    pending.remove(id)?.complete(
                        Bundle().apply {
                            putBoolean(NpuWorkerService.KEY_OK, false)
                            putString(NpuWorkerService.KEY_DETAIL, detail)
                        },
                    )
                }

                else -> super.handleMessage(message)
            }
        }
    }

    private val replyMessenger = Messenger(replyHandler)

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            val messenger = Messenger(binder ?: return)
            remote = messenger
            connectDeferred?.complete(messenger)
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            remote = null
            connectDeferred?.completeExceptionally(IllegalStateException("NPU worker process disconnected"))
            connectDeferred = null
        }

        override fun onBindingDied(name: ComponentName?) {
            onServiceDisconnected(name)
        }

        override fun onNullBinding(name: ComponentName?) {
            onServiceDisconnected(name)
        }
    }

    private suspend fun connect(): Messenger {
        remote?.let { return it }
        val ready = withContext(Dispatchers.Main.immediate) {
            synchronized(lock) {
                remote?.let {
                    return@withContext CompletableDeferred<Messenger>().also { deferred ->
                        deferred.complete(it)
                    }
                }

                connectDeferred?.let { return@synchronized it }

                val deferred = CompletableDeferred<Messenger>()
                connectDeferred = deferred
                val intent = Intent(appContext, NpuWorkerService::class.java)
                val didBind = appContext.bindService(intent, connection, Context.BIND_AUTO_CREATE)
                bound = didBind
                if (!didBind) {
                    connectDeferred = null
                    deferred.completeExceptionally(IllegalStateException("Could not bind NPU worker service"))
                }
                deferred
            }
        }
        return withTimeout(30_000) { ready.await() }
    }

    private suspend fun request(command: Int, data: Bundle = Bundle()): Bundle {
        val id = nextRequestId.getAndIncrement()
        val response = CompletableDeferred<Bundle>()
        pending[id] = response
        try {
            val message = Message.obtain(null, command).apply {
                arg1 = id
                this.data = data
                replyTo = replyMessenger
            }
            connect().send(message)
            val result = withTimeout(120_000) { response.await() }
            check(result.getBoolean(NpuWorkerService.KEY_OK)) {
                result.getString(NpuWorkerService.KEY_DETAIL) ?: "NPU worker command failed"
            }
            return result
        } catch (error: Throwable) {
            pending.remove(id)
            throw error
        }
    }

    suspend fun loadModel(modelPath: String, tokenizerPath: String?, config: com.geniex.sdk.bean.ModelConfig) {
        val data = Bundle().apply {
            putString(NpuWorkerService.KEY_MODEL_PATH, modelPath)
            putString(NpuWorkerService.KEY_TOKENIZER_PATH, tokenizerPath)
            putInt(NpuWorkerService.KEY_N_CTX, config.nCtx)
            putInt(NpuWorkerService.KEY_N_THREADS, config.nThreads)
            putInt(NpuWorkerService.KEY_GPU_LAYERS, config.nGpuLayers)
        }
        request(NpuWorkerService.MSG_LOAD, data)
    }

    fun generateStreamFlow(
        messages: List<ChatMessage>,
        tools: String?,
        enableThinking: Boolean,
    ): Flow<LlmStreamResult> = callbackFlow {
        val id = nextRequestId.getAndIncrement()
        val channel = Channel<LlmStreamResult>(Channel.BUFFERED)
        streams[id] = channel

        val roles = ArrayList(messages.map { it.role })
        val contents = ArrayList(messages.map { it.content })
        val data = Bundle().apply {
            putStringArrayList(NpuWorkerService.KEY_ROLES, roles)
            putStringArrayList(NpuWorkerService.KEY_CONTENTS, contents)
            putString(NpuWorkerService.KEY_TOOLS, tools)
            putBoolean(NpuWorkerService.KEY_THINKING, enableThinking)
        }

        try {
            val message = Message.obtain(null, NpuWorkerService.MSG_GENERATE).apply {
                arg1 = id
                this.data = data
                replyTo = replyMessenger
            }
            connect().send(message)

            val relay = launch {
                for (result in channel) trySend(result)
                close()
            }
            awaitClose {
                streams.remove(id)
                channel.close()
                relay.cancel()
            }
        } catch (error: Throwable) {
            streams.remove(id)
            channel.close()
            close(error)
        }
    }

    suspend fun stopStream() {
        request(NpuWorkerService.MSG_STOP)
    }

    suspend fun reset() {
        request(NpuWorkerService.MSG_RESET)
    }

    suspend fun unloadModel() {
        request(NpuWorkerService.MSG_UNLOAD)
    }

    fun close() {
        if (!bound) return
        runCatching { appContext.unbindService(connection) }
        bound = false
        remote = null
        connectDeferred = null
        pending.values.forEach { it.completeExceptionally(IllegalStateException("NPU worker client closed")) }
        pending.clear()
        streams.values.forEach { it.close() }
        streams.clear()
    }
}
