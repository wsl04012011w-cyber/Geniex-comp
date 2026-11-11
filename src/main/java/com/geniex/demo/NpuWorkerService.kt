package com.geniex.demo

import android.app.Service
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.Message
import android.os.Messenger
import android.system.Os
import android.util.Log
import com.geniex.sdk.GenieXSdk
import com.geniex.sdk.LlmWrapper
import com.geniex.sdk.bean.ChatMessage
import com.geniex.sdk.bean.ComputeUnitValue
import com.geniex.sdk.bean.GenerationConfig
import com.geniex.sdk.bean.LlmCreateInput
import com.geniex.sdk.bean.LlmStreamResult
import com.geniex.sdk.bean.ModelConfig
import com.geniex.sdk.bean.ProfilingData
import java.io.File
import java.io.FileOutputStream
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.collect

/**
 * Hexagon worker hosted in a dedicated Android process. The main process ships
 * the baseline ARMv8-A CPU/OpenCL plugin; this process registers the private
 * Hexagon-enabled plugin from filesDir/npu_runtime so loading ggml-hexagon
 * cannot take down the chat UI process.
 */
class NpuWorkerService : Service() {
    @Volatile
    private var llm: LlmWrapper? = null

    @Volatile
    private var generationJob: Job? = null

    @Volatile
    private var pluginRegistered = false

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private val messenger by lazy {
        Messenger(
            Handler(Looper.getMainLooper()) { message ->
                handleCommand(message)
                true
            },
        )
    }

    override fun onBind(intent: Intent?): IBinder = messenger.binder

    private fun handleCommand(message: Message) {
        val requestId = message.arg1
        val replyTo = message.replyTo ?: return
        val data = message.data ?: Bundle()

        when (message.what) {
            MSG_LOAD -> scope.launch {
                val result = runCatching {
                    loadModel(
                        modelPath = data.getString(KEY_MODEL_PATH)
                            ?: error("model path missing"),
                        tokenizerPath = data.getString(KEY_TOKENIZER_PATH),
                        nCtx = data.getInt(KEY_N_CTX, 1024),
                        nThreads = data.getInt(KEY_N_THREADS, 4),
                        nGpuLayers = data.getInt(KEY_GPU_LAYERS, 999),
                    )
                }
                result.fold(
                    onSuccess = {
                        sendResponse(replyTo, requestId, true, "Hexagon worker loaded the GGUF model")
                    },
                    onFailure = {
                        Log.e(TAG, "NPU model load failed", it)
                        sendResponse(replyTo, requestId, false, it.message ?: it.toString())
                    },
                )
            }

            MSG_GENERATE -> {
                if (generationJob?.isActive == true) {
                    sendError(replyTo, requestId, "A generation is already running")
                    return
                }
                generationJob = scope.launch {
                    generate(data, replyTo, requestId)
                }
            }

            MSG_STOP -> scope.launch {
                runCatching { llm?.stopStream() }
                sendResponse(replyTo, requestId, true, "Generation stop requested")
            }

            MSG_RESET -> scope.launch {
                val result = runCatching {
                    llm?.reset() ?: error("NPU model is not loaded")
                }
                result.fold(
                    onSuccess = { sendResponse(replyTo, requestId, true, "Context reset") },
                    onFailure = { sendResponse(replyTo, requestId, false, it.message ?: it.toString()) },
                )
            }

            MSG_UNLOAD -> scope.launch {
                generationJob?.cancel()
                val result = runCatching {
                    llm?.stopStream()
                    llm?.destroy()
                    llm = null
                }
                result.fold(
                    onSuccess = { sendResponse(replyTo, requestId, true, "NPU model unloaded") },
                    onFailure = { sendResponse(replyTo, requestId, false, it.message ?: it.toString()) },
                )
            }

            else -> sendResponse(replyTo, requestId, false, "Unknown NPU worker command")
        }
    }

    private suspend fun loadModel(
        modelPath: String,
        tokenizerPath: String?,
        nCtx: Int,
        nThreads: Int,
        nGpuLayers: Int,
    ) {
        llm?.destroy()
        llm = null

        val backendRoot = prepareHexagonRuntime()
        val plugin = File(backendRoot, "llama_cpp/libgeniex_plugin.so")
        check(plugin.isFile) { "Hexagon llama_cpp plugin missing: ${plugin.absolutePath}" }

        // plugin.cpp reads GENIEX_PLUGIN_PATH in its constructor, then loads
        // GGML backends from its llama_cpp subdirectory and points FastRPC at
        // that directory for the matching HTP skel.
        Os.setenv("GENIEX_PLUGIN_PATH", backendRoot.absolutePath, true)
        val htpSearchPath = listOf(
            backendDir.absolutePath,
            "/vendor/lib/rfsa/adsp",
            "/vendor/dsp/cdsp",
            "/dsp",
        ).joinToString(";")
        Os.setenv(
            "LD_LIBRARY_PATH",
            "${backendDir.absolutePath}:${applicationInfo.nativeLibraryDir}:/system/lib64:/vendor/lib64",
            true,
        )
        // FastRPC resolves the HTP skeleton by basename (libggml-htp-v73.so).
        // Keep the app-private skeleton directory first, then platform fallback paths.
        Os.setenv("ADSP_LIBRARY_PATH", htpSearchPath, true)
        Os.setenv("DSP_LIBRARY_PATH", htpSearchPath, true)
        Log.i(TAG, "NPU_WORKER_DSP_PATH=$htpSearchPath")

        if (!pluginRegistered) {
            val registerResult = GenieXSdk.getInstance().registerPlugin(plugin.absolutePath)
            check(registerResult == 0) { "Hexagon plugin registration failed (rc=$registerResult)" }
            pluginRegistered = true
        }

        val input = LlmCreateInput(
            model_path = modelPath,
            tokenizer_path = tokenizerPath,
            config = ModelConfig(
                nCtx = nCtx,
                nThreads = nThreads,
                nGpuLayers = nGpuLayers,
            ),
            runtime_id = "llama_cpp",
            compute_unit = ComputeUnitValue.NPU.value,
        )
        llm = LlmWrapper.builder().llmCreateInput(input).build().getOrElse {
            throw IllegalStateException("GGML Hexagon model creation failed: ${it.message}", it)
        }

        Log.i(TAG, "NPU_WORKER_LOAD_SUCCESS device=npu plugin=llama_cpp model=$modelPath")
    }

    private fun prepareHexagonRuntime(): File {
        val root = File(filesDir, "npu_runtime")
        val backendDir = File(root, "llama_cpp")
        check(backendDir.mkdirs() || backendDir.isDirectory) {
            "Unable to create Hexagon runtime directory"
        }

        val assetDir = "npu_runtime/llama_cpp"
        val names = assets.list(assetDir)?.filter { it.endsWith(".so") }.orEmpty()
        check(names.contains("libgeniex_plugin.so")) {
            "Hexagon runtime assets are missing; rebuild the experimental APK"
        }
        check(names.contains("libggml-hexagon.so")) {
            "libggml-hexagon.so is missing from the experimental APK"
        }

        names.forEach { name ->
            val destination = File(backendDir, name)
            assets.open("$assetDir/$name").use { input ->
                FileOutputStream(destination).use { output -> input.copyTo(output) }
            }
        }

        Log.i(TAG, "NPU_WORKER_RUNTIME path=${backendDir.absolutePath} libraries=${names.size}")
        return root
    }

    private suspend fun generate(data: Bundle, replyTo: Messenger, requestId: Int) {
        try {
            val wrapper = llm ?: error("NPU model is not loaded")
            val roles = data.getStringArrayList(KEY_ROLES).orEmpty()
            val contents = data.getStringArrayList(KEY_CONTENTS).orEmpty()
            check(roles.size == contents.size && roles.isNotEmpty()) {
                "Chat history is empty or malformed"
            }

            val messages = Array(roles.size) { index ->
                ChatMessage(role = roles[index], content = contents[index])
            }
            val template = wrapper.applyChatTemplate(
                messages = messages,
                tools = data.getString(KEY_TOOLS),
                enableThinking = data.getBoolean(KEY_THINKING, false),
            ).getOrElse {
                throw IllegalStateException("Chat template failed: ${it.message}", it)
            }

            wrapper.generateStreamFlow(
                template.formattedText,
                config = GenerationConfigSample().toGenerationConfig(),
            ).collect { result ->
                when (result) {
                    is LlmStreamResult.Token -> sendToken(replyTo, requestId, result.text)
                    is LlmStreamResult.Completed -> sendCompleted(replyTo, requestId, result.profile)
                    is LlmStreamResult.Error -> sendError(replyTo, requestId, result.throwable.message ?: result.throwable.toString())
                }
            }
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: Throwable) {
            Log.e(TAG, "NPU generation failed", error)
            sendError(replyTo, requestId, error.message ?: error.toString())
        }
    }

    private fun sendResponse(replyTo: Messenger, requestId: Int, ok: Boolean, detail: String) {
        val response = Message.obtain(null, MSG_RESPONSE).apply {
            arg1 = requestId
            data = Bundle().apply {
                putBoolean(KEY_OK, ok)
                putString(KEY_DETAIL, detail)
            }
        }
        runCatching { replyTo.send(response) }
    }

    private fun sendToken(replyTo: Messenger, requestId: Int, token: String) {
        val response = Message.obtain(null, MSG_TOKEN).apply {
            arg1 = requestId
            data = Bundle().apply { putString(KEY_TOKEN, token) }
        }
        runCatching { replyTo.send(response) }
    }

    private fun sendCompleted(replyTo: Messenger, requestId: Int, profile: ProfilingData) {
        val response = Message.obtain(null, MSG_COMPLETE).apply {
            arg1 = requestId
            data = Bundle().apply {
                putDouble(KEY_TTFT, profile.ttftMs)
                putDouble(KEY_MEDIA_MS, profile.mediaMs)
                putDouble(KEY_PROMPT_MS, profile.promptTimeMs)
                putDouble(KEY_DECODE_MS, profile.decodeTimeMs)
                putLong(KEY_PROMPT_TOKENS, profile.promptTokens)
                putLong(KEY_GENERATED_TOKENS, profile.generatedTokens)
                putDouble(KEY_PREFILL_SPEED, profile.prefillSpeed)
                putDouble(KEY_DECODE_SPEED, profile.decodingSpeed)
                putLong(KEY_DRAFT_TOTAL, profile.draftNTotal)
                putLong(KEY_DRAFT_ACCEPTED, profile.draftNAccepted)
                putString(KEY_STOP_REASON, profile.stopReason)
            }
        }
        runCatching { replyTo.send(response) }
    }

    private fun sendError(replyTo: Messenger, requestId: Int, detail: String) {
        val response = Message.obtain(null, MSG_ERROR).apply {
            arg1 = requestId
            data = Bundle().apply { putString(KEY_DETAIL, detail) }
        }
        runCatching { replyTo.send(response) }
    }

    override fun onDestroy() {
        generationJob?.cancel()
        runCatching { llm?.destroy() }
        llm = null
        scope.cancel()
        super.onDestroy()
    }

    companion object {
        private const val TAG = "GenieXNpuWorker"

        const val MSG_LOAD = 1
        const val MSG_GENERATE = 2
        const val MSG_STOP = 3
        const val MSG_RESET = 4
        const val MSG_UNLOAD = 5

        const val MSG_RESPONSE = 100
        const val MSG_TOKEN = 101
        const val MSG_COMPLETE = 102
        const val MSG_ERROR = 103

        const val KEY_MODEL_PATH = "model_path"
        const val KEY_TOKENIZER_PATH = "tokenizer_path"
        const val KEY_N_CTX = "n_ctx"
        const val KEY_N_THREADS = "n_threads"
        const val KEY_GPU_LAYERS = "n_gpu_layers"
        const val KEY_ROLES = "roles"
        const val KEY_CONTENTS = "contents"
        const val KEY_TOOLS = "tools"
        const val KEY_THINKING = "thinking"
        const val KEY_OK = "ok"
        const val KEY_DETAIL = "detail"
        const val KEY_TOKEN = "token"
        const val KEY_TTFT = "ttft"
        const val KEY_MEDIA_MS = "media_ms"
        const val KEY_PROMPT_MS = "prompt_ms"
        const val KEY_DECODE_MS = "decode_ms"
        const val KEY_PROMPT_TOKENS = "prompt_tokens"
        const val KEY_GENERATED_TOKENS = "generated_tokens"
        const val KEY_PREFILL_SPEED = "prefill_speed"
        const val KEY_DECODE_SPEED = "decode_speed"
        const val KEY_DRAFT_TOTAL = "draft_total"
        const val KEY_DRAFT_ACCEPTED = "draft_accepted"
        const val KEY_STOP_REASON = "stop_reason"
    }
}
