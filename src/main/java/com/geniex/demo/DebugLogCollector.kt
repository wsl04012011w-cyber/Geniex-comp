// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.geniex.demo

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import java.util.Date
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Small rolling logcat recorder. The active and previous session logs live in
 * app-private storage so they survive a process crash and can be exported later.
 */
object DebugLogCollector {
    private const val TAG = "GenieXDebug"
    private const val MAX_FILE_BYTES = 4L * 1024L * 1024L
    private val started = AtomicBoolean(false)
    private val lock = Any()
    @Volatile private var writer: BufferedWriter? = null
    @Volatile private var logFile: File? = null
    @Volatile private var backupFile: File? = null

    fun start(context: Context) {
        if (!started.compareAndSet(false, true)) return
        val dir = File(context.filesDir, "diagnostics").apply { mkdirs() }
        logFile = File(dir, "geniex-debug.log")
        backupFile = File(dir, "geniex-debug-previous.log")

        synchronized(lock) {
            val active = logFile!!
            if (active.exists() && active.length() > MAX_FILE_BYTES) {
                backupFile!!.delete()
                active.renameTo(backupFile!!)
            }
            openWriter()
            writeLocked("\n===== APP SESSION START Tue Nov 10 2026 23:24:56 GMT+0000 (Coordinated Universal Time) =====\n")
        }

        installCrashHandler()
        record("RECORDER_STARTED sdk=${Build.VERSION.SDK_INT} device=${Build.MANUFACTURER} ${Build.MODEL} fingerprint=${Build.FINGERPRINT}")

        Thread({
            try {
                // The Android log buffers also retain the immediately preceding
                // process's fatal Java/native crash records until they are evicted.
                val process = ProcessBuilder("logcat", "-b", "all", "-v", "threadtime")
                    .redirectErrorStream(true)
                    .start()
                process.inputStream.bufferedReader().useLines { lines ->
                    lines.forEach { line ->
                        synchronized(lock) {
                            if ((logFile?.length() ?: 0L) > MAX_FILE_BYTES) {
                                backupFile?.delete()
                                logFile?.renameTo(backupFile!!)
                                writer?.close()
                                openWriter()
                                writeLocked("\n===== LOG ROTATED ${Date()} =====\n")
                            }
                            writeLocked(line + "\n")
                        }
                    }
                }
                record("logcat reader ended exit=${process.waitFor()}")
            } catch (t: Throwable) {
                record("logcat capture unavailable: ${Log.getStackTraceString(t)}")
            }
        }, "GenieX-logcat-recorder").apply { isDaemon = true; start() }
    }

    fun record(message: String) {
        Log.i(TAG, message)
    }

    private fun installCrashHandler() {
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, error ->
            synchronized(lock) {
                writeLocked("\n===== UNCAUGHT EXCEPTION thread=${thread.name} time=${Date()} =====\n")
                writeLocked(Log.getStackTraceString(error) + "\n")
            }
            try { previous?.uncaughtException(thread, error) } catch (_: Throwable) { }
        }
    }

    private fun openWriter() {
        val file = logFile ?: return
        writer = BufferedWriter(OutputStreamWriter(FileOutputStream(file, true), Charsets.UTF_8))
    }

    private fun writeLocked(value: String) {
        try {
            writer?.apply { write(value); flush() }
        } catch (t: Throwable) {
            Log.e(TAG, "Unable to persist diagnostic log", t)
        }
    }

    /** Writes a single diagnostic text file to the public Downloads collection. */
    fun exportToDownloads(context: Context): String {
        val resolver = context.contentResolver
        val name = "GenieX-debug-${System.currentTimeMillis()}.txt"
        val values = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, name)
            put(MediaStore.Downloads.MIME_TYPE, "text/plain")
            put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
            put(MediaStore.Downloads.IS_PENDING, 1)
        }
        val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            ?: error("Android não criou o arquivo em Downloads")

        try {
            resolver.openOutputStream(uri)?.bufferedWriter(Charsets.UTF_8)?.use { out ->
                out.appendLine("GenieX Android diagnostic report")
                out.appendLine("Exported: ${Date()}")
                out.appendLine("Device: ${Build.MANUFACTURER} ${Build.MODEL}")
                out.appendLine("Android: ${Build.VERSION.RELEASE} (SDK ${Build.VERSION.SDK_INT})")
                out.appendLine("App version: ${runCatching {
                    context.packageManager.getPackageInfo(context.packageName, 0).versionName
                }.getOrNull() ?: "unknown"}")
                out.appendLine()
                out.appendLine("===== RECENT ANDROID LOGCAT SNAPSHOT (last 5000 lines) =====")
                val snapshot = runCatching {
                    val p = ProcessBuilder("logcat", "-d", "-b", "all", "-v", "threadtime", "-t", "5000")
                        .redirectErrorStream(true).start()
                    val content = p.inputStream.bufferedReader().use { it.readText() }
                    p.waitFor()
                    content
                }.getOrElse { "Could not read logcat snapshot: ${Log.getStackTraceString(it)}" }
                out.append(snapshot)
                out.appendLine()
                synchronized(lock) {
                    backupFile?.takeIf { it.exists() }?.let {
                        out.appendLine("===== PREVIOUS APP SESSION (rotated) =====")
                        it.bufferedReader().useLines { lines -> lines.forEach(out::appendLine) }
                    }
                    logFile?.takeIf { it.exists() }?.let {
                        out.appendLine("===== PERSISTED APP LOG =====")
                        it.bufferedReader().useLines { lines -> lines.forEach(out::appendLine) }
                    }
                }
            } ?: error("Não foi possível escrever o arquivo de diagnóstico")
            val done = ContentValues().apply { put(MediaStore.Downloads.IS_PENDING, 0) }
            resolver.update(uri, done, null, null)
            return name
        } catch (t: Throwable) {
            resolver.delete(uri, null, null)
            throw t
        }
    }
}
