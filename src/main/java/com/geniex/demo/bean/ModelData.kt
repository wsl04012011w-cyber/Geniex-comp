// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.geniex.demo.bean

import android.annotation.SuppressLint
import kotlinx.serialization.Serializable

/**
 * Demo-side description of a model that the Rust model manager can
 * download and resolve. Only the fields needed by `geniex_model_pull`
 * and the UI live here — file paths come from `ModelManagerWrapper
 * .getPaths(modelName)` after the pull completes.
 */
@SuppressLint("UnsafeOptInUsageError")
@Serializable
data class ModelData(
    /** Stable UI key (spinner selection, SharedPreferences). */
    val id: String,
    /** Human-readable label shown in the UI. */
    val displayName: String,
    /**
     * `org/repo` as the Rust hub expects. Not an alias — keep the
     * alias layer (`resolveAlias`) out of the demo for simplicity.
     */
    val modelName: String,
    /** "chat" / "llm" / "vlm". */
    val type: String? = null,
    /**
     * Runtime that owns this model: `"llama_cpp"` for GGUF models or
     * `"qairt"` for NPU-only AI Hub bundles. This build exposes llama.cpp
     * CPU/OpenCL and an experimental isolated GGML Hexagon worker for text GGUF.
     */
    val runtime: String? = null,
    /** Quantization hint passed to `geniex_model_pull`. */
    val quant: String? = null,
    /**
     * Hub name — matches [com.geniex.sdk.bean.HubSource] by enum name.
     * Default `AUTO` lets Rust pick based on `modelName`.
     */
    val hub: String? = "AUTO",
    /** AI Hub display_name; required when [hub] is `AIHUB`. */
    val aiHubDisplayName: String? = null,
    /**
     * Target chipset for AI Hub pulls (e.g. "SM8650"). On Android the
     * Rust side has no auto-detect, so `aiHubDisplayName` entries must
     * pair with an explicit `chipset`.
     */
    val chipset: String? = null,
)

/** Text GGUF models can opt into the experimental isolated Hexagon worker. */
fun ModelData.getSupportPluginIds(): ArrayList<String> =
    if (type == "chat" || type == "llm") {
        arrayListOf("cpu", "gpu", "npu")
    } else {
        arrayListOf("cpu", "gpu")
    }

/** True when the model runs on the QAIRT NPU runtime. */
fun ModelData.isNpuModel(): Boolean = runtime == "qairt"
