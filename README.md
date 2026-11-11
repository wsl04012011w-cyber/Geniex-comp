[![Qualcomm® AI Hub Apps](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](https://aihub.qualcomm.com)

# GenieX Demo

On-device AI chat application for Android on Snapdragon® powered by the
[GenieX SDK](https://github.com/qualcomm/geniex). Runs Large Language Models
(LLMs) and Vision-Language Models (VLMs) on the Snapdragon® Neural Processing
Unit (NPU), GPU, or CPU through a single pluggable runtime.

Unlike most AI Hub apps, the model weights are **not** bundled into the APK.
The app downloads the model you select at runtime from an in-app catalog, so
no model export or asset copying is required to build and run it.

We recommend using a device from [QDC](https://qdc.qualcomm.com/) for this
demo.

## Get the App

### Option A: Using the CLI (Recommended)

Install the CLI and fetch the app source:

```bash
pip install qai-hub-apps
qai-hub-apps fetch geniex_chat_android --output-dir ~
cd ~/geniex_chat_android
```

> [!NOTE]
> `--model` is **not** required for this app. GenieX downloads the model you
> pick from the in-app catalog at runtime, so no model binaries need to be
> placed in the project before building.

### Option B: Cloning the Repo

If you cloned the repository, the app directory under
`apps/geniex_chat_android` is already self-contained — there are no model
weights to fetch separately.

## Prerequisites

1. Download [Android Studio](https://developer.android.com/studio).
   **Version 2024.3.1 or newer** is required.
2. [Enable USB debugging](https://developer.android.com/studio/debug/dev-options) on your Android device.

This fork builds a custom **GenieX v0.10.0 Android AAR** targeting baseline
ARMv8.0 with the llama.cpp **OpenCL GPU backend** enabled. Hexagon and QAIRT
are excluded, so GGUF models can use CPU or OpenCL GPU without initializing
the Hexagon backend that crashed on the original device. GitHub Actions builds
the SDK and AAR from the pinned upstream source. The GPU path is still
**device/driver-dependent**: Adreno 710 is not listed among the Android GPUs
verified by upstream llama.cpp, so runtime support must be confirmed on the
phone. Select CPU if OpenCL initialization or inference fails.

## Build the APK

1. Open this folder in Android Studio.
2. Run Gradle sync.
3. Build the `app` target (`Build` → `Build Bundle(s) / APK(s)` → `Build APK(s)`).
4. The APK is written to:

   ```text
   build/outputs/apk/{build_type}/app-{build_type}.apk
   ```

## Run on Device

1. Install the APK via `adb`:

   ```bash
   adb install -t build/outputs/apk/debug/app-debug.apk
   ```

2. Launch the app from the device's launcher. Models are downloaded on first
   use; pick one from the in-app catalog.

## Supported Hardware

- **CPU**: ARM64-v8a baseline
- **GPU**: Adreno through OpenCL when the Android driver exposes a compatible implementation
- **NPU**: not included in this baseline OpenCL build

## Technologies Used by this App

- [Android SDK](https://developer.android.com/studio)
- [GenieX SDK](https://github.com/qualcomm/geniex) (Maven Central:
  `com.qualcomm.qti:geniex-android`)

## License

This app is released under the [BSD-3 License](LICENSE) found at the
root of this repository.

The GenieX SDK dependency is released under its own license. Refer to the
[GenieX repository](https://github.com/qualcomm/geniex) for details.


## Experimental Hexagon NPU build

The `experiment/localdream-npu-isolation` branch keeps two different native runtimes:
the app process uses the baseline ARMv8-A CPU/OpenCL AAR without Hexagon, while
the GGML Hexagon plugin and its backend libraries are packaged as private assets.
When a text GGUF model is explicitly loaded with **NPU (experimental)**, the app
binds to `NpuWorkerService` in Android process `:npu`. That worker extracts the
private runtime, sets `GENIEX_PLUGIN_PATH`, registers the Hexagon-enabled
`llama_cpp` plugin, and performs model loading and token generation away from
the UI process. Tokens and profiling results return over Messenger IPC.

The worker process boundary is intended to contain a native Hexagon crash; it
does not guarantee that the HTP driver or runtime will initialize on every
device. The first runtime target is the Galaxy A36 / Snapdragon 6 Gen 3
(Hexagon V73). NPU selection is experimental and limited to text GGUF models.
CPU and OpenCL remain the default/supported fallback paths. A successful APK
build verifies packaging only; on-device inference still needs testing.
