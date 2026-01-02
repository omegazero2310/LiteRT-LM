# LiteRT-LM C# Bindings

C# bindings for the LiteRT-LM library, providing on-device language model inference with a native C# API.

## Overview

This package provides C# bindings for LiteRT-LM through P/Invoke, allowing you to run language models on-device in .NET applications. The API is designed to be idiomatic C# while maintaining feature parity with the Java/JNI implementation.

## Features

- **Engine Management**: Create and manage model engines with various backends (CPU, GPU, NPU)
- **Session-based Generation**: Simple text generation with prefill/decode phases
- **Conversation API**: Multi-turn conversations with system instructions and tool calling
- **Streaming Support**: Real-time token generation with callbacks
- **Multimodal Input**: Support for text, image, and audio inputs
- **Benchmark Tools**: Performance measurement and profiling
- **Cross-platform**: Windows, Linux, and macOS support

## Requirements

- .NET 6.0 or later
- Bazel (for building)
- C++17 compatible compiler
- LiteRT-LM C++ library

## Building

### 1. Add to WORKSPACE

Add the following to your `WORKSPACE` file:

```python
http_archive(
    name = "rules_dotnet",
    sha256 = "...",
    strip_prefix = "rules_dotnet-0.15.1",
    urls = ["https://github.com/bazelbuild/rules_dotnet/archive/v0.15.1.tar.gz"],
)

load("@rules_dotnet//dotnet:repositories.bzl", "dotnet_register_toolchains", "rules_dotnet_dependencies")

rules_dotnet_dependencies()
dotnet_register_toolchains(name = "dotnet", dotnet_version = "8.0.100")
```

### 2. Build the Library

> Note: In order to run on GPU on all platforms, we need to take extra steps:
>
> 1. Add `--define=litert_link_capi_so=true`
  `--define=resolve_symbols_in_exec=false` in the build command.
> 2. `mkdir -p <test_dir>; cp <your litert_lm_main> <test_dir>; cp ./prebuilt/<your OS>/<shared libaries> <test_dir>/`
 and make sure the prebuilt .so/.dll/.dylib files are in the same directory as
  `litert_lm_main` binary
> 3. Running GPU on Windows needs DirectXShaderCompiler. See
 [this Note](#windows_gpu) for more details.


```bash
# Build the C++ native library
bazel build //csharp:litertlm_csharp

# Build the C# library
bazel build //csharp:LiteRtLm

# Run examples
bazel run //csharp:LiteRtLm_Example

# If the example program throw exception (dll not found) do this
# Install the tool
sudo apt update && sudo apt install patchelf

# Set the search path (RPATH) for the main library
# The single quotes around $ORIGIN are CRITICAL so the shell doesn't change it
patchelf --set-rpath '$ORIGIN' liblitertlm_csharp.so

# Now verify it with ldd
ldd liblitertlm_csharp.so
```

### 3. Using in Your Project

Reference the built assembly in your C# project:

```xml
<ItemGroup>
  <Reference Include="LiteRtLm">
    <HintPath>path/to/bazel-bin/csharp/LiteRtLm.dll</HintPath>
  </Reference>
</ItemGroup>
```

Make sure the native library (`litertlm_csharp.dll`/`.so`/`.dylib`) is in your application's runtime path.

## Quick Start

### Simple Text Generation

```csharp
using Google.AI.Edge.LiteRtLm;

// Create an engine
using var engine = new Engine(
    modelPath: "path/to/model.tflite",
    backend: "cpu",
    maxNumTokens: 2048);

// Create a session with sampler config
using var session = engine.CreateSession(
    new SamplerConfig
    {
        TopK = 40,
        TopP = 0.95,
        Temperature = 0.8
    });

// Generate content
var prompt = new InputData.Text("What is the capital of France?");
string response = session.GenerateContent(prompt);
Console.WriteLine(response);
```

### Streaming Generation

```csharp
var prompt = new InputData.Text("Write a short story.");

session.GenerateContentStream(
    inputs: new[] { prompt },
    onResponse: (token) => Console.Write(token),
    onComplete: () => Console.WriteLine("\nDone!"),
    onError: (code, msg) => Console.WriteLine($"Error: {msg}"));
```

### Multi-turn Conversation

```csharp
string systemInstruction = @"{
    ""role"": ""system"",
    ""content"": ""You are a helpful assistant.""
}";

using var conversation = engine.CreateConversation(
    systemMessageJson: systemInstruction);

// Send messages
string userMsg = @"{
    ""role"": ""user"",
    ""content"": ""Hello!""
}";

string response = conversation.SendMessage(userMsg);
Console.WriteLine(response);
```

### Multimodal Input

```csharp
using var engine = new Engine(
    modelPath: "path/to/multimodal_model.tflite",
    visionBackend: "gpu");

using var session = engine.CreateSession();

byte[] imageData = File.ReadAllBytes("image.jpg");

var inputs = new InputData[]
{
    new InputData.Image(imageData),
    new InputData.Text("What's in this image?")
};

string response = session.GenerateContent(inputs);
```

## API Reference

### Engine

The main entry point for creating model instances.

```csharp
public class Engine : IDisposable
{
    public Engine(
        string modelPath,
        string backend = "cpu",
        string visionBackend = null,
        string audioBackend = null,
        int maxNumTokens = 0,
        string cacheDir = null,
        bool enableBenchmark = false);

    public Session CreateSession(SamplerConfig samplerConfig = null);
    
    public Conversation CreateConversation(
        SamplerConfig samplerConfig = null,
        string systemMessageJson = null,
        string toolsJson = null,
        bool enableConstrainedDecoding = false);

    public static void SetMinLogSeverity(LogSeverity severity);
}
```

**Backends**: `"cpu"`, `"gpu"`, `"npu"`

### Session

For single-turn or manual generation control.

```csharp
public class Session : IDisposable
{
    public void RunPrefill(params InputData[] inputs);
    public string RunDecode();
    public string GenerateContent(params InputData[] inputs);
    
    public void GenerateContentStream(
        InputData[] inputs,
        Action<string> onResponse,
        Action onComplete = null,
        Action<int, string> onError = null);

    public void CancelProcess();
}
```

### Conversation

For multi-turn conversations with context.

```csharp
public class Conversation : IDisposable
{
    public string SendMessage(string messageJson);
    
    public void SendMessageAsync(
        string messageJson,
        Action<string> onMessage,
        Action onComplete = null,
        Action<int, string> onError = null);

    public void CancelProcess();
    public BenchmarkInfo GetBenchmarkInfo();
}
```

### InputData

Polymorphic input types for different modalities.

```csharp
public abstract class InputData
{
    public class Text : InputData
    {
        public Text(string content);
    }

    public class Audio : InputData
    {
        public Audio(byte[] data);
    }

    public class Image : InputData
    {
        public Image(byte[] data);
    }
}
```

### SamplerConfig

Configuration for text generation sampling.

```csharp
public class SamplerConfig
{
    public int TopK { get; set; } = 40;
    public double TopP { get; set; } = 0.95;
    public double Temperature { get; set; } = 0.8;
    public int Seed { get; set; } = 0;
}
```

### LogSeverity

```csharp
public enum LogSeverity
{
    Verbose = 0,
    Debug = 1,
    Info = 2,
    Warning = 3,
    Error = 4,
    Fatal = 5,
    Silent = 6
}
```

## Error Handling

All methods that can fail will throw `LiteRtLmException`:

```csharp
try
{
    using var engine = new Engine("model.tflite");
    // ... use engine
}
catch (LiteRtLmException ex)
{
    Console.WriteLine($"LiteRT-LM error (code {ex.StatusCode}): {ex.Message}");
}
```

## Performance Tips

1. **Reuse Engine and Session**: Creating these is expensive, reuse them when possible
2. **Enable Benchmarking**: Set `enableBenchmark: true` to measure performance
3. **Use Appropriate Backend**: GPU for large models, CPU for small/mobile models
4. **Cache Directory**: Set `cacheDir` to speed up repeated loads
5. **Manage Token Limits**: Set `maxNumTokens` to control memory usage

## Thread Safety

- `Engine` instances are thread-safe for creating sessions/conversations
- `Session` and `Conversation` instances are **not** thread-safe
- Each thread should have its own `Session` or `Conversation` instance
- Callbacks are invoked on background threads, marshal to UI thread if needed

## Memory Management

All classes implement `IDisposable`. Always use `using` statements or call `Dispose()`:

```csharp
using (var engine = new Engine("model.tflite"))
using (var session = engine.CreateSession())
{
    // Use session
} // Automatically disposed
```

## Platform-Specific Notes

### Windows
- The native library is named `litertlm_csharp.dll`
- Place it in the same directory as your executable or in the system PATH

### Linux
- The native library is named `litertlm_csharp.so`
- Set `LD_LIBRARY_PATH` if not in a standard location

### macOS
- The native library is named `litertlm_csharp.dylib`
- Set `DYLD_LIBRARY_PATH` if needed

## Examples

See `Example.cs` for comprehensive examples including:
- Simple text generation
- Streaming generation
- Multi-turn conversations
- Function calling with tools
- Multimodal inputs
- Manual prefill/decode control
- Cancellation handling

## Troubleshooting

### DllNotFoundException

If you get `DllNotFoundException`, ensure:
1. The native library is built: `bazel build //csharp:litertlm_csharp`
2. It's in your application's directory or system library path
3. All dependencies (TensorFlow Lite, etc.) are available

### Memory Access Violations

- Ensure you're not using disposed objects
- Don't access sessions/conversations from multiple threads
- Keep callback delegates alive during async operations (use `GC.KeepAlive`)

### Model Loading Errors

- Verify the model file exists and is readable
- Check the model is compatible with LiteRT-LM
- Ensure sufficient memory is available

## Contributing

Contributions are welcome! Please ensure:
- Code follows C# naming conventions
- All public APIs are documented
- Error handling is consistent
- Memory is properly managed

## License

Licensed under the Apache License, Version 2.0. See LICENSE for details.

## Related Projects

- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) - Main C++ library
- [MediaPipe](https://github.com/google/mediapipe) - ML inference framework
- [TensorFlow Lite](https://www.tensorflow.org/lite) - Lightweight ML framework
