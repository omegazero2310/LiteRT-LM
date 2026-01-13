# LiteRT-LM C# Bindings

C# bindings for the LiteRT-LM library, providing on-device language model inference with a native C# API.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Windows Setup](#windows-setup)
  - [Linux Setup](#linux-setup)
  - [macOS Setup](#macos-setup)
- [Building](#building)
- [Quick Start](#quick-start)
- [Complete Examples](#complete-examples)
- [API Reference](#api-reference)
- [Platform-Specific Notes](#platform-specific-notes)
- [Troubleshooting](#troubleshooting)
- [Performance Tips](#performance-tips)

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

- **.NET**: 6.0 or later
- **Bazelisk**: Bazel version manager
- **C++ Compiler**: C++17 compatible
- **LiteRT-LM**: C++ library (included in build)

### Platform-Specific Requirements

**Windows:**
- Visual Studio Build Tools 2022 with VC++ tools
- Git for Windows

**Linux:**
- GCC 7+ or Clang 5+
- Build essentials

**macOS:**
- Xcode Command Line Tools

## Installation

### Windows Setup

#### Step 1: Install Chocolatey (Package Manager)

Open PowerShell as Administrator and run:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

#### Step 2: Install Required Tools

```powershell
# Install Git and Python
choco install git python3 -y

# Install Visual Studio Build Tools with VC++ components
choco install visualstudio2022buildtools -y
choco install visualstudio2022-workload-vctools -y

# Install Bazelisk
choco install bazelisk -y
```

**Alternative: Manual Installation**

If you prefer manual installation:

1. **Bazelisk**: Download from [GitHub releases](https://github.com/bazelbuild/bazelisk/releases)
   - Download `bazelisk-windows-amd64.exe`
   - Rename to `bazel.exe`
   - Add to your PATH

2. **Visual Studio Build Tools**: Download from [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/)
   - Run installer
   - Select "Desktop development with C++"
   - Ensure "MSVC v143 - VS 2022 C++ x64/x86 build tools" is checked
   - Install

3. **Git**: Download from [git-scm.com](https://git-scm.com/download/win)

#### Step 3: Configure Environment Variables

Open PowerShell and run:

```powershell
# Set BAZEL_VC to Visual Studio VC folder
[System.Environment]::SetEnvironmentVariable("BAZEL_VC", "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC", "User")

# Set BAZEL_SH to Git bash.exe
[System.Environment]::SetEnvironmentVariable("BAZEL_SH", "C:\Program Files\Git\bin\bash.exe", "User")

# Refresh environment variables in current session
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

#### Step 4: Fix Windows Long Path Issue

Windows has a 260-character path length limit that can cause build issues. Create or edit `.bazelrc` in your project root:

```bash
# .bazelrc
# Move Bazel output to shorter path to avoid Windows path length limitations
startup --output_user_root=D:/_bzl
```

**Note:** You can change `D:/_bzl` to any short path on your system (e.g., `C:/_bzl`).

#### Step 5: Verify Installation

```powershell
# Check Bazel
bazel --version

# Check Visual Studio compiler
where cl.exe

# Check Git
git --version

# Check Python
python --version
```

### Linux Setup

#### Ubuntu/Debian

```bash
# Install Bazelisk
sudo wget -O /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
sudo chmod +x /usr/local/bin/bazel

# Install build dependencies
sudo apt-get update
sudo apt-get install build-essential git python3 -y

# Install .NET SDK (if not installed)
wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh
chmod +x dotnet-install.sh
./dotnet-install.sh --channel 8.0
```

#### Fedora/RHEL

```bash
# Install Bazelisk
sudo wget -O /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
sudo chmod +x /usr/local/bin/bazel

# Install build dependencies
sudo dnf install gcc-c++ git python3 -y
```

### macOS Setup

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Bazelisk
brew install bazelisk

# Install Xcode Command Line Tools
xcode-select --install

# Install .NET SDK (if not installed)
brew install --cask dotnet-sdk
```

## Building

### 1. Build the Library

#### CPU Mode (Default)

```bash
# Build the native C++ library
bazel build //csharp:litertlm_csharp

# Build the C# assembly
bazel build //csharp:LiteRtLm

# Run examples
bazel run //csharp:LiteRtLm_Example
```

#### GPU Mode

To enable GPU support, add the `use_gpu` define:

```bash
# Build with GPU support
bazel build --define=use_gpu=true //csharp:litertlm_csharp
bazel build --define=use_gpu=true //csharp:LiteRtLm

# Run with GPU support
bazel run --define=use_gpu=true //csharp:LiteRtLm_Example
```


#### Build Output Location

After building, the compiled libraries will be located at:

```
bazel-bin/csharp/LiteRtLm.dll                      # C# assembly
bazel-bin/csharp/litertlm_csharp.dll (.so/.dylib)  # Native library
```

### 2. Using in Your Project

#### Option 1: Direct Reference

Copy the built files to your project and reference them:

```bash
# Copy libraries to your project
cp bazel-bin/csharp/LiteRtLm.dll ./MyProject/libs/
cp bazel-bin/csharp/litertlm_csharp.dll ./MyProject/libs/
```

Reference in your `.csproj`:

```xml
<ItemGroup>
  <Reference Include="LiteRtLm">
    <HintPath>libs/LiteRtLm.dll</HintPath>
  </Reference>
</ItemGroup>
```

Make sure the native library (`litertlm_csharp.dll`/`.so`/`.dylib`) is in your application's runtime directory or system library path.

#### Option 2: Using from Bazel Output

Reference directly from Bazel output:

```xml
<ItemGroup>
  <Reference Include="LiteRtLm">
    <HintPath>path/to/bazel-bin/csharp/LiteRtLm.dll</HintPath>
  </Reference>
</ItemGroup>
```

## Quick Start

### Simple Text Generation

```csharp
using Google.AI.Edge.LiteRtLm;

// Create an engine
using var engine = new Engine(
    modelPath: "path/to/model.litertlm",
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
using var conversation = engine.CreateConversation(
    samplerConfig: new SamplerConfig { Temperature = 0.7 });

// Send messages in JSON format
string userMsg = @"{""role"":""user"",""content"":""Hello!""}";
string response = conversation.SendMessage(userMsg);
Console.WriteLine(response);
```

### Multimodal Input

```csharp
using var engine = new Engine(
    modelPath: "path/to/multimodal_model.litertlm",
    backend: "cpu",
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

## Complete Examples

### Example 1: Simple Text Generation

```csharp
using System;
using Google.AI.Edge.LiteRtLm;

static void SimpleTextGeneration()
{
    Console.WriteLine("=== Simple Text Generation ===\n");

    try
    {
        using var engine = new Engine(
            modelPath: "D:/models/gemma-3n-E4B-it-int4.litertlm",
            backend: "cpu",
            maxNumTokens: 2048);

        using var session = engine.CreateSession(
            new SamplerConfig
            {
                TopK = 40,
                TopP = 0.95,
                Temperature = 0.8
            });

        var prompt = new InputData.Text("What is the capital of France?");
        string response = session.GenerateContent(prompt);

        Console.WriteLine($"Prompt: {((InputData.Text)prompt).Content}");
        Console.WriteLine($"Response: {response}\n");
    }
    catch (LiteRtLmException ex)
    {
        Console.WriteLine($"Error (code {ex.StatusCode}): {ex.Message}");
    }
}
```

### Example 2: Streaming Generation

```csharp
using System.Threading;

static void StreamingTextGeneration()
{
    Console.WriteLine("=== Streaming Text Generation ===\n");

    try
    {
        using var engine = new Engine(
            modelPath: "D:/models/gemma-3n-E4B-it-int4.litertlm",
            backend: "cpu");

        using var session = engine.CreateSession();

        var prompt = new InputData.Text("Write a short story about a robot.");
        
        Console.WriteLine($"Prompt: {((InputData.Text)prompt).Content}");
        Console.Write("Response: ");

        var completionEvent = new ManualResetEvent(false);

        session.GenerateContentStream(
            inputs: new[] { prompt },
            onResponse: (response) =>
            {
                Console.Write(response);
            },
            onComplete: () =>
            {
                Console.WriteLine("\n[Generation complete]");
                completionEvent.Set();
            },
            onError: (code, message) =>
            {
                Console.WriteLine($"\n[Error {code}: {message}]");
                completionEvent.Set();
            });

        // Wait for completion
        completionEvent.WaitOne();
        Console.WriteLine();
    }
    catch (LiteRtLmException ex)
    {
        Console.WriteLine($"Error (code {ex.StatusCode}): {ex.Message}");
    }
}
```

### Example 3: Multi-turn Conversation

```csharp
static void MultiTurnConversation()
{
    Console.WriteLine("=== Multi-turn Conversation ===\n");

    try
    {
        using var engine = new Engine(
            modelPath: "D:/models/gemma-3n-E4B-it-int4.litertlm",
            backend: "cpu",
            enableBenchmark: true);

        using var conversation = engine.CreateConversation(
            samplerConfig: new SamplerConfig { Temperature = 0.7 });

        // Turn 1
        string userMessage1 = @"{""role"":""user"",""content"":""What is machine learning?""}";

        Console.WriteLine("User: What is machine learning?");
        string response1 = conversation.SendMessage(userMessage1);
        Console.WriteLine($"Assistant: {response1}\n");

        // Turn 2
        string userMessage2 = @"{""role"":""user"",""content"":""Can you give me an example?""}";

        Console.WriteLine("User: Can you give me an example?");
        string response2 = conversation.SendMessage(userMessage2);
        Console.WriteLine($"Assistant: {response2}\n");

        // Get benchmark info
        var benchmarkInfo = conversation.GetBenchmarkInfo();
        Console.WriteLine($"Benchmark Info:");
        Console.WriteLine($"  Prefill tokens: {benchmarkInfo.PrefillTokenCount}");
        Console.WriteLine($"  Decode tokens: {benchmarkInfo.DecodeTokenCount}\n");
    }
    catch (LiteRtLmException ex)
    {
        Console.WriteLine($"Error (code {ex.StatusCode}): {ex.Message}");
    }
}
```

### Example 4: Function Calling with Tools

```csharp
static void ConversationWithTools()
{
    Console.WriteLine("=== Conversation with Tools ===\n");

    try
    {
        using var engine = new Engine(
            modelPath: "D:/models/gemma-3n-E4B-it-int4.litertlm",
            backend: "cpu");

        string toolsJson = @"[
            {
                ""name"": ""get_weather"",
                ""description"": ""Get the current weather for a location"",
                ""parameters"": {
                    ""type"": ""object"",
                    ""properties"": {
                        ""location"": {
                            ""type"": ""string"",
                            ""description"": ""The city and state, e.g. San Francisco, CA""
                        }
                    },
                    ""required"": [""location""]
                }
            }
        ]";

        using var conversation = engine.CreateConversation(
            toolsJson: toolsJson,
            enableConstrainedDecoding: true);

        var completionEvent = new ManualResetEvent(false);

        string userMessage = @"{""role"":""user"",""content"":""What's the weather like in New York?""}";

        Console.WriteLine("User: What's the weather like in New York?");
        Console.Write("Assistant: ");

        conversation.SendMessageAsync(
            messageJson: userMessage,
            onMessage: (message) =>
            {
                Console.WriteLine($"Received message chunk: {message}");
            },
            onComplete: () =>
            {
                Console.WriteLine("\n[Conversation turn complete]");
                completionEvent.Set();
            },
            onError: (code, message) =>
            {
                Console.WriteLine($"\n[Error {code}: {message}]");
                completionEvent.Set();
            });

        completionEvent.WaitOne();
        Console.WriteLine();
    }
    catch (LiteRtLmException ex)
    {
        Console.WriteLine($"Error (code {ex.StatusCode}): {ex.Message}");
    }
}
```

### Example 5: Manual Prefill and Decode

```csharp
static void ManualPrefillDecode()
{
    Console.WriteLine("=== Manual Prefill and Decode ===\n");

    try
    {
        using var engine = new Engine(
            modelPath: "D:/models/gemma-3n-E4B-it-int4.litertlm",
            backend: "cpu");

        using var session = engine.CreateSession();

        // Prefill phase
        var prompt = new InputData.Text("The quick brown fox");
        session.RunPrefill(prompt);
        Console.WriteLine($"Prefilled with: {((InputData.Text)prompt).Content}");

        // Decode phase - generate one token at a time
        Console.Write("Generated tokens: ");
        for (int i = 0; i < 10; i++)
        {
            string token = session.RunDecode();
            Console.Write(token);
            
            // Add stopping conditions
            if (token.Contains("."))
            {
                break;
            }
        }
        Console.WriteLine("\n");
    }
    catch (LiteRtLmException ex)
    {
        Console.WriteLine($"Error (code {ex.StatusCode}): {ex.Message}");
    }
}
```

### Example 6: Cancelling Generation

```csharp
using System.Threading;

static void CancelGeneration()
{
    Console.WriteLine("=== Cancel Generation ===\n");

    try
    {
        using var engine = new Engine(
            modelPath: "D:/models/gemma-3n-E4B-it-int4.litertlm",
            backend: "cpu");

        using var session = engine.CreateSession();

        var cancelEvent = new ManualResetEvent(false);

        var prompt = new InputData.Text("Write a very long story about space exploration.");

        // Start generation
        session.GenerateContentStream(
            inputs: new[] { prompt },
            onResponse: (response) =>
            {
                Console.Write(response);
            },
            onComplete: () =>
            {
                Console.WriteLine("\n[Generation complete]");
                cancelEvent.Set();
            },
            onError: (code, message) =>
            {
                Console.WriteLine($"\n[Error or cancelled: {message}]");
                cancelEvent.Set();
            });

        // Cancel after 2 seconds
        Thread.Sleep(2000);
        Console.WriteLine("\n[Cancelling generation...]");
        session.CancelProcess();

        cancelEvent.WaitOne();
        Console.WriteLine();
    }
    catch (LiteRtLmException ex)
    {
        Console.WriteLine($"Error (code {ex.StatusCode}): {ex.Message}");
    }
}
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

**Parameters:**
- `modelPath`: Path to the `.litertlm` model file
- `backend`: Inference backend - `"cpu"`, `"gpu"`, or `"npu"`
- `visionBackend`: Backend for vision models (optional)
- `audioBackend`: Backend for audio models (optional)
- `maxNumTokens`: Maximum number of tokens to generate (0 = no limit)
- `cacheDir`: Directory to cache compiled models
- `enableBenchmark`: Enable performance benchmarking

### Session

For single-turn generation or manual control.

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

**Methods:**
- `RunPrefill`: Process input and prepare for generation
- `RunDecode`: Generate one token at a time
- `GenerateContent`: Generate complete response (blocking)
- `GenerateContentStream`: Generate response with streaming callbacks
- `CancelProcess`: Cancel ongoing generation

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

**Message Format (JSON):**
```json
{
    "role": "user",
    "content": "Your message here"
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
        public string Content { get; }
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

**Parameters:**
- `TopK`: Top-k sampling (40 recommended)
- `TopP`: Nucleus sampling threshold (0.95 recommended)
- `Temperature`: Sampling temperature (0.8 = balanced, lower = more focused)
- `Seed`: Random seed for reproducibility (0 = random)

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

### BenchmarkInfo

Performance metrics for conversation.

```csharp
public class BenchmarkInfo
{
    public int PrefillTokenCount { get; }
    public int DecodeTokenCount { get; }
}
```

## Platform-Specific Notes

### Windows

**Native Library:** `litertlm_csharp.dll`

**Deployment:**
1. Copy `litertlm_csharp.dll` to your application's directory
2. Or add to system PATH
3. Ensure Visual C++ Runtime is installed

**GPU Support:**
- Requires CUDA-compatible GPU
- Install appropriate CUDA toolkit

### Linux

**Native Library:** `litertlm_csharp.so`

**Deployment:**
```bash
# Option 1: Copy to application directory
cp litertlm_csharp.so /path/to/your/app/

# Option 2: Set library path
export LD_LIBRARY_PATH=/path/to/lib:$LD_LIBRARY_PATH

# Option 3: Install system-wide
sudo cp litertlm_csharp.so /usr/local/lib/
sudo ldconfig
```

**GPU Support:**
- Install NVIDIA drivers
- Install CUDA toolkit

### macOS

**Native Library:** `litertlm_csharp.dylib`

**Deployment:**
```bash
# Option 1: Copy to application directory
cp litertlm_csharp.dylib /path/to/your/app/

# Option 2: Set library path
export DYLD_LIBRARY_PATH=/path/to/lib:$DYLD_LIBRARY_PATH
```

**GPU Support:**
- Use Metal backend for Apple Silicon
- Specify `backend: "gpu"` in Engine constructor

## Troubleshooting

### DllNotFoundException / Library Not Found

**Windows:**
```powershell
# Verify library exists
dir bazel-bin\csharp\litertlm_csharp.dll

# Copy to application directory
copy bazel-bin\csharp\litertlm_csharp.dll .\MyApp\bin\Debug\net8.0\
```

**Linux/macOS:**
```bash
# Verify library exists
ls bazel-bin/csharp/litertlm_csharp.so

# Check dependencies
ldd bazel-bin/csharp/litertlm_csharp.so  # Linux
otool -L bazel-bin/csharp/litertlm_csharp.dylib  # macOS

# Set library path
export LD_LIBRARY_PATH=$(pwd)/bazel-bin/csharp:$LD_LIBRARY_PATH
```

### Build Errors on Windows

**Error: "BAZEL_VC not set"**
```powershell
[System.Environment]::SetEnvironmentVariable("BAZEL_VC", "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC", "User")
```

**Error: "Path too long"**
```bash
# Add to .bazelrc
startup --output_user_root=C:/_bzl
```

**Error: "cl.exe not found"**
```powershell
# Reinstall VC++ tools
choco install visualstudio2022-workload-vctools -y --force
```

### Model Loading Errors

**Error: "Failed to load model"**
- Verify file exists: `ls path/to/model.litertlm`
- Check file permissions
- Ensure sufficient memory available
- Verify model format is compatible

**Error: "Out of memory"**
- Reduce `maxNumTokens` in Engine constructor
- Use quantized models (int4/int8)
- Close other applications

### Memory Access Violations

- Don't use disposed objects
- Don't access sessions from multiple threads
- Keep callback delegates alive with `GC.KeepAlive(callback)`
- Always use `using` statements for proper disposal

### Performance Issues

1. **Slow first run**: Model compilation occurs on first load
   - Solution: Set `cacheDir` parameter
   
2. **Slow generation**: Wrong backend selected
   - Solution: Use GPU backend for large models
   
3. **High memory usage**: Token limit too high
   - Solution: Set appropriate `maxNumTokens`

## Performance Tips

1. **Reuse Engine and Session**: Creating these is expensive, reuse when possible

2. **Enable Benchmarking**: 
   ```csharp
   var engine = new Engine(modelPath, enableBenchmark: true);
   ```

3. **Use Appropriate Backend**:
   - CPU: Small models, mobile devices
   - GPU: Large models, desktop/server
   - NPU: Dedicated AI accelerators

4. **Set Cache Directory**:
   ```csharp
   var engine = new Engine(modelPath, cacheDir: "./cache");
   ```

5. **Optimize Sampler Config**:
   ```csharp
   var config = new SamplerConfig 
   {
       Temperature = 0.7,  // Lower = more focused
       TopK = 20,          // Smaller = faster
       TopP = 0.9          // Higher = more diverse
   };
   ```

6. **Manage Token Limits**:
   ```csharp
   var engine = new Engine(modelPath, maxNumTokens: 1024);
   ```

## Thread Safety

- **Engine**: Thread-safe for creating sessions/conversations
- **Session**: NOT thread-safe - use one per thread
- **Conversation**: NOT thread-safe - use one per thread
- **Callbacks**: Invoked on background threads - marshal to UI thread if needed

**Example: Thread-safe usage**
```csharp
using var engine = new Engine(modelPath);

// Create separate sessions for each thread
var tasks = Enumerable.Range(0, 4).Select(i => Task.Run(() =>
{
    using var session = engine.CreateSession();
    var prompt = new InputData.Text($"Tell me about topic {i}");
    return session.GenerateContent(prompt);
}));

var results = await Task.WhenAll(tasks);
```

## Memory Management

All classes implement `IDisposable`. Always use `using` statements:

```csharp
// Good - automatic disposal
using (var engine = new Engine("model.litertlm"))
using (var session = engine.CreateSession())
{
    // Use session
} // Automatically disposed

// Also good - using declarations (C# 8.0+)
using var engine = new Engine("model.litertlm");
using var session = engine.CreateSession();
// Use session
// Disposed at end of scope
```

## Error Handling

All methods that can fail throw `LiteRtLmException`:

```csharp
try
{
    using var engine = new Engine("model.litertlm");
    using var session = engine.CreateSession();
    var response = session.GenerateContent(new InputData.Text("Hello"));
}
catch (LiteRtLmException ex)
{
    Console.WriteLine($"LiteRT-LM error (code {ex.StatusCode}): {ex.Message}");
    // Handle error appropriately
}
catch (Exception ex)
{
    Console.WriteLine($"Unexpected error: {ex.Message}");
}
```
