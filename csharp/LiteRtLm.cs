// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Google.AI.Edge.LiteRtLm
{
    /// <summary>
    /// Exception thrown by LiteRT-LM operations.
    /// </summary>
    public class LiteRtLmException : Exception
    {
        public int StatusCode { get; }

        public LiteRtLmException(int statusCode, string message) 
            : base(message)
        {
            StatusCode = statusCode;
        }
    }

    /// <summary>
    /// Log severity levels.
    /// </summary>
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

    /// <summary>
    /// Input data types for the model.
    /// </summary>
    public abstract class InputData
    {
        public class Text : InputData
        {
            public string Content { get; }
            public Text(string content) => Content = content;
        }

        public class Audio : InputData
        {
            public byte[] Data { get; }
            public Audio(byte[] data) => Data = data;
        }

        public class Image : InputData
        {
            public byte[] Data { get; }
            public Image(byte[] data) => Data = data;
        }
    }

    /// <summary>
    /// Sampler configuration for content generation.
    /// </summary>
    public class SamplerConfig
    {
        public int TopK { get; set; } = 40;
        public double TopP { get; set; } = 0.95;
        public double Temperature { get; set; } = 0.8;
        public int Seed { get; set; } = 0;
    }

    /// <summary>
    /// Benchmark information.
    /// </summary>
    public class BenchmarkInfo
    {
        public int PrefillTokenCount { get; set; }
        public int DecodeTokenCount { get; set; }
    }

    /// <summary>
    /// Native P/Invoke methods.
    /// </summary>
    internal static class NativeMethods
    {
        private const string DllName = "litertlm_csharp";

        [StructLayout(LayoutKind.Sequential)]
        internal struct InputDataItem
        {
            public int Type;
            public IntPtr TextData;
            public IntPtr BinaryData;
            public int BinaryLength;
        }

        // Delegates for callbacks
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void ResponseCallback([MarshalAs(UnmanagedType.LPStr)] string response);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void CompletionCallback();

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void ErrorCallback(int code, [MarshalAs(UnmanagedType.LPStr)] string message);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void MessageCallback([MarshalAs(UnmanagedType.LPStr)] string message);

        // Error handling
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int LiteRtLm_GetLastErrorCode();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr LiteRtLm_GetLastErrorMessage();

        // Logging
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void LiteRtLm_SetMinLogSeverity(int severity);

        // Engine
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        internal static extern IntPtr LiteRtLm_CreateEngine(
            string modelPath,
            string backend,
            string visionBackend,
            string audioBackend,
            int maxNumTokens,
            string cacheDir,
            bool enableBenchmark);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void LiteRtLm_DeleteEngine(IntPtr engine);

        // Session
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr LiteRtLm_CreateSession(
            IntPtr engine,
            int topK,
            double topP,
            double temperature,
            int seed,
            bool useSampler);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void LiteRtLm_DeleteSession(IntPtr session);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool LiteRtLm_RunPrefill(
            IntPtr session,
            InputDataItem[] inputs,
            int numInputs);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr LiteRtLm_RunDecode(IntPtr session);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr LiteRtLm_GenerateContent(
            IntPtr session,
            InputDataItem[] inputs,
            int numInputs);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool LiteRtLm_GenerateContentStream(
            IntPtr session,
            InputDataItem[] inputs,
            int numInputs,
            ResponseCallback onResponse,
            CompletionCallback onComplete,
            ErrorCallback onError);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void LiteRtLm_CancelProcess(IntPtr session);

        // Conversation
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        internal static extern IntPtr LiteRtLm_CreateConversation(
            IntPtr engine,
            int topK,
            double topP,
            double temperature,
            int seed,
            bool useSampler,
            string systemMessageJson,
            string toolsJson,
            bool enableConstrainedDecoding);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void LiteRtLm_DeleteConversation(IntPtr conversation);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        internal static extern IntPtr LiteRtLm_SendMessage(
            IntPtr conversation,
            string messageJson);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        internal static extern bool LiteRtLm_SendMessageAsync(
            IntPtr conversation,
            string messageJson,
            MessageCallback onMessage,
            CompletionCallback onComplete,
            ErrorCallback onError);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void LiteRtLm_ConversationCancelProcess(IntPtr conversation);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool LiteRtLm_GetBenchmarkInfo(
            IntPtr conversation,
            out int prefillTokenCount,
            out int decodeTokenCount);

        // Memory management
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void LiteRtLm_FreeString(IntPtr str);
    }

    /// <summary>
    /// Utility class for error handling.
    /// </summary>
    internal static class ErrorHandler
    {
        public static void CheckAndThrow()
        {
            int code = NativeMethods.LiteRtLm_GetLastErrorCode();
            if (code != 0)
            {
                IntPtr msgPtr = NativeMethods.LiteRtLm_GetLastErrorMessage();
                string message = Marshal.PtrToStringAnsi(msgPtr) ?? "Unknown error";
                throw new LiteRtLmException(code, message);
            }
        }
    }

    /// <summary>
    /// The LiteRT-LM engine.
    /// </summary>
    public class Engine : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        /// <summary>
        /// Sets the minimum log severity level.
        /// </summary>
        public static void SetMinLogSeverity(LogSeverity severity)
        {
            NativeMethods.LiteRtLm_SetMinLogSeverity((int)severity);
        }

        /// <summary>
        /// Creates a new engine instance.
        /// </summary>
        public Engine(
            string modelPath,
            string backend = "cpu",
            string visionBackend = null,
            string audioBackend = null,
            int maxNumTokens = 0,
            string cacheDir = null,
            bool enableBenchmark = false)
        {
            _handle = NativeMethods.LiteRtLm_CreateEngine(
                modelPath,
                backend,
                visionBackend ?? string.Empty,
                audioBackend ?? string.Empty,
                maxNumTokens,
                cacheDir ?? string.Empty,
                enableBenchmark);

            if (_handle == IntPtr.Zero)
            {
                ErrorHandler.CheckAndThrow();
            }
        }

        /// <summary>
        /// Creates a new session.
        /// </summary>
        public Session CreateSession(SamplerConfig samplerConfig = null)
        {
            ThrowIfDisposed();
            return new Session(this, samplerConfig);
        }

        /// <summary>
        /// Creates a new conversation.
        /// </summary>
        public Conversation CreateConversation(
            SamplerConfig samplerConfig = null,
            string systemMessageJson = null,
            string toolsJson = null,
            bool enableConstrainedDecoding = false)
        {
            ThrowIfDisposed();
            return new Conversation(
                this,
                samplerConfig,
                systemMessageJson,
                toolsJson,
                enableConstrainedDecoding);
        }

        internal IntPtr Handle
        {
            get
            {
                ThrowIfDisposed();
                return _handle;
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(Engine));
            }
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (_handle != IntPtr.Zero)
                {
                    NativeMethods.LiteRtLm_DeleteEngine(_handle);
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
            }
        }

        ~Engine()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>
    /// A session for interacting with the model.
    /// </summary>
    public class Session : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        internal Session(Engine engine, SamplerConfig samplerConfig)
        {
            bool useSampler = samplerConfig != null;
            int topK = samplerConfig?.TopK ?? 40;
            double topP = samplerConfig?.TopP ?? 0.95;
            double temperature = samplerConfig?.Temperature ?? 0.8;
            int seed = samplerConfig?.Seed ?? 0;

            _handle = NativeMethods.LiteRtLm_CreateSession(
                engine.Handle,
                topK,
                topP,
                temperature,
                seed,
                useSampler);

            if (_handle == IntPtr.Zero)
            {
                ErrorHandler.CheckAndThrow();
            }
        }

        /// <summary>
        /// Runs the prefill phase with the given input data.
        /// </summary>
        public void RunPrefill(params InputData[] inputs)
        {
            ThrowIfDisposed();
            var nativeInputs = ConvertInputs(inputs);
            
            if (!NativeMethods.LiteRtLm_RunPrefill(_handle, nativeInputs, nativeInputs.Length))
            {
                ErrorHandler.CheckAndThrow();
            }

            FreeInputs(nativeInputs);
        }

        /// <summary>
        /// Runs a single decode step.
        /// </summary>
        public string RunDecode()
        {
            ThrowIfDisposed();
            IntPtr resultPtr = NativeMethods.LiteRtLm_RunDecode(_handle);
            
            if (resultPtr == IntPtr.Zero)
            {
                ErrorHandler.CheckAndThrow();
            }

            string result = Marshal.PtrToStringAnsi(resultPtr);
            NativeMethods.LiteRtLm_FreeString(resultPtr);
            return result;
        }

        /// <summary>
        /// Generates content synchronously.
        /// </summary>
        public string GenerateContent(params InputData[] inputs)
        {
            ThrowIfDisposed();
            var nativeInputs = ConvertInputs(inputs);
            
            IntPtr resultPtr = NativeMethods.LiteRtLm_GenerateContent(
                _handle,
                nativeInputs,
                nativeInputs.Length);

            FreeInputs(nativeInputs);

            if (resultPtr == IntPtr.Zero)
            {
                ErrorHandler.CheckAndThrow();
            }

            string result = Marshal.PtrToStringAnsi(resultPtr);
            NativeMethods.LiteRtLm_FreeString(resultPtr);
            return result;
        }

        /// <summary>
        /// Generates content with streaming responses.
        /// </summary>
        public void GenerateContentStream(
            InputData[] inputs,
            Action<string> onResponse,
            Action onComplete = null,
            Action<int, string> onError = null)
        {
            ThrowIfDisposed();
            var nativeInputs = ConvertInputs(inputs);

            // Create delegates that won't be garbage collected
            var responseCallback = new NativeMethods.ResponseCallback(onResponse);
            var completionCallback = onComplete != null 
                ? new NativeMethods.CompletionCallback(onComplete) 
                : null;
            var errorCallback = onError != null 
                ? new NativeMethods.ErrorCallback(onError) 
                : null;

            bool success = NativeMethods.LiteRtLm_GenerateContentStream(
                _handle,
                nativeInputs,
                nativeInputs.Length,
                responseCallback,
                completionCallback,
                errorCallback);

            FreeInputs(nativeInputs);

            if (!success)
            {
                ErrorHandler.CheckAndThrow();
            }

            // Keep delegates alive during async operation
            GC.KeepAlive(responseCallback);
            GC.KeepAlive(completionCallback);
            GC.KeepAlive(errorCallback);
        }

        /// <summary>
        /// Cancels any ongoing processing.
        /// </summary>
        public void CancelProcess()
        {
            ThrowIfDisposed();
            NativeMethods.LiteRtLm_CancelProcess(_handle);
        }

        private NativeMethods.InputDataItem[] ConvertInputs(InputData[] inputs)
        {
            var nativeInputs = new NativeMethods.InputDataItem[inputs.Length];

            for (int i = 0; i < inputs.Length; i++)
            {
                if (inputs[i] is InputData.Text text)
                {
                    nativeInputs[i].Type = 0;
                    nativeInputs[i].TextData = Marshal.StringToHGlobalAnsi(text.Content);
                    nativeInputs[i].BinaryData = IntPtr.Zero;
                    nativeInputs[i].BinaryLength = 0;
                }
                else if (inputs[i] is InputData.Audio audio)
                {
                    nativeInputs[i].Type = 1;
                    nativeInputs[i].TextData = IntPtr.Zero;
                    nativeInputs[i].BinaryData = Marshal.AllocHGlobal(audio.Data.Length);
                    Marshal.Copy(audio.Data, 0, nativeInputs[i].BinaryData, audio.Data.Length);
                    nativeInputs[i].BinaryLength = audio.Data.Length;
                }
                else if (inputs[i] is InputData.Image image)
                {
                    nativeInputs[i].Type = 2;
                    nativeInputs[i].TextData = IntPtr.Zero;
                    nativeInputs[i].BinaryData = Marshal.AllocHGlobal(image.Data.Length);
                    Marshal.Copy(image.Data, 0, nativeInputs[i].BinaryData, image.Data.Length);
                    nativeInputs[i].BinaryLength = image.Data.Length;
                }
            }

            return nativeInputs;
        }

        private void FreeInputs(NativeMethods.InputDataItem[] nativeInputs)
        {
            foreach (var input in nativeInputs)
            {
                if (input.TextData != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(input.TextData);
                }
                if (input.BinaryData != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(input.BinaryData);
                }
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(Session));
            }
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (_handle != IntPtr.Zero)
                {
                    NativeMethods.LiteRtLm_DeleteSession(_handle);
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
            }
        }

        ~Session()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>
    /// A conversation for multi-turn interactions with the model.
    /// </summary>
    public class Conversation : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        internal Conversation(
            Engine engine,
            SamplerConfig samplerConfig,
            string systemMessageJson,
            string toolsJson,
            bool enableConstrainedDecoding)
        {
            bool useSampler = samplerConfig != null;
            int topK = samplerConfig?.TopK ?? 40;
            double topP = samplerConfig?.TopP ?? 0.95;
            double temperature = samplerConfig?.Temperature ?? 0.8;
            int seed = samplerConfig?.Seed ?? 0;

            _handle = NativeMethods.LiteRtLm_CreateConversation(
                engine.Handle,
                topK,
                topP,
                temperature,
                seed,
                useSampler,
                systemMessageJson ?? string.Empty,
                toolsJson ?? "[]",
                enableConstrainedDecoding);

            if (_handle == IntPtr.Zero)
            {
                ErrorHandler.CheckAndThrow();
            }
        }

        /// <summary>
        /// Sends a message synchronously and returns the response.
        /// </summary>
        public string SendMessage(string messageJson)
        {
            ThrowIfDisposed();
            IntPtr resultPtr = NativeMethods.LiteRtLm_SendMessage(_handle, messageJson);

            if (resultPtr == IntPtr.Zero)
            {
                ErrorHandler.CheckAndThrow();
            }

            string result = Marshal.PtrToStringAnsi(resultPtr);
            NativeMethods.LiteRtLm_FreeString(resultPtr);
            return result;
        }

        /// <summary>
        /// Sends a message asynchronously with streaming responses.
        /// </summary>
        public void SendMessageAsync(
            string messageJson,
            Action<string> onMessage,
            Action onComplete = null,
            Action<int, string> onError = null)
        {
            ThrowIfDisposed();

            // Create delegates that won't be garbage collected
            var messageCallback = new NativeMethods.MessageCallback(onMessage);
            var completionCallback = onComplete != null 
                ? new NativeMethods.CompletionCallback(onComplete) 
                : null;
            var errorCallback = onError != null 
                ? new NativeMethods.ErrorCallback(onError) 
                : null;

            bool success = NativeMethods.LiteRtLm_SendMessageAsync(
                _handle,
                messageJson,
                messageCallback,
                completionCallback,
                errorCallback);

            if (!success)
            {
                ErrorHandler.CheckAndThrow();
            }

            // Keep delegates alive during async operation
            GC.KeepAlive(messageCallback);
            GC.KeepAlive(completionCallback);
            GC.KeepAlive(errorCallback);
        }

        /// <summary>
        /// Cancels any ongoing processing.
        /// </summary>
        public void CancelProcess()
        {
            ThrowIfDisposed();
            NativeMethods.LiteRtLm_ConversationCancelProcess(_handle);
        }

        /// <summary>
        /// Gets benchmark information for the conversation.
        /// </summary>
        public BenchmarkInfo GetBenchmarkInfo()
        {
            ThrowIfDisposed();
            
            if (!NativeMethods.LiteRtLm_GetBenchmarkInfo(
                _handle,
                out int prefillTokenCount,
                out int decodeTokenCount))
            {
                ErrorHandler.CheckAndThrow();
            }

            return new BenchmarkInfo
            {
                PrefillTokenCount = prefillTokenCount,
                DecodeTokenCount = decodeTokenCount
            };
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(Conversation));
            }
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (_handle != IntPtr.Zero)
                {
                    NativeMethods.LiteRtLm_DeleteConversation(_handle);
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
            }
        }

        ~Conversation()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
    }
}
