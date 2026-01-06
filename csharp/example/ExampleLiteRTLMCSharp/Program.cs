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
using System.Threading;
using Google.AI.Edge.LiteRtLm;

namespace Google.AI.Edge.LiteRtLm.Examples
{
    /// <summary>
    /// Example usage of the LiteRT-LM C# bindings.
    /// </summary>
    public class Example
    {
        private const string modelPath = "D:/litertlm_model/gemma-3n-E4B-it-int4.litertlm";
        public static void Main(string[] args)
        {
            // Set logging level
            Engine.SetMinLogSeverity(LogSeverity.Verbose);

            // Example 1: Simple text generation with Session
            SimpleTextGeneration();

            // Example 2: Streaming text generation
            StreamingTextGeneration(); // Commented out - GPU backend has issues

            // Example 3: Multi-turn conversation
            MultiTurnConversation();

            // Example 4: Conversation with function calling
            ConversationWithTools();
        }

        /// <summary>
        /// Example 1: Simple text generation using a Session.
        /// </summary>
        static void SimpleTextGeneration()
        {
            Console.WriteLine("=== Example 1: Simple Text Generation ===\n");

            try
            {
                using var engine = new Engine(
                    modelPath: modelPath,
                    backend: "gpu",
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

        /// <summary>
        /// Example 2: Streaming text generation.
        /// </summary>
        static void StreamingTextGeneration()
        {
            Console.WriteLine("=== Example 2: Streaming Text Generation ===\n");

            try
            {
                // Use gpu backend instead of GPU to avoid compilation errors
                using var engine = new Engine(
                    modelPath: modelPath,
                    backend: "gpu");

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

        /// <summary>
        /// Example 3: Multi-turn conversation.
        /// </summary>
        static void MultiTurnConversation()
        {
            Console.WriteLine("=== Example 3: Multi-turn Conversation ===\n");

            try
            {
                using var engine = new Engine(
                    modelPath: modelPath,
                    backend: "gpu",
                    enableBenchmark: true);

                // Fixed: Remove system instruction for now, or format it correctly
                // The Conversation API may not handle system messages the same way
                using var conversation = engine.CreateConversation(
                    samplerConfig: new SamplerConfig { Temperature = 0.7 });

                // Turn 1 - Fixed JSON format: content must be a plain string
                string userMessage1 = @"{""role"":""user"",""content"":""What is machine learning?""}";

                Console.WriteLine("User: What is machine learning?");
                string response1 = conversation.SendMessage(userMessage1);
                Console.WriteLine($"Assistant: {response1}\n");

                // Turn 2 - Fixed JSON format
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

        /// <summary>
        /// Example 4: Conversation with function calling/tools.
        /// </summary>
        static void ConversationWithTools()
        {
            Console.WriteLine("=== Example 4: Conversation with Tools ===\n");

            try
            {
                using var engine = new Engine(
                    modelPath: modelPath,
                    backend: "gpu");

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

                // Fixed JSON format: single line, no formatting
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

        /// <summary>
        /// Example 5: Using multimodal input (image + text).
        /// </summary>
        static void MultimodalGeneration()
        {
            Console.WriteLine("=== Example 5: Multimodal Generation ===\n");

            try
            {
                using var engine = new Engine(
                    modelPath: "path/to/multimodal_model.tflite",
                    backend: "gpu",
                    visionBackend: "gpu");

                using var session = engine.CreateSession();

                // Load image data
                byte[] imageData = System.IO.File.ReadAllBytes("path/to/image.jpg");

                var inputs = new InputData[]
                {
                    new InputData.Image(imageData),
                    new InputData.Text("What do you see in this image?")
                };

                string response = session.GenerateContent(inputs);

                Console.WriteLine("Question: What do you see in this image?");
                Console.WriteLine($"Response: {response}\n");
            }
            catch (LiteRtLmException ex)
            {
                Console.WriteLine($"Error (code {ex.StatusCode}): {ex.Message}");
            }
        }

        /// <summary>
        /// Example 6: Manual prefill and decode steps.
        /// </summary>
        static void ManualPrefillDecode()
        {
            Console.WriteLine("=== Example 6: Manual Prefill and Decode ===\n");

            try
            {
                using var engine = new Engine(
                    modelPath: "/home/nguyenan/Downloads/gemma-3n-E4B-it-int4.litertlm",
                    backend: "gpu");

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
                    
                    // You could add stopping conditions here
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

        /// <summary>
        /// Example 7: Cancelling generation.
        /// </summary>
        static void CancelGeneration()
        {
            Console.WriteLine("=== Example 7: Cancel Generation ===\n");

            try
            {
                using var engine = new Engine(
                    modelPath: "/home/nguyenan/Downloads/gemma-3n-E4B-it-int4.litertlm",
                    backend: "gpu");

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
    }
}