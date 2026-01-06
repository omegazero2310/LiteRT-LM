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

#include <sys/stat.h>

#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/absl_log.h"
#include "absl/log/globals.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "nlohmann/json.hpp"
#include "litert/c/internal/litert_logging.h"
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/sampler_params.pb.h"
#include "tflite/logger.h"
#include "tflite/minimal_logging.h"

// Windows-specific includes
#if defined(_WIN32)
#include <io.h>
#define ACCESS _access
#define F_OK 0
#else
#include <unistd.h>
#define ACCESS access
#endif

#if defined(_WIN32)
#define LITERTLM_EXPORT __declspec(dllexport)
#else
#define LITERTLM_EXPORT __attribute__((visibility("default")))
#endif

namespace {
using litert::lm::Backend;
using litert::lm::Conversation;
using litert::lm::ConversationConfig;
using litert::lm::Engine;
using litert::lm::EngineSettings;
using litert::lm::InputAudio;
using litert::lm::InputData;
using litert::lm::InputImage;
using litert::lm::InputText;
using litert::lm::JsonMessage;
using litert::lm::JsonPreface;
using litert::lm::Message;
using litert::lm::ModelAssets;
using litert::lm::Preface;
using litert::lm::Responses;
using litert::lm::SessionConfig;
using litert::lm::proto::SamplerParameters;

// Error handling structure
struct ErrorInfo {
  int code;
  char message[1024];
};

thread_local ErrorInfo g_last_error = {0, ""};

void SetLastError(int code, const std::string& message) {
  g_last_error.code = code;
  strncpy(g_last_error.message, message.c_str(), sizeof(g_last_error.message) - 1);
  g_last_error.message[sizeof(g_last_error.message) - 1] = '\0';
}

void ClearLastError() {
  g_last_error.code = 0;
  g_last_error.message[0] = '\0';
}

// Better cross-platform file existence check
bool FileExists(const char* path) {
#if defined(_WIN32)
  // On Windows, use _access_s for better compatibility
  #include <io.h>
  return _access_s(path, 0) == 0;
#else
  struct stat buffer;
  return (stat(path, &buffer) == 0);
#endif
}

// Callback function pointer types for C#
typedef void (*ResponseCallback)(const char* response);
typedef void (*CompletionCallback)();
typedef void (*ErrorCallback)(int code, const char* message);
typedef void (*MessageCallback)(const char* message);

}  // namespace

extern "C" {

// Error handling
LITERTLM_EXPORT int LiteRtLm_GetLastErrorCode() {
  return g_last_error.code;
}

LITERTLM_EXPORT const char* LiteRtLm_GetLastErrorMessage() {
  return g_last_error.message;
}

// Logging
LITERTLM_EXPORT void LiteRtLm_SetMinLogSeverity(int log_severity) {
  absl::LogSeverityAtLeast absl_log_severity;
  LiteRtLogSeverity litert_log_severity;
  tflite::LogSeverity tflite_log_severity;

  switch (log_severity) {
    case 0:  // verbose
      absl_log_severity = absl::LogSeverityAtLeast::kInfo;
      litert_log_severity = kLiteRtLogSeverityVerbose;
      tflite_log_severity = tflite::TFLITE_LOG_VERBOSE;
      break;
    case 1:  // debug
      absl_log_severity = absl::LogSeverityAtLeast::kInfo;
      litert_log_severity = kLiteRtLogSeverityDebug;
      tflite_log_severity = tflite::TFLITE_LOG_VERBOSE;
      break;
    case 2:  // info
      absl_log_severity = absl::LogSeverityAtLeast::kInfo;
      litert_log_severity = kLiteRtLogSeverityInfo;
      tflite_log_severity = tflite::TFLITE_LOG_INFO;
      break;
    case 3:  // warning
      absl_log_severity = absl::LogSeverityAtLeast::kWarning;
      litert_log_severity = kLiteRtLogSeverityWarning;
      tflite_log_severity = tflite::TFLITE_LOG_WARNING;
      break;
    case 4:  // error
      absl_log_severity = absl::LogSeverityAtLeast::kError;
      litert_log_severity = kLiteRtLogSeverityError;
      tflite_log_severity = tflite::TFLITE_LOG_ERROR;
      break;
    case 5:  // fatal
      absl_log_severity = absl::LogSeverityAtLeast::kFatal;
      litert_log_severity = kLiteRtLogSeverityError;
      tflite_log_severity = tflite::TFLITE_LOG_ERROR;
      break;
    default:  // infinity
      absl_log_severity = absl::LogSeverityAtLeast::kInfinity;
      litert_log_severity = kLiteRtLogSeveritySilent;
      tflite_log_severity = tflite::TFLITE_LOG_SILENT;
      break;
  }

  absl::SetMinLogLevel(absl_log_severity);
  LiteRtSetMinLoggerSeverity(LiteRtGetDefaultLogger(), litert_log_severity);
  tflite::logging_internal::MinimalLogger::SetMinimumLogSeverity(tflite_log_severity);
}

// Optional: Add a debug function to test path reception
LITERTLM_EXPORT const char* LiteRtLm_TestPathEcho(const char* path) {
  if (!path) return nullptr;
  
  // Log what we received
  ABSL_LOG(INFO) << "Path echo test: [" << path << "]";
  ABSL_LOG(INFO) << "Path length: " << strlen(path);
  
  // Check if file exists
  bool exists = FileExists(path);
  ABSL_LOG(INFO) << "File exists: " << (exists ? "YES" : "NO");
  
  // Return a status message
  std::string result = std::string("Received: ") + path + 
                      " | Exists: " + (exists ? "YES" : "NO");
  char* output = new char[result.length() + 1];
  strcpy(output, result.c_str());
  return output;
}

// Engine management
LITERTLM_EXPORT void* LiteRtLm_CreateEngine(
    const char* model_path,
    const char* backend,
    const char* vision_backend,
    const char* audio_backend,
    int max_num_tokens,
    const char* cache_dir,
    bool enable_benchmark) {
  
  ClearLastError();

  // Enhanced logging for debugging
  ABSL_LOG(INFO) << "LiteRtLm_CreateEngine called";
  ABSL_LOG(INFO) << "Model path received: [" << (model_path ? model_path : "NULL") << "]";
  ABSL_LOG(INFO) << "Backend: [" << (backend ? backend : "NULL") << "]";

  if (!model_path || strlen(model_path) == 0) {
    SetLastError(static_cast<int>(absl::StatusCode::kInvalidArgument),
                 "Model path is null or empty");
    return nullptr;
  }

  // Check file existence with improved method
  if (!FileExists(model_path)) {
    std::string error_msg = "Model file not found: ";
    error_msg += model_path;
    
    // Add more debugging info
    ABSL_LOG(ERROR) << error_msg;
    ABSL_LOG(ERROR) << "Path length: " << strlen(model_path);
    ABSL_LOG(ERROR) << "First char: " << (int)model_path[0];
    
    SetLastError(static_cast<int>(absl::StatusCode::kNotFound), error_msg);
    return nullptr;
  }

  ABSL_LOG(INFO) << "File exists, proceeding with model loading...";

  auto model_assets = ModelAssets::Create(model_path);
  if (!model_assets.ok()) {
    SetLastError(static_cast<int>(model_assets.status().code()),
                 "Failed to create model assets: " + model_assets.status().ToString());
    return nullptr;
  }

  auto backend_enum = litert::lm::GetBackendFromString(backend);
  if (!backend_enum.ok()) {
    SetLastError(static_cast<int>(backend_enum.status().code()),
                 backend_enum.status().ToString());
    return nullptr;
  }

  std::optional<Backend> vision_backend_optional = std::nullopt;
  if (vision_backend && strlen(vision_backend) > 0) {
    auto vision_backend_enum = litert::lm::GetBackendFromString(vision_backend);
    if (!vision_backend_enum.ok()) {
      SetLastError(static_cast<int>(vision_backend_enum.status().code()),
                   vision_backend_enum.status().ToString());
      return nullptr;
    }
    vision_backend_optional = vision_backend_enum.value();
  }

  std::optional<Backend> audio_backend_optional = std::nullopt;
  if (audio_backend && strlen(audio_backend) > 0) {
    auto audio_backend_enum = litert::lm::GetBackendFromString(audio_backend);
    if (!audio_backend_enum.ok()) {
      SetLastError(static_cast<int>(audio_backend_enum.status().code()),
                   audio_backend_enum.status().ToString());
      return nullptr;
    }
    audio_backend_optional = audio_backend_enum.value();
  }

  auto settings = EngineSettings::CreateDefault(
      *model_assets, *backend_enum, vision_backend_optional, audio_backend_optional);
  if (!settings.ok()) {
    SetLastError(static_cast<int>(settings.status().code()),
                 "Failed to create engine settings: " + settings.status().ToString());
    return nullptr;
  }

  if (cache_dir && strlen(cache_dir) > 0) {
    settings->GetMutableMainExecutorSettings().SetCacheDir(cache_dir);
    if (vision_backend_optional.has_value()) {
      settings->GetMutableVisionExecutorSettings()->SetCacheDir(cache_dir);
    }
    if (audio_backend_optional.has_value()) {
      settings->GetMutableAudioExecutorSettings()->SetCacheDir(cache_dir);
    }
  }

  if (max_num_tokens > 0) {
    settings->GetMutableMainExecutorSettings().SetMaxNumTokens(max_num_tokens);
  }

  if (enable_benchmark) {
    settings->GetMutableBenchmarkParams();
  }

  auto engine = Engine::CreateEngine(*settings);
  if (!engine.ok()) {
    SetLastError(static_cast<int>(engine.status().code()),
                 "Failed to create engine: " + engine.status().ToString());
    return nullptr;
  }

  return engine->release();
}

LITERTLM_EXPORT void LiteRtLm_DeleteEngine(void* engine) {
  if (engine) {
    delete reinterpret_cast<Engine*>(engine);
  }
}

// Session management
LITERTLM_EXPORT void* LiteRtLm_CreateSession(
    void* engine,
    int top_k,
    double top_p,
    double temperature,
    int seed,
    bool use_sampler) {
  
  ClearLastError();

  if (!engine) {
    SetLastError(static_cast<int>(absl::StatusCode::kInvalidArgument), "Engine is null");
    return nullptr;
  }

  auto session_config = SessionConfig::CreateDefault();

  if (use_sampler) {
    SamplerParameters sampler_params;
    sampler_params.set_type(SamplerParameters::TOP_P);
    sampler_params.set_k(top_k);
    sampler_params.set_p(top_p);
    sampler_params.set_temperature(temperature);
    sampler_params.set_seed(seed);
    session_config.GetMutableSamplerParams() = sampler_params;
  }

  Engine* eng = reinterpret_cast<Engine*>(engine);
  auto session = eng->CreateSession(session_config);
  if (!session.ok()) {
    SetLastError(static_cast<int>(session.status().code()),
                 "Failed to create session: " + session.status().ToString());
    return nullptr;
  }

  return session->release();
}

LITERTLM_EXPORT void LiteRtLm_DeleteSession(void* session) {
  if (session) {
    delete reinterpret_cast<Engine::Session*>(session);
  }
}

// Input data structures
struct InputDataItem {
  int type;  // 0=text, 1=audio, 2=image
  const char* text_data;
  const unsigned char* binary_data;
  int binary_length;
};

LITERTLM_EXPORT bool LiteRtLm_RunPrefill(void* session, InputDataItem* inputs, int num_inputs) {
  ClearLastError();

  if (!session) {
    SetLastError(static_cast<int>(absl::StatusCode::kInvalidArgument), "Session is null");
    return false;
  }

  Engine::Session* sess = reinterpret_cast<Engine::Session*>(session);
  std::vector<InputData> contents;
  contents.reserve(num_inputs);

  for (int i = 0; i < num_inputs; ++i) {
    switch (inputs[i].type) {
      case 0:  // text
        contents.emplace_back(InputText(inputs[i].text_data));
        break;
      case 1:  // audio
        contents.emplace_back(InputAudio(std::string(
            reinterpret_cast<const char*>(inputs[i].binary_data),
            inputs[i].binary_length)));
        break;
      case 2:  // image
        contents.emplace_back(InputImage(std::string(
            reinterpret_cast<const char*>(inputs[i].binary_data),
            inputs[i].binary_length)));
        break;
      default:
        SetLastError(static_cast<int>(absl::StatusCode::kInvalidArgument),
                     "Unsupported input data type");
        return false;
    }
  }

  auto status = sess->RunPrefill(contents);
  if (!status.ok()) {
    SetLastError(static_cast<int>(status.code()),
                 "Failed to run prefill: " + status.ToString());
    return false;
  }

  return true;
}

LITERTLM_EXPORT char* LiteRtLm_RunDecode(void* session) {
  ClearLastError();

  if (!session) {
    SetLastError(static_cast<int>(absl::StatusCode::kInvalidArgument), "Session is null");
    return nullptr;
  }

  Engine::Session* sess = reinterpret_cast<Engine::Session*>(session);
  auto responses = sess->RunDecode();

  if (!responses.ok()) {
    SetLastError(static_cast<int>(responses.status().code()),
                 "Failed to run decode: " + responses.status().ToString());
    return nullptr;
  }

  if (responses->GetTexts().size() != 1) {
    SetLastError(static_cast<int>(absl::StatusCode::kInternal),
                 "Number of output candidates should be 1");
    return nullptr;
  }

  const std::string& text = responses->GetTexts()[0];
  char* result = new char[text.length() + 1];
  strcpy(result, text.c_str());
  return result;
}

LITERTLM_EXPORT char* LiteRtLm_GenerateContent(void* session, InputDataItem* inputs, int num_inputs) {
  ClearLastError();

  if (!session) {
    SetLastError(static_cast<int>(absl::StatusCode::kInvalidArgument), "Session is null");
    return nullptr;
  }

  Engine::Session* sess = reinterpret_cast<Engine::Session*>(session);
  std::vector<InputData> contents;
  contents.reserve(num_inputs);

  for (int i = 0; i < num_inputs; ++i) {
    switch (inputs[i].type) {
      case 0:  // text
        contents.emplace_back(InputText(inputs[i].text_data));
        break;
      case 1:  // audio
        contents.emplace_back(InputAudio(std::string(
            reinterpret_cast<const char*>(inputs[i].binary_data),
            inputs[i].binary_length)));
        break;
      case 2:  // image
        contents.emplace_back(InputImage(std::string(
            reinterpret_cast<const char*>(inputs[i].binary_data),
            inputs[i].binary_length)));
        break;
      default:
        SetLastError(static_cast<int>(absl::StatusCode::kInvalidArgument),
                     "Unsupported input data type");
        return nullptr;
    }
  }

  auto responses = sess->GenerateContent(contents);
  if (!responses.ok()) {
    SetLastError(static_cast<int>(responses.status().code()),
                 "Failed to generate content: " + responses.status().ToString());
    return nullptr;
  }

  if (responses->GetTexts().empty()) {
    char* result = new char[1];
    result[0] = '\0';
    return result;
  }

  const std::string& text = responses->GetTexts()[0];
  char* result = new char[text.length() + 1];
  strcpy(result, text.c_str());
  return result;
}

LITERTLM_EXPORT bool LiteRtLm_GenerateContentStream(
    void* session,
    InputDataItem* inputs,
    int num_inputs,
    ResponseCallback on_response,
    CompletionCallback on_complete,
    ErrorCallback on_error) {
  
  ClearLastError();

  if (!session) {
    SetLastError(static_cast<int>(absl::StatusCode::kInvalidArgument), "Session is null");
    return false;
  }

  Engine::Session* sess = reinterpret_cast<Engine::Session*>(session);
  std::vector<InputData> contents;
  contents.reserve(num_inputs);

  for (int i = 0; i < num_inputs; ++i) {
    switch (inputs[i].type) {
      case 0:  // text
        contents.emplace_back(InputText(inputs[i].text_data));
        break;
      case 1:  // audio
        contents.emplace_back(InputAudio(std::string(
            reinterpret_cast<const char*>(inputs[i].binary_data),
            inputs[i].binary_length)));
        break;
      case 2:  // image
        contents.emplace_back(InputImage(std::string(
            reinterpret_cast<const char*>(inputs[i].binary_data),
            inputs[i].binary_length)));
        break;
      default:
        SetLastError(static_cast<int>(absl::StatusCode::kInvalidArgument),
                     "Unsupported input data type");
        return false;
    }
  }

  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback_fn =
      [on_response, on_complete, on_error](absl::StatusOr<Responses> responses) {
        if (responses.ok()) {
          if (responses->GetTaskState() == litert::lm::TaskState::kDone) {
            if (on_complete) on_complete();
          } else if (responses->GetTaskState() == litert::lm::TaskState::kMaxNumTokensReached) {
            if (on_error) {
              on_error(static_cast<int>(absl::StatusCode::kInternal),
                      "Maximum kv-cache size reached.");
            }
          } else {
            if (on_response && !responses->GetTexts().empty()) {
              on_response(responses->GetTexts()[0].c_str());
            }
          }
        } else {
          if (on_error) {
            on_error(static_cast<int>(responses.status().code()),
                    responses.status().message().data());
          }
        }
      };

  auto status = sess->GenerateContentStream(contents, std::move(callback_fn));
  if (!status.ok()) {
    SetLastError(static_cast<int>(status.code()),
                 "Failed to start GenerateContentStream: " + status.ToString());
    return false;
  }

  return true;
}

LITERTLM_EXPORT void LiteRtLm_CancelProcess(void* session) {
  if (session) {
    Engine::Session* sess = reinterpret_cast<Engine::Session*>(session);
    sess->CancelProcess();
  }
}

// Conversation management
LITERTLM_EXPORT void* LiteRtLm_CreateConversation(
    void* engine,
    int top_k,
    double top_p,
    double temperature,
    int seed,
    bool use_sampler,
    const char* system_message_json,
    const char* tools_json,
    bool enable_constrained_decoding) {
  
  ClearLastError();

  if (!engine) {
    SetLastError(static_cast<int>(absl::StatusCode::kInvalidArgument), "Engine is null");
    return nullptr;
  }

  Engine* eng = reinterpret_cast<Engine*>(engine);

  auto session_config = SessionConfig::CreateDefault();
  if (use_sampler) {
    SamplerParameters sampler_params;
    sampler_params.set_type(SamplerParameters::TOP_P);
    sampler_params.set_k(top_k);
    sampler_params.set_p(top_p);
    sampler_params.set_temperature(temperature);
    sampler_params.set_seed(seed);
    session_config.GetMutableSamplerParams() = sampler_params;
  }

  JsonPreface json_preface;

  if (system_message_json && strlen(system_message_json) > 0) {
    nlohmann::ordered_json system_message;
    system_message["role"] = "system";
    system_message["content"] = nlohmann::ordered_json::parse(system_message_json);

    nlohmann::ordered_json::array_t messages;
    messages.push_back(system_message);
    json_preface.messages = messages;
  }

  if (tools_json && strlen(tools_json) > 0) {
    auto tool_json = nlohmann::ordered_json::parse(tools_json);
    if (tool_json.is_array()) {
      json_preface.tools = tool_json.get<nlohmann::ordered_json::array_t>();
    } else {
      SetLastError(static_cast<int>(absl::StatusCode::kInvalidArgument),
                   "tools_json should be a JSON array");
      return nullptr;
    }
  }

  std::optional<Preface> preface = json_preface;

  auto conversation_config = ConversationConfig::CreateFromSessionConfig(
      *eng, session_config, preface, std::nullopt, enable_constrained_decoding);

  if (!conversation_config.ok()) {
    SetLastError(static_cast<int>(conversation_config.status().code()),
                 "Failed to create conversation config: " + conversation_config.status().ToString());
    return nullptr;
  }

  auto conversation = Conversation::Create(*eng, *conversation_config);
  if (!conversation.ok()) {
    SetLastError(static_cast<int>(conversation.status().code()),
                 "Failed to create conversation: " + conversation.status().ToString());
    return nullptr;
  }

  return conversation->release();
}

LITERTLM_EXPORT void LiteRtLm_DeleteConversation(void* conversation) {
  if (conversation) {
    delete reinterpret_cast<Conversation*>(conversation);
  }
}

LITERTLM_EXPORT char* LiteRtLm_SendMessage(void* conversation, const char* message_json) {
  ClearLastError();

  if (!conversation) {
    SetLastError(static_cast<int>(absl::StatusCode::kInvalidArgument), "Conversation is null");
    return nullptr;
  }

  Conversation* conv = reinterpret_cast<Conversation*>(conversation);
  litert::lm::JsonMessage json_message = nlohmann::ordered_json::parse(message_json);

  auto response = conv->SendMessage(json_message);
  if (!response.ok()) {
    SetLastError(static_cast<int>(response.status().code()),
                 "Failed to send message: " + response.status().ToString());
    return nullptr;
  }

  if (!std::holds_alternative<litert::lm::JsonMessage>(*response)) {
    SetLastError(static_cast<int>(absl::StatusCode::kInternal),
                 "Response is not a JsonMessage");
    return nullptr;
  }

  auto json_response = std::get<litert::lm::JsonMessage>(*response);
  std::string response_str = json_response.dump();
  char* result = new char[response_str.length() + 1];
  strcpy(result, response_str.c_str());
  return result;
}

LITERTLM_EXPORT bool LiteRtLm_SendMessageAsync(
    void* conversation,
    const char* message_json,
    MessageCallback on_message,
    CompletionCallback on_complete,
    ErrorCallback on_error) {
  
  ClearLastError();

  if (!conversation) {
    SetLastError(static_cast<int>(absl::StatusCode::kInvalidArgument), "Conversation is null");
    return false;
  }

  Conversation* conv = reinterpret_cast<Conversation*>(conversation);
  litert::lm::JsonMessage json_message = nlohmann::ordered_json::parse(message_json);

  absl::AnyInvocable<void(absl::StatusOr<Message>)> callback_fn =
      [on_message, on_complete, on_error](absl::StatusOr<Message> message) {
        if (message.ok()) {
          if (!std::holds_alternative<litert::lm::JsonMessage>(*message)) {
            if (on_error) {
              on_error(static_cast<int>(absl::StatusCode::kInternal),
                      "Response is not a JsonMessage");
            }
          } else {
            auto json_message = std::get<litert::lm::JsonMessage>(*message);
            if (json_message.is_null()) {
              if (on_complete) on_complete();
            } else {
              if (on_message) {
                std::string message_str = json_message.dump();
                on_message(message_str.c_str());
              }
            }
          }
        } else {
          if (on_error) {
            on_error(static_cast<int>(message.status().code()),
                    message.status().message().data());
          }
        }
      };

  auto status = conv->SendMessageAsync(json_message, std::move(callback_fn));
  if (!status.ok()) {
    SetLastError(static_cast<int>(status.code()),
                 "Failed to start SendMessageAsync: " + status.ToString());
    return false;
  }

  return true;
}

LITERTLM_EXPORT void LiteRtLm_ConversationCancelProcess(void* conversation) {
  if (conversation) {
    Conversation* conv = reinterpret_cast<Conversation*>(conversation);
    conv->CancelProcess();
  }
}

LITERTLM_EXPORT bool LiteRtLm_GetBenchmarkInfo(
    void* conversation,
    int* out_prefill_token_count,
    int* out_decode_token_count) {
  
  ClearLastError();

  if (!conversation) {
    SetLastError(static_cast<int>(absl::StatusCode::kInvalidArgument), "Conversation is null");
    return false;
  }

  Conversation* conv = reinterpret_cast<Conversation*>(conversation);
  auto benchmark_info = conv->GetBenchmarkInfo();
  
  if (!benchmark_info.ok()) {
    SetLastError(static_cast<int>(benchmark_info.status().code()),
                 "Failed to get benchmark info: " + benchmark_info.status().ToString());
    return false;
  }

  int last_prefill_token_count = 0;
  if (benchmark_info->GetTotalPrefillTurns() > 0) {
    last_prefill_token_count =
        benchmark_info->GetPrefillTurn(benchmark_info->GetTotalPrefillTurns() - 1).num_tokens;
  }

  int last_decode_token_count = 0;
  if (benchmark_info->GetTotalDecodeTurns() > 0) {
    last_decode_token_count =
        benchmark_info->GetDecodeTurn(benchmark_info->GetTotalDecodeTurns() - 1).num_tokens;
  }

  *out_prefill_token_count = last_prefill_token_count;
  *out_decode_token_count = last_decode_token_count;
  return true;
}

// Memory management helper
LITERTLM_EXPORT void LiteRtLm_FreeString(char* str) {
  if (str) {
    delete[] str;
  }
}

}  // extern "C"
