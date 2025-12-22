// runtime/engine/litert_lm_main.cc
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <utility>
#include <variant>

// Add these two single-header libraries to your project
#include "httplib.h"
// Don't include json.hpp separately - use nlohmann from includes
#include "absl/base/log_severity.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/globals.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/status_macros.h"

// Define flags (keep necessary ones)
ABSL_FLAG(std::string, model_path, "", "Path to the .litertlm model file.");
ABSL_FLAG(std::string, tokenizer_path, "", "Path to the tokenizer (optional if embedded).");
ABSL_FLAG(int, port, 11434, "Port to run the server on.");

using ::litert::lm::Backend;
using ::litert::lm::Conversation;
using ::litert::lm::ConversationConfig;
using ::litert::lm::Engine;
using ::litert::lm::EngineSettings;
using ::litert::lm::InputText;
using ::litert::lm::InputData;
using ::litert::lm::JsonMessage;
using ::litert::lm::Message;
using ::litert::lm::ModelAssets;
using ::litert::lm::Responses;
using ::litert::lm::SessionConfig;
using namespace litert::lm;

int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);

    std::string model_path = absl::GetFlag(FLAGS_model_path);
    if (model_path.empty()) {
        std::cerr << "Error: --model_path is required." << std::endl;
        return 1;
    }

    // 1. Create Model Assets
    std::cout << "Loading model from: " << model_path << " ..." << std::endl;
    
    auto model_assets_result = ModelAssets::Create(model_path);
    if (!model_assets_result.ok()) {
        std::cerr << "Failed to create model assets: " << model_assets_result.status().message() << std::endl;
        return 1;
    }
    auto model_assets = std::move(model_assets_result.value());

    // 2. Create Engine Settings
    auto engine_settings_result = EngineSettings::CreateDefault(
        std::move(model_assets), Backend::CPU);
    if (!engine_settings_result.ok()) {
        std::cerr << "Failed to create engine settings: " << engine_settings_result.status().message() << std::endl;
        return 1;
    }
    auto engine_settings = std::move(engine_settings_result.value());

    // 3. Create the Engine
    auto engine_result = Engine::CreateEngine(std::move(engine_settings));
    if (!engine_result.ok()) {
        std::cerr << "Failed to create engine: " << engine_result.status().message() << std::endl;
        return 1;
    }
    auto engine = std::move(engine_result.value());

    std::cout << "Model loaded successfully. Starting server on port " << absl::GetFlag(FLAGS_port) << "..." << std::endl;

    // 4. Setup HTTP Server
    httplib::Server svr;

    // POST /api/chat - Ollama Compatible Endpoint
    svr.Post("/api/chat", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            auto body = nlohmann::json::parse(req.body);
            bool stream = body.value("stream", false); // Streaming not implemented in this simple example
            
            if (!body.contains("messages")) {
                res.status = 400;
                res.set_content("Missing 'messages' field", "text/plain");
                return;
            }

            // Create a new session for this request (Stateless API behavior)
            auto session_config = SessionConfig::CreateDefault();
            
            auto session_result = engine->CreateSession(session_config);
            if (!session_result.ok()) {
                res.status = 500;
                res.set_content("Failed to create session", "text/plain");
                return;
            }
            auto session = std::move(session_result.value());

            auto messages = body["messages"];
            std::string prompt;

            // Extract the last user message
            if (!messages.empty()) {
                prompt = messages.back().value("content", "");
            }

            // Execute Inference using GenerateContent
            // Build inputs vector manually to avoid copy constructor issues
            std::vector<InputData> inputs;
            inputs.emplace_back(InputText(prompt));
            
            auto result = session->GenerateContent(std::move(inputs)); 
            
            if (!result.ok()) {
                res.status = 500;
                res.set_content(std::string(result.status().message()), "text/plain");
                return;
            }

            // Extract response text from Responses
            // Responses has operator<< overload, so convert to string
            std::ostringstream oss;
            oss << result.value();
            std::string last_response = oss.str();

            // Format Response (Ollama format)
            nlohmann::json response_json = {
                {"model", "litert-model"},
                {"created_at", "2023-01-01T00:00:00Z"}, // Dummy timestamp
                {"message", {
                    {"role", "assistant"},
                    {"content", last_response}
                }},
                {"done", true}
            };

            res.set_content(response_json.dump(), "application/json");

        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(e.what(), "text/plain");
        }
    });

    // Health check
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("LiteRT-LM Server is running", "text/plain");
    });

    svr.listen("0.0.0.0", absl::GetFlag(FLAGS_port));

    return 0;
}