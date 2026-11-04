#ifndef LLAMA_DESKTOP_CORE_H_
#define LLAMA_DESKTOP_CORE_H_

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace llama_desktop {

struct DesktopChatMessage {
    std::string role;
    std::string content;
    std::string image_path;
};

struct DesktopGenerateRequest {
    std::string prompt;
    int max_tokens = 512;
    double temperature = 0.7;
    double top_p = 0.9;
    int top_k = 40;
    double min_p = 0.05;
    double typical_p = 1.0;
    double repeat_penalty = 1.1;
    double frequency_penalty = 0.0;
    double presence_penalty = 0.0;
    int repeat_last_n = 64;
    int mirostat = 0;
    double mirostat_tau = 5.0;
    double mirostat_eta = 0.1;
    int64_t seed = -1;
    bool penalize_newline = true;
};

struct DesktopChatRequest {
    std::vector<DesktopChatMessage> messages;
    std::string template_name;
    DesktopGenerateRequest gen;
};

struct DesktopContextInfo {
    int64_t tokens_used = 0;
    int64_t context_size = 0;
    double usage_percentage = 0.0;
};

struct DesktopGpuInfo {
    bool vulkan_supported = false;
    std::string gpu_name = "CPU";
    int64_t vulkan_api_version = -1;
    int64_t device_local_memory_bytes = -1;
    int64_t free_ram_bytes = -1;
    int64_t recommended_gpu_layers = 0;
};

struct DesktopWorkerStatus {
    std::string state = "stopped";
    std::string error;
    std::string peer;
    int64_t held_mb = 0;
    int64_t ram_mb = 0;
    double bandwidth = 0.0;
    bool throttled = false;
    std::string id;
    bool battery_unrestricted = true;
    int battery_pct = -1;
    bool charging = true;
};

class LlamaDesktopCore {
public:
    static LlamaDesktopCore& instance();

    // Host API
    std::string loadModel(
        const std::string& model_path,
        int n_threads,
        int context_size,
        int n_gpu_layers,
        const std::string& mmproj_path,
        std::function<void(double)> on_progress,
        std::string* out_vision_warning = nullptr);

    std::string generate(
        const DesktopGenerateRequest& req,
        std::function<void(const std::string&)> on_token);

    std::string generateChat(
        const DesktopChatRequest& req,
        std::function<void(const std::string&)> on_token);

    std::vector<std::string> getSupportedTemplates();
    void registerCustomTemplate(const std::string& name, const std::string& content);
    void unregisterCustomTemplate(const std::string& name);

    void stop();
    void dispose();
    bool isModelLoaded() const;
    DesktopContextInfo getContextInfo() const;
    void clearContext();
    void setSystemPromptLength(int length);
    DesktopGpuInfo detectGpu() const;

    // Cluster API
    std::string startWorker(int port, int threads, const std::string& cache_dir);
    void stopWorker();
    DesktopWorkerStatus workerStatus();
    int64_t localFreeRamMb() const;
    double benchmark(int threads);
    std::vector<std::pair<std::string, double>> lastLoadBuffers() const;
    bool probe(const std::string& endpoint, int timeout_ms = 2000) const;
    std::string setCluster(
        const std::vector<std::string>& endpoints,
        const std::vector<double>& weights,
        double local_weight);
    void setKeepScreenOn(bool on);

private:
    LlamaDesktopCore();
};

}  // namespace llama_desktop

#endif  // LLAMA_DESKTOP_CORE_H_
