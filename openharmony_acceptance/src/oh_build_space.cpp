#include "oh_common.h"

int main(int argc, char **argv) {
  try {
    oh::Args args(argc, argv);
    const auto raw = std::filesystem::path(args.get("raw"));
    const auto prefix = args.get("index-prefix");
    const auto out_json = std::filesystem::path(args.get("out-json", "results/space_audit.json"));
    const auto out_csv = std::filesystem::path(args.get("out-csv", "results/space_audit.csv"));

    if (raw.empty() || prefix.empty()) {
      throw std::runtime_error("--raw and --index-prefix are required");
    }

    std::vector<std::filesystem::path> core_files = {
        prefix + "_disk.index",
        prefix + "_disk.index.tags",
        prefix + "_pq_pivots.bin",
        prefix + "_pq_compressed.bin",
    };

    uint64_t raw_bytes = oh::file_size_or_zero(raw);
    uint64_t index_bytes = 0;
    std::ostringstream files_json;
    files_json << "[";
    bool first = true;
    for (const auto &file : core_files) {
      uint64_t bytes = oh::file_size_or_zero(file);
      index_bytes += bytes;
      if (!first) {
        files_json << ",";
      }
      first = false;
      files_json << "{\"path\":\"" << oh::json_escape(file.string()) << "\",\"bytes\":" << bytes << "}";
    }
    files_json << "]";
    double ratio = raw_bytes == 0 ? 0.0 : static_cast<double>(index_bytes) / static_cast<double>(raw_bytes);

    oh::write_text(out_json,
                   "{\n  \"raw_bytes\": " + std::to_string(raw_bytes) + ",\n  \"index_bytes\": " +
                       std::to_string(index_bytes) + ",\n  \"expansion_ratio\": " + std::to_string(ratio) +
                       ",\n  \"pass\": " + std::string(ratio < 2.0 ? "true" : "false") + ",\n  \"core_files\": " +
                       files_json.str() + "\n}\n");
    oh::write_text(out_csv, "raw_bytes,index_bytes,expansion_ratio,pass\n" + std::to_string(raw_bytes) + "," +
                                std::to_string(index_bytes) + "," + std::to_string(ratio) + "," +
                                std::string(ratio < 2.0 ? "true" : "false") + "\n");
    std::cout << "space expansion ratio: " << ratio << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "oh_build_space failed: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
