#include "oh_common.h"

#include <cctype>
#include <cmath>
#include <limits>

#include "utils/picojson.h"

namespace {

struct Failure {
  std::string check;
  std::string detail;
};

struct MetricStats {
  uint64_t rows = 0;
  double worst_avg_latency_ms = 0.0;
  double min_recall = std::numeric_limits<double>::infinity();
  double max_delete_ms_per_vector = 0.0;
};

std::string read_text_or_empty(const std::filesystem::path &path) {
  std::ifstream reader(path);
  if (!reader) {
    return "";
  }
  std::ostringstream out;
  out << reader.rdbuf();
  return out.str();
}

std::vector<std::string> read_lines(const std::filesystem::path &path) {
  std::vector<std::string> lines;
  std::ifstream reader(path);
  std::string line;
  while (std::getline(reader, line)) {
    if (!line.empty()) {
      lines.push_back(line);
    }
  }
  return lines;
}

bool parse_json_value(const std::string &text, picojson::value &value) {
  if (text.empty()) {
    return false;
  }
  std::string err = picojson::parse(value, text);
  return err.empty() && !value.is<picojson::null>();
}

double extract_number(const std::string &text, const std::string &key, double fallback) {
  const std::string needle = "\"" + key + "\":";
  auto pos = text.find(needle);
  if (pos == std::string::npos) {
    return fallback;
  }
  pos += needle.size();
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
    ++pos;
  }
  size_t end = pos;
  while (end < text.size()) {
    char c = text[end];
    if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') {
      ++end;
    } else {
      break;
    }
  }
  if (end == pos) {
    return fallback;
  }
  return std::stod(text.substr(pos, end - pos));
}

std::string extract_string(const std::string &text, const std::string &key, const std::string &fallback) {
  const std::string needle = "\"" + key + "\":\"";
  auto pos = text.find(needle);
  if (pos == std::string::npos) {
    return fallback;
  }
  pos += needle.size();
  auto end = text.find('"', pos);
  if (end == std::string::npos) {
    return fallback;
  }
  return text.substr(pos, end - pos);
}

int64_t parse_max_rss_kbytes(const std::filesystem::path &time_file) {
  std::ifstream reader(time_file);
  std::string line;
  const std::string prefix = "Maximum resident set size (kbytes):";
  while (std::getline(reader, line)) {
    auto pos = line.find(prefix);
    if (pos != std::string::npos) {
      pos += prefix.size();
      while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) {
        ++pos;
      }
      return std::stoll(line.substr(pos));
    }
  }
  return -1;
}

void add_failure(std::vector<Failure> &failures, const std::string &check, const std::string &detail) {
  failures.push_back({check, detail});
}

std::string json_number(double value) {
  if (!std::isfinite(value)) {
    return "null";
  }
  std::ostringstream out;
  out << value;
  return out.str();
}

MetricStats check_recall_latency_jsonl(const std::filesystem::path &path, const std::string &check_name,
                                       double recall_min, double latency_lt, std::vector<Failure> &failures) {
  MetricStats stats;
  auto lines = read_lines(path);
  if (lines.empty()) {
    add_failure(failures, check_name, "missing_or_empty:" + path.string());
    return stats;
  }
  for (const auto &line : lines) {
    picojson::value parsed;
    if (!parse_json_value(line, parsed)) {
      add_failure(failures, check_name, "invalid_json_line:" + path.string());
      continue;
    }
    ++stats.rows;
    const double recall = extract_number(line, "recall_at_10", -1.0);
    const double avg = extract_number(line, "avg_latency_ms", std::numeric_limits<double>::infinity());
    stats.worst_avg_latency_ms = std::max(stats.worst_avg_latency_ms, avg);
    stats.min_recall = std::min(stats.min_recall, recall);
    if (!(recall >= recall_min) || !(avg < latency_lt)) {
      std::string selector = extract_string(line, "selector_id", "unknown");
      std::string cycle = std::to_string(static_cast<int64_t>(extract_number(line, "cycle", -1)));
      add_failure(failures, check_name,
                  "selector=" + selector + ",cycle=" + cycle + ",recall=" + std::to_string(recall) +
                      ",avg_latency_ms=" + std::to_string(avg));
    }
  }
  return stats;
}

MetricStats check_foreground_jsonl(const std::filesystem::path &path, const std::string &check_name, double latency_lt,
                                   std::vector<Failure> &failures) {
  MetricStats stats;
  auto lines = read_lines(path);
  if (lines.empty()) {
    add_failure(failures, check_name, "missing_or_empty:" + path.string());
    return stats;
  }
  for (const auto &line : lines) {
    picojson::value parsed;
    if (!parse_json_value(line, parsed)) {
      add_failure(failures, check_name, "invalid_json_line:" + path.string());
      continue;
    }
    ++stats.rows;
    const double avg = extract_number(line, "avg_latency_ms", std::numeric_limits<double>::infinity());
    stats.worst_avg_latency_ms = std::max(stats.worst_avg_latency_ms, avg);
    if (!(avg < latency_lt)) {
      std::string phase = extract_string(line, "phase", "unknown");
      std::string cycle = std::to_string(static_cast<int64_t>(extract_number(line, "cycle", -1)));
      add_failure(failures, check_name,
                  "phase=" + phase + ",cycle=" + cycle + ",avg_latency_ms=" + std::to_string(avg));
    }
  }
  return stats;
}

MetricStats check_delete_jsonl(const std::filesystem::path &path, const std::string &check_name,
                               double delete_ms_per_vector_lte,
                               std::vector<Failure> &failures) {
  MetricStats stats;
  auto lines = read_lines(path);
  if (lines.empty()) {
    add_failure(failures, check_name, "missing_or_empty:" + path.string());
    return stats;
  }
  for (const auto &line : lines) {
    picojson::value parsed;
    if (!parse_json_value(line, parsed)) {
      add_failure(failures, check_name, "invalid_json_line:" + path.string());
      continue;
    }
    ++stats.rows;
    double value = extract_number(line, "delete_ms_per_vector", std::numeric_limits<double>::infinity());
    stats.max_delete_ms_per_vector = std::max(stats.max_delete_ms_per_vector, value);
    if (!(value <= delete_ms_per_vector_lte)) {
      std::string cycle = std::to_string(static_cast<int64_t>(extract_number(line, "cycle", -1)));
      add_failure(failures, check_name, "cycle=" + cycle + ",delete_ms_per_vector=" + std::to_string(value));
    }
  }
  return stats;
}

void check_nonempty_artifact(const std::filesystem::path &path, const std::string &check_name,
                             std::vector<Failure> &failures) {
  if (read_lines(path).empty()) {
    add_failure(failures, check_name, "missing_or_empty:" + path.string());
  }
}

void write_summary(const std::filesystem::path &path, const std::vector<Failure> &failures, double space_ratio,
                   const MetricStats &static_stats, const MetricStats &checkpoint_stats,
                   const MetricStats &foreground_stats, const MetricStats &foreground_delete_stats,
                   const MetricStats &batch_delete_stats, double single_latency_ms,
                   int64_t rss_bytes, double space_lt, double recall_min, double latency_lt,
                   double delete_ms_per_vector_lte, int64_t rss_lt) {
  oh::ensure_parent(path);
  std::ofstream out(path);
  out << "{\n";
  out << "  \"pass\": " << (failures.empty() ? "true" : "false") << ",\n";
  out << "  \"thresholds\": {\"space_expansion_lt\": " << space_lt << ", \"recall_at_10_min\": " << recall_min
      << ", \"avg_latency_ms_lt\": " << latency_lt << ", \"delete_ms_per_vector_lte\": "
      << delete_ms_per_vector_lte << ", \"single_query_max_rss_bytes_lt\": " << rss_lt << "},\n";
  out << "  \"metrics\": {\n";
  out << "    \"space_expansion_ratio\": " << json_number(space_ratio) << ",\n";
  out << "    \"static_rows\": " << static_stats.rows << ", \"static_min_recall\": "
      << json_number(static_stats.min_recall) << ", \"static_worst_avg_latency_ms\": "
      << json_number(static_stats.worst_avg_latency_ms) << ",\n";
  out << "    \"dynamic_batch_checkpoint_rows\": " << checkpoint_stats.rows << ", \"dynamic_batch_checkpoint_min_recall\": "
      << json_number(checkpoint_stats.min_recall) << ", \"dynamic_batch_checkpoint_worst_avg_latency_ms\": "
      << json_number(checkpoint_stats.worst_avg_latency_ms) << ",\n";
  out << "    \"dynamic_foreground_rows\": " << foreground_stats.rows << ", \"dynamic_foreground_worst_avg_latency_ms\": "
      << json_number(foreground_stats.worst_avg_latency_ms) << ",\n";
  out << "    \"dynamic_foreground_delete_rows\": " << foreground_delete_stats.rows
      << ", \"dynamic_foreground_max_delete_ms_per_vector\": "
      << json_number(foreground_delete_stats.max_delete_ms_per_vector) << ",\n";
  out << "    \"dynamic_batch_delete_rows\": " << batch_delete_stats.rows
      << ", \"dynamic_batch_max_delete_ms_per_vector\": "
      << json_number(batch_delete_stats.max_delete_ms_per_vector) << ",\n";
  out << "    \"single_query_latency_ms\": " << json_number(single_latency_ms)
      << ", \"single_query_max_rss_bytes\": " << rss_bytes << "\n";
  out << "  },\n";
  out << "  \"failures\": [";
  for (size_t i = 0; i < failures.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << "\n    {\"check\":\"" << oh::json_escape(failures[i].check) << "\",\"detail\":\""
        << oh::json_escape(failures[i].detail) << "\"}";
  }
  if (!failures.empty()) {
    out << "\n  ";
  }
  out << "]\n";
  out << "}\n";
}

}  // namespace

int main(int argc, char **argv) {
  try {
    oh::Args args(argc, argv);
    const auto results_dir = std::filesystem::path(args.get("results-dir"));
    const auto out_json = std::filesystem::path(args.get("out-json", (results_dir / "acceptance_summary.json").string()));
    const double space_lt = args.f64("space-expansion-lt", 2.0);
    const double recall_min = args.f64("recall-min", 98.0);
    const double latency_lt = args.f64("latency-lt", 10.0);
    const double delete_ms_per_vector_lte = args.f64("delete-ms-per-vector-lte", 0.5);
    const int64_t rss_lt = static_cast<int64_t>(args.u64("single-query-max-rss-bytes-lt", 30000000));
    const uint64_t dynamic_foreground_cycles = args.u64("dynamic-foreground-cycles", 1);
    const uint64_t dynamic_batch_cycles = args.u64("dynamic-batch-cycles", 5);
    if (results_dir.empty()) {
      throw std::runtime_error("--results-dir is required");
    }

    std::vector<Failure> failures;
    std::string space_text = read_text_or_empty(results_dir / "space_audit.json");
    if (space_text.empty()) {
      add_failure(failures, "space", "missing:" + (results_dir / "space_audit.json").string());
    } else {
      picojson::value parsed_space;
      if (!parse_json_value(space_text, parsed_space)) {
        add_failure(failures, "space", "invalid_json:" + (results_dir / "space_audit.json").string());
      }
    }
    const double space_ratio = extract_number(space_text, "expansion_ratio", std::numeric_limits<double>::infinity());
    if (!(space_ratio < space_lt)) {
      add_failure(failures, "space", "expansion_ratio=" + std::to_string(space_ratio));
    }

    auto static_stats =
        check_recall_latency_jsonl(results_dir / "static_filtered.jsonl", "static", recall_min, latency_lt, failures);
    auto checkpoint_stats = check_recall_latency_jsonl(results_dir / "dynamic_batch_checkpoint_search.jsonl",
                                                       "dynamic_batch_checkpoint", recall_min, latency_lt, failures);
    auto foreground_stats =
        check_foreground_jsonl(results_dir / "dynamic_foreground_latency.jsonl", "dynamic_foreground", latency_lt, failures);
    auto foreground_delete_stats = check_delete_jsonl(results_dir / "dynamic_foreground_chain.jsonl",
                                                      "dynamic_foreground_delete", delete_ms_per_vector_lte, failures);
    auto batch_delete_stats =
        check_delete_jsonl(results_dir / "dynamic_batch_chain.jsonl", "dynamic_batch_delete", delete_ms_per_vector_lte,
                           failures);
    check_nonempty_artifact(results_dir / "dynamic_foreground_progress.jsonl", "dynamic_foreground_progress", failures);
    check_nonempty_artifact(results_dir / "dynamic_batch_progress.jsonl", "dynamic_batch_progress", failures);
    if (foreground_delete_stats.rows < dynamic_foreground_cycles) {
      add_failure(failures, "dynamic_foreground_delete",
                  "rows=" + std::to_string(foreground_delete_stats.rows) +
                      ",expected_at_least=" + std::to_string(dynamic_foreground_cycles));
    }
    if (batch_delete_stats.rows < dynamic_batch_cycles) {
      add_failure(failures, "dynamic_batch_delete",
                  "rows=" + std::to_string(batch_delete_stats.rows) +
                      ",expected_at_least=" + std::to_string(dynamic_batch_cycles));
    }
    if (static_stats.rows > 0 && checkpoint_stats.rows < static_stats.rows * dynamic_batch_cycles) {
      add_failure(failures, "dynamic_batch_checkpoint",
                  "rows=" + std::to_string(checkpoint_stats.rows) + ",expected_at_least=" +
                      std::to_string(static_stats.rows * dynamic_batch_cycles));
    }

    auto single_lines = read_lines(results_dir / "single_query_resource.jsonl");
    double single_latency_ms = std::numeric_limits<double>::infinity();
    if (single_lines.empty()) {
      add_failure(failures, "single_query", "missing_or_empty:" + (results_dir / "single_query_resource.jsonl").string());
    } else {
      picojson::value parsed_single;
      if (!parse_json_value(single_lines.back(), parsed_single)) {
        add_failure(failures, "single_query", "invalid_json_line:" + (results_dir / "single_query_resource.jsonl").string());
      }
      single_latency_ms = extract_number(single_lines.back(), "latency_ms", std::numeric_limits<double>::infinity());
      if (!(single_latency_ms < latency_lt)) {
        add_failure(failures, "single_query", "latency_ms=" + std::to_string(single_latency_ms));
      }
    }

    int64_t rss_bytes = -1;
    int64_t rss_kbytes = parse_max_rss_kbytes(results_dir / "single_query_time.txt");
    if (rss_kbytes < 0) {
      add_failure(failures, "rss", "missing_max_rss:" + (results_dir / "single_query_time.txt").string());
    } else {
      rss_bytes = rss_kbytes * 1024;
      if (!(rss_bytes < rss_lt)) {
        add_failure(failures, "rss", "max_rss_bytes=" + std::to_string(rss_bytes));
      }
    }

    write_summary(out_json, failures, space_ratio, static_stats, checkpoint_stats, foreground_stats,
                  foreground_delete_stats, batch_delete_stats,
                  single_latency_ms, rss_bytes, space_lt, recall_min, latency_lt, delete_ms_per_vector_lte, rss_lt);
    std::cout << "Summary written to " << out_json << " pass=" << (failures.empty() ? "true" : "false") << std::endl;
    return failures.empty() ? 0 : 2;
  } catch (const std::exception &e) {
    std::cerr << "oh_summarize_results failed: " << e.what() << std::endl;
    return 1;
  }
}
