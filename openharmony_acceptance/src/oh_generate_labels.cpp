#include "oh_common.h"

namespace {

uint32_t eq_label(size_t i) {
  return static_cast<uint32_t>(i);
}

uint32_t int_a_label(size_t i) {
  return static_cast<uint32_t>(100 + i * 2);
}

uint32_t int_b_label(size_t i) {
  return static_cast<uint32_t>(101 + i * 2);
}

std::string selector_config(const std::string &type, const std::string &base_file, const std::string &query_file) {
  if (type == "match_all") {
    return "{}\n";
  }
  std::string qtype = type == "intersect" ? "label_and" : (type == "equality" ? "label" : "range");
  uint32_t key = type == "range" ? 1 : 0;
  std::ostringstream out;
  out << "{\n"
      << "  \"base\": [{\"key\": " << key << ", \"type\": \"" << (type == "range" ? "range" : "label")
      << "\", \"file\": \"" << oh::json_escape(base_file) << "\"}],\n"
      << "  \"query\": {\"key\": " << key << ", \"base_key\": " << key << ", \"type\": \"" << qtype
      << "\", \"file\": \"" << oh::json_escape(query_file) << "\"}\n"
      << "}\n";
  return out.str();
}

}  // namespace

int main(int argc, char **argv) {
  try {
    oh::Args args(argc, argv);
    const uint64_t npoints = args.u64("npoints", 1000000);
    const uint64_t nqueries = args.u64("nqueries", 1000);
    const auto out_dir = std::filesystem::path(args.get("out-dir", "acceptance_work/labels"));
    const auto index_prefix = args.get("index-prefix", "acceptance_work/index/sift1m");
    std::filesystem::create_directories(out_dir);

    auto sels = oh::selectivities();
    auto rank = oh::stable_ranks(npoints);

    const int64_t n_label_cols = 128;
    std::vector<std::vector<int32_t>> label_rows(npoints);
    std::vector<std::vector<float>> label_vals(npoints);
    std::vector<uint32_t> range_values(npoints);
    for (uint64_t id = 0; id < npoints; ++id) {
      range_values[id] = static_cast<uint32_t>(rank[id]);
      for (size_t s = 0; s < sels.size(); ++s) {
        uint32_t count = oh::target_count(npoints, sels[s].second);
        if (rank[id] < count) {
          label_rows[id].push_back(static_cast<int32_t>(eq_label(s)));
          label_vals[id].push_back(1.0f);
          label_rows[id].push_back(static_cast<int32_t>(int_a_label(s)));
          label_vals[id].push_back(1.0f);
          label_rows[id].push_back(static_cast<int32_t>(int_b_label(s)));
          label_vals[id].push_back(1.0f);
        }
      }
    }

    auto label_spmat = out_dir / "base_labels.spmat";
    auto range_bin = out_dir / "base_range.bin";
    oh::write_spmat(label_spmat, static_cast<int64_t>(npoints), n_label_cols, label_rows, label_vals);
    oh::write_bin_matrix<uint32_t>(range_bin, range_values, static_cast<uint32_t>(npoints), 1);

    const std::string built_label_index = index_prefix + ".label.0";
    const std::string built_range_index = index_prefix + ".label.1";
    auto manifest_path = out_dir / "selector_manifest.csv";
    std::ofstream manifest(manifest_path);
    manifest << "selector_id,selector_type,target_selectivity,candidate_count,query_file,label_config\n";

    auto write_query_label = [&](const std::string &name, const std::vector<int32_t> &labels) {
      std::vector<std::vector<int32_t>> rows(nqueries, labels);
      std::vector<std::vector<float>> vals(nqueries, std::vector<float>(labels.size(), 1.0f));
      auto path = out_dir / (name + ".query.spmat");
      oh::write_spmat(path, static_cast<int64_t>(nqueries), n_label_cols, rows, vals);
      return path;
    };

    auto write_query_range = [&](const std::string &name, uint32_t count) {
      std::vector<std::vector<int32_t>> rows(nqueries, std::vector<int32_t>{0, 0});
      std::vector<std::vector<float>> vals(nqueries, std::vector<float>{0.0f, static_cast<float>(count)});
      auto path = out_dir / (name + ".query.spmat");
      oh::write_spmat(path, static_cast<int64_t>(nqueries), 1, rows, vals);
      return path;
    };

    for (size_t s = 0; s < sels.size(); ++s) {
      uint32_t count = oh::target_count(npoints, sels[s].second);
      const std::string suffix = sels[s].first;

      {
        auto q = write_query_label("equality_" + suffix, {static_cast<int32_t>(eq_label(s))});
        auto cfg = out_dir / ("equality_" + suffix + ".json");
        oh::write_text(cfg, selector_config("equality", built_label_index, q.string()));
        manifest << "equality_" << suffix << ",equality," << sels[s].second << "," << count << "," << q.string()
                 << "," << cfg.string() << "\n";
      }
      {
        auto q = write_query_label("intersect_" + suffix,
                                   {static_cast<int32_t>(int_a_label(s)), static_cast<int32_t>(int_b_label(s))});
        auto cfg = out_dir / ("intersect_" + suffix + ".json");
        oh::write_text(cfg, selector_config("intersect", built_label_index, q.string()));
        manifest << "intersect_" << suffix << ",intersect," << sels[s].second << "," << count << "," << q.string()
                 << "," << cfg.string() << "\n";
      }
      {
        auto q = write_query_range("range_" + suffix, count);
        auto cfg = out_dir / ("range_" + suffix + ".json");
        oh::write_text(cfg, selector_config("range", built_range_index, q.string()));
        manifest << "range_" << suffix << ",range," << sels[s].second << "," << count << "," << q.string()
                 << "," << cfg.string() << "\n";
      }
    }

    auto match_all_cfg = out_dir / "match_all_s100.json";
    oh::write_text(match_all_cfg, "{}\n");
    manifest << "match_all_s100,match_all,1," << npoints << ",," << match_all_cfg.string() << "\n";

    oh::write_text(out_dir / "build_attrs.args",
                   "label_spmat " + label_spmat.string() + " range " + range_bin.string() + "\n");
    std::cout << "{\"label_spmat\":\"" << label_spmat.string() << "\",\"range_bin\":\"" << range_bin.string()
              << "\",\"manifest\":\"" << manifest_path.string() << "\"}\n";
  } catch (const std::exception &e) {
    std::cerr << "oh_generate_labels failed: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
