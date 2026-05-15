#include <cstdint>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <string>
#include "distance.h"
#include "filter/densebit_index.h"
#include "nbr/nbr.h"
#include "filter/selector.h"
#include "filter/label.h"
#include "omp.h"

#include "utils/index_build_utils.h"
#include "utils.h"

namespace {
struct CalibrationOptions {
  std::string selector_type = "intersect";
  std::string threads = "1";
  std::string beamwidth = "4";
  std::string k = "10";
  std::string mem_l = "0";
  std::string search_l = "100";
};

std::string shell_quote(const std::string &value) {
  std::string quoted = "'";
  for (char ch : value) {
    if (ch == '\'') {
      quoted += "'\\''";
    } else {
      quoted += ch;
    }
  }
  quoted += "'";
  return quoted;
}

std::string repo_root_from_binary(const std::string &argv0) {
  const std::string marker = "/build/";
  const auto marker_pos = argv0.find(marker);
  if (marker_pos != std::string::npos) {
    return argv0.substr(0, marker_pos);
  }
  return ".";
}

bool parse_calibration_options(int argc, char **argv, int start_index, CalibrationOptions *options) {
  for (int index = start_index; index < argc; ++index) {
    const std::string arg(argv[index]);
    auto require_value = [&](const std::string &name) -> const char * {
      if (index + 1 >= argc) {
        LOG(ERROR) << "Missing value for " << name;
        return nullptr;
      }
      return argv[++index];
    };

    if (arg == "--calibration-selector-type") {
      const char *value = require_value(arg);
      if (value == nullptr) {
        return false;
      }
      options->selector_type = value;
      if (options->selector_type != "intersect" && options->selector_type != "subset"
          && options->selector_type != "range") {
        LOG(ERROR) << "Unsupported --calibration-selector-type: " << options->selector_type;
        return false;
      }
      continue;
    }
    if (arg == "--calibration-threads") {
      const char *value = require_value(arg);
      if (value == nullptr) {
        return false;
      }
      options->threads = value;
      continue;
    }
    if (arg == "--calibration-beamwidth") {
      const char *value = require_value(arg);
      if (value == nullptr) {
        return false;
      }
      options->beamwidth = value;
      continue;
    }
    if (arg == "--calibration-k") {
      const char *value = require_value(arg);
      if (value == nullptr) {
        return false;
      }
      options->k = value;
      continue;
    }
    if (arg == "--calibration-mem-l") {
      const char *value = require_value(arg);
      if (value == nullptr) {
        return false;
      }
      options->mem_l = value;
      continue;
    }
    if (arg == "--calibration-l-search") {
      const char *value = require_value(arg);
      if (value == nullptr) {
        return false;
      }
      options->search_l = value;
      continue;
    }

    LOG(ERROR) << "Unknown option: " << arg;
    return false;
  }
  return true;
}

bool write_sidecar_and_calibrate(const std::string &argv0, const std::string &index_type,
                                 const std::string &data_file, const std::string &index_prefix,
                                 const std::string &threads, const std::string &similarity,
                                 const std::string &nbr_type, pipeann::AbstractLabel *label,
                                 const CalibrationOptions &calibration_options) {
  (void) threads;
  auto *spmat_label = dynamic_cast<pipeann::SpmatLabel *>(label);
  if (spmat_label == nullptr) {
    return true;
  }

  const std::string sidecar_path = pipeann::DenseBitsetIndex::default_sidecar_path(index_prefix);
  LOG(INFO) << "Writing mixed densebit/posting sidecar: " << sidecar_path;
  try {
    pipeann::DenseBitsetIndex::write_atomically(sidecar_path, spmat_label->labels_.size(), spmat_label->nlabels(),
                                                spmat_label->labels_);
  } catch (const std::exception &e) {
    LOG(ERROR) << "Failed to write densebit sidecar: " << e.what();
    return false;
  }

  const std::string repo_root = repo_root_from_binary(argv0);
  const std::string script_path = repo_root + "/scripts/pipeann_hybrid_experiment.py";
  const std::string build_dir = repo_root + "/build";
  std::ostringstream command;
  command << "python3 "
          << shell_quote(script_path) << " calibrate-tau"
          << " --index-prefix " << shell_quote(index_prefix)
          << " --base-bin " << shell_quote(data_file)
          << " --index-type " << shell_quote(index_type)
          << " --selector-type " << shell_quote(calibration_options.selector_type)
          << " --similarity " << shell_quote(similarity)
          << " --nbr-type " << shell_quote(nbr_type)
          << " --build-dir " << shell_quote(build_dir)
          << " --threads " << shell_quote(calibration_options.threads)
          << " --beamwidth " << shell_quote(calibration_options.beamwidth)
          << " --k " << shell_quote(calibration_options.k)
          << " --mem-l " << shell_quote(calibration_options.mem_l)
          << " --search-l " << shell_quote(calibration_options.search_l)
          << " --cleanup-work-dir";

  LOG(INFO) << "Running post-build tau_m calibration: " << command.str();
  const int rc = std::system(command.str().c_str());
  if (rc != 0) {
    LOG(ERROR) << "Post-build tau_m calibration failed with status " << rc;
    return false;
  }
  return true;
}
}  // namespace

int main(int argc, char **argv) {
  if (argc < 11) {
    std::cout << "Usage: " << argv[0]
              << " <data_type (float/int8/uint8)>  <data_file.bin>"
                 " <index_prefix_path> <R>  <L>  <PQ_bytes>  <M>  <T>"
                 " <similarity metric (cosine/l2/mips) case sensitive> <nbr_type (pq/rabitq)> <(optional) label_type "
                 "(spmat)> <(optional) label_file.spmat>"
                 " [--calibration-selector-type intersect/subset/range]"
                 " [--calibration-threads N] [--calibration-beamwidth B]"
                 " [--calibration-k K] [--calibration-mem-l L] [--calibration-l-search L]."
                 " See README for more information on parameters."
              << std::endl;
    return -1;
  } else {
    std::string dist_metric(argv[9]);

    pipeann::Metric m = pipeann::get_metric(dist_metric);

    std::string nbr_type = argv[10];
    const bool has_label_args = argc > 12 && std::string(argv[11]).rfind("--", 0) != 0;
    std::string label_type = has_label_args ? argv[11] : "null";
    std::string label_file = has_label_args ? argv[12] : "";
    const char *label_source_file = label_file.empty() ? nullptr : label_file.c_str();
    CalibrationOptions calibration_options;
    if (!parse_calibration_options(argc, argv, has_label_args ? 13 : 11, &calibration_options)) {
      return -1;
    }

    pipeann::AbstractLabel *label = nullptr;
    if (label_type == "spmat") {
      if (label_file.empty()) {
        LOG(ERROR) << "Error. label_file is required for spmat label writer.";
        crash();
      }
      label = new pipeann::SpmatLabel(label_file);
    }

    bool build_ok = false;
    if (std::string(argv[1]) == std::string("float")) {
      pipeann::AbstractNeighbor<float> *nbr_handler = pipeann::get_nbr_handler<float>(m, nbr_type);

      build_ok = pipeann::build_disk_index<float>(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]),
                                                  std::stoi(argv[7]), std::stoi(argv[8]), std::stoi(argv[6]), m,
                                                  nullptr, nbr_handler, label, label_source_file);
    } else if (std::string(argv[1]) == std::string("int8")) {
      pipeann::AbstractNeighbor<int8_t> *nbr_handler = pipeann::get_nbr_handler<int8_t>(m, nbr_type);
      build_ok = pipeann::build_disk_index<int8_t>(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]),
                                                   std::stoi(argv[7]), std::stoi(argv[8]), std::stoi(argv[6]), m,
                                                   nullptr, nbr_handler, label, label_source_file);
    } else if (std::string(argv[1]) == std::string("uint8")) {
      pipeann::AbstractNeighbor<uint8_t> *nbr_handler = pipeann::get_nbr_handler<uint8_t>(m, nbr_type);
      build_ok = pipeann::build_disk_index<uint8_t>(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]),
                                                    std::stoi(argv[7]), std::stoi(argv[8]), std::stoi(argv[6]), m,
                                                    nullptr, nbr_handler, label, label_source_file);
    } else {
      LOG(ERROR) << "Error. wrong file type";
      delete label;
      return -1;
    }

    if (build_ok) {
      build_ok = write_sidecar_and_calibrate(argv[0], argv[1], argv[2], argv[3], argv[8], dist_metric, nbr_type, label,
                                             calibration_options);
    }
    delete label;
    return build_ok ? 0 : -1;
  }
}
