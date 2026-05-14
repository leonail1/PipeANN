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

bool write_sidecar_and_calibrate(const std::string &argv0, const std::string &index_type,
                                 const std::string &data_file, const std::string &index_prefix,
                                 const std::string &threads, const std::string &similarity,
                                 const std::string &nbr_type, pipeann::AbstractLabel *label) {
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
          << " --selector-type intersect"
          << " --similarity " << shell_quote(similarity)
          << " --nbr-type " << shell_quote(nbr_type)
          << " --build-dir " << shell_quote(build_dir)
          << " --threads 1"
          << " --beamwidth 4"
          << " --k 10"
          << " --mem-l 0"
          << " --search-l 100"
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
                 "(spmat)> <(optional) label_file.spmat>."
                 " See README for more information on parameters."
              << std::endl;
    return -1;
  } else {
    std::string dist_metric(argv[9]);

    pipeann::Metric m = pipeann::get_metric(dist_metric);

    std::string nbr_type = argv[10];
    std::string label_type = argc > 12 ? argv[11] : "null";
    std::string label_file = argc > 12 ? argv[12] : "";
    const char *label_source_file = label_file.empty() ? nullptr : label_file.c_str();

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
      build_ok = write_sidecar_and_calibrate(argv[0], argv[1], argv[2], argv[3], argv[8], dist_metric, nbr_type, label);
    }
    delete label;
    return build_ok ? 0 : -1;
  }
}
