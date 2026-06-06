#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <nlohmann/json.hpp>
#include <omp.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cctype>
#include <ctime>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr const char* kBackendExact = "faiss_exact_subset";
constexpr const char* kBackendHybrid = "faiss_ivf_exact_hybrid";
constexpr const char* kAnnIndexFile = "faiss_ivf.index";
constexpr const char* kRuntimeCacheDir = "runtime_cache";
constexpr int64_t kAnnMinPoints = 100000;
constexpr int64_t kAnnNlist = 4096;
constexpr int64_t kAnnTrainPoints = 200000;
constexpr int64_t kAnnNprobe = 224;
constexpr int64_t kAnnFullSelectivityNprobe = 128;
constexpr int64_t kAnnRerankK = 500;
constexpr int64_t kAnnRetryRerankK = 4000;
constexpr int64_t kSingleQueryNprobe = 1;
constexpr int64_t kSingleQueryRerankK = 20;
constexpr int64_t kSingleQueryRetryRerankK = 1000;
constexpr int64_t kExactCandidateLimit = 10000;

const std::vector<std::string> kSelectivityFields = {
    "eq_s0001", "eq_s001", "eq_s01", "eq_s05", "eq_s10", "eq_s25", "eq_s50", "eq_s100",
    "int_s0001_a", "int_s0001_b", "int_s001_a", "int_s001_b", "int_s01_a", "int_s01_b",
    "int_s05_a", "int_s05_b", "int_s10_a", "int_s10_b", "int_s25_a", "int_s25_b",
    "int_s50_a", "int_s50_b", "int_s100_a", "int_s100_b"};

constexpr const char* kRangeField = "range_uniform";

struct Matrix {
    int64_t rows = 0;
    int64_t dim = 0;
    std::vector<float> data;

    const float* row(int64_t i) const {
        return data.data() + i * dim;
    }
};

struct Segment {
    int64_t segment_id = 0;
    fs::path vectors_path;
    fs::path ids_path;
    int64_t count = 0;
    bool reference_only = false;
};

struct LabelState {
    std::vector<int64_t> ids;
    std::vector<uint8_t> live;
    std::unordered_map<std::string, std::vector<uint8_t>> fields;
    std::vector<float> range_uniform;
};

struct SearchRow {
    int query_id = 0;
    std::vector<int64_t> ids;
    double latency_ms = 0.0;
    std::string backend;
    int64_t candidate_count = 0;
};

class FileLock {
  public:
    FileLock(const fs::path& state_dir, bool exclusive) {
        fs::create_directories(state_dir);
        fs::path lock_path = state_dir / "adapter.lock";
        fd_ = ::open(lock_path.c_str(), O_CREAT | O_RDWR, 0644);
        if (fd_ < 0) {
            throw std::runtime_error("cannot open lock file: " + lock_path.string());
        }
        int op = exclusive ? LOCK_EX : LOCK_SH;
        if (::flock(fd_, op) != 0) {
            ::close(fd_);
            throw std::runtime_error("cannot lock state file: " + lock_path.string());
        }
    }

    ~FileLock() {
        if (fd_ >= 0) {
            ::flock(fd_, LOCK_UN);
            ::close(fd_);
        }
    }

    FileLock(const FileLock&) = delete;
    FileLock& operator=(const FileLock&) = delete;

  private:
    int fd_ = -1;
};

std::string require_arg(const std::unordered_map<std::string, std::string>& args, const std::string& key) {
    auto it = args.find(key);
    if (it == args.end() || it->second.empty()) {
        throw std::runtime_error("missing required argument: --" + key);
    }
    return it->second;
}

std::unordered_map<std::string, std::string> parse_options(int argc, char** argv, int start) {
    std::unordered_map<std::string, std::string> out;
    for (int i = start; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0) {
            throw std::runtime_error("unexpected argument: " + key);
        }
        key = key.substr(2);
        if (i + 1 >= argc) {
            throw std::runtime_error("missing value for --" + key);
        }
        out[key] = argv[++i];
    }
    return out;
}

void ensure_parent(const fs::path& path) {
    if (!path.parent_path().empty()) {
        fs::create_directories(path.parent_path());
    }
}

template <typename T>
void write_binary_vector(const fs::path& path, const std::vector<T>& values) {
    ensure_parent(path);
    fs::path tmp = path;
    tmp += ".tmp";
    std::ofstream out(tmp, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot write: " + tmp.string());
    }
    if (!values.empty()) {
        out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
    }
    out.close();
    fs::rename(tmp, path);
}

template <typename T>
std::vector<T> read_binary_vector(const fs::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("cannot read: " + path.string());
    }
    auto bytes = in.tellg();
    if (bytes < 0 || static_cast<uint64_t>(bytes) % sizeof(T) != 0) {
        throw std::runtime_error("bad binary vector size: " + path.string());
    }
    std::vector<T> values(static_cast<size_t>(bytes) / sizeof(T));
    in.seekg(0);
    if (!values.empty()) {
        in.read(reinterpret_cast<char*>(values.data()), bytes);
    }
    return values;
}

void write_json(const fs::path& path, const json& payload) {
    ensure_parent(path);
    fs::path tmp = path;
    tmp += ".tmp";
    std::ofstream out(tmp);
    if (!out) {
        throw std::runtime_error("cannot write json: " + tmp.string());
    }
    out << std::setw(2) << payload << "\n";
    out.close();
    fs::rename(tmp, path);
}

json read_json(const fs::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot read json: " + path.string());
    }
    json payload;
    in >> payload;
    return payload;
}

fs::path state_path(const fs::path& state_dir) {
    return state_dir / "adapter_state.json";
}

json read_state(const fs::path& state_dir) {
    fs::path path = state_path(state_dir);
    if (!fs::exists(path)) {
        return json::object();
    }
    return read_json(path);
}

void write_state(const fs::path& state_dir, const json& state) {
    write_json(state_path(state_dir), state);
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    for (char ch : line) {
        if (ch == ',') {
            fields.push_back(current);
            current.clear();
        } else if (ch != '\r') {
            current.push_back(ch);
        }
    }
    fields.push_back(current);
    return fields;
}

std::vector<int64_t> read_ids(const fs::path& path, int64_t limit = -1) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot read ids: " + path.string());
    }
    std::vector<int64_t> ids;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line == "id") {
            continue;
        }
        auto fields = split_csv_line(line);
        if (fields.empty() || fields[0] == "id" || fields[0].empty()) {
            continue;
        }
        ids.push_back(std::stoll(fields[0]));
        if (limit >= 0 && static_cast<int64_t>(ids.size()) >= limit) {
            break;
        }
    }
    return ids;
}

std::pair<int64_t, int64_t> read_fbin_header(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot read vector file: " + path.string());
    }
    int32_t n = 0;
    int32_t d = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
    if (!in || n <= 0 || d <= 0) {
        throw std::runtime_error("invalid fbin header: " + path.string());
    }
    uint64_t expected = 8ull + static_cast<uint64_t>(n) * static_cast<uint64_t>(d) * sizeof(float);
    if (fs::file_size(path) != expected) {
        throw std::runtime_error("fbin size mismatch: " + path.string());
    }
    return {n, d};
}

std::vector<int64_t> parse_npy_shape(const std::string& header) {
    auto pos = header.find("'shape'");
    if (pos == std::string::npos) {
        pos = header.find("\"shape\"");
    }
    if (pos == std::string::npos) {
        throw std::runtime_error("npy header missing shape");
    }
    auto open = header.find('(', pos);
    auto close = header.find(')', open);
    if (open == std::string::npos || close == std::string::npos) {
        throw std::runtime_error("cannot parse npy shape");
    }
    std::string inside = header.substr(open + 1, close - open - 1);
    std::vector<int64_t> shape;
    std::string token;
    for (char ch : inside) {
        if (ch == ',') {
            if (!token.empty()) {
                shape.push_back(std::stoll(token));
                token.clear();
            }
        } else if (!std::isspace(static_cast<unsigned char>(ch))) {
            token.push_back(ch);
        }
    }
    if (!token.empty()) {
        shape.push_back(std::stoll(token));
    }
    return shape;
}

std::string parse_npy_descr(const std::string& header) {
    auto pos = header.find("'descr'");
    if (pos == std::string::npos) {
        pos = header.find("\"descr\"");
    }
    if (pos == std::string::npos) {
        throw std::runtime_error("npy header missing descr");
    }
    auto colon = header.find(':', pos);
    auto quote = header.find_first_of("'\"", colon);
    auto end = header.find(header[quote], quote + 1);
    return header.substr(quote + 1, end - quote - 1);
}

Matrix load_npy_matrix(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot read npy: " + path.string());
    }
    char magic[6];
    in.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("not an npy file: " + path.string());
    }
    uint8_t major = 0;
    uint8_t minor = 0;
    in.read(reinterpret_cast<char*>(&major), 1);
    in.read(reinterpret_cast<char*>(&minor), 1);
    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t h = 0;
        in.read(reinterpret_cast<char*>(&h), 2);
        header_len = h;
    } else if (major == 2 || major == 3) {
        in.read(reinterpret_cast<char*>(&header_len), 4);
    } else {
        throw std::runtime_error("unsupported npy version: " + path.string());
    }
    std::string header(header_len, '\0');
    in.read(header.data(), header_len);
    if (parse_npy_descr(header) != "<f4" && parse_npy_descr(header) != "|f4") {
        throw std::runtime_error("only float32 npy vectors are supported: " + path.string());
    }
    if (header.find("True") != std::string::npos) {
        throw std::runtime_error("fortran-order npy is not supported: " + path.string());
    }
    auto shape = parse_npy_shape(header);
    if (shape.size() != 2) {
        throw std::runtime_error("vector npy must be 2D: " + path.string());
    }
    Matrix m;
    m.rows = shape[0];
    m.dim = shape[1];
    m.data.resize(static_cast<size_t>(m.rows * m.dim));
    in.read(reinterpret_cast<char*>(m.data.data()), static_cast<std::streamsize>(m.data.size() * sizeof(float)));
    if (!in) {
        throw std::runtime_error("short npy vector read: " + path.string());
    }
    return m;
}

bool has_npy_magic(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    char magic[6];
    in.read(magic, 6);
    return in && std::memcmp(magic, "\x93NUMPY", 6) == 0;
}

Matrix load_vectors(const fs::path& path) {
    if (path.extension() == ".npy" || has_npy_magic(path)) {
        return load_npy_matrix(path);
    }
    auto [n, d] = read_fbin_header(path);
    Matrix m;
    m.rows = n;
    m.dim = d;
    m.data.resize(static_cast<size_t>(n * d));
    std::ifstream in(path, std::ios::binary);
    in.seekg(8);
    in.read(reinterpret_cast<char*>(m.data.data()), static_cast<std::streamsize>(m.data.size() * sizeof(float)));
    if (!in) {
        throw std::runtime_error("short fbin vector read: " + path.string());
    }
    return m;
}

void write_fbin(const fs::path& path, const Matrix& matrix) {
    ensure_parent(path);
    fs::path tmp = path;
    tmp += ".tmp";
    std::ofstream out(tmp, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot write fbin: " + tmp.string());
    }
    int32_t n = static_cast<int32_t>(matrix.rows);
    int32_t d = static_cast<int32_t>(matrix.dim);
    out.write(reinterpret_cast<const char*>(&n), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&d), sizeof(int32_t));
    if (!matrix.data.empty()) {
        out.write(reinterpret_cast<const char*>(matrix.data.data()), static_cast<std::streamsize>(matrix.data.size() * sizeof(float)));
    }
    out.close();
    fs::rename(tmp, path);
}

std::vector<Segment> parse_segments(const json& state) {
    std::vector<Segment> segments;
    for (const auto& item : state.value("segments", json::array())) {
        Segment seg;
        seg.segment_id = item.value("segment_id", 0);
        seg.vectors_path = item.value("vectors_path", "");
        seg.ids_path = item.value("ids_path", "");
        seg.count = item.value("count", 0);
        seg.reference_only = item.value("reference_only", false);
        segments.push_back(seg);
    }
    return segments;
}

json segment_to_json(const Segment& seg) {
    return {
        {"segment_id", seg.segment_id},
        {"vectors_path", fs::absolute(seg.vectors_path).string()},
        {"ids_path", fs::absolute(seg.ids_path).string()},
        {"count", seg.count},
        {"reference_only", seg.reference_only},
    };
}

std::vector<size_t> sorted_order(const std::vector<int64_t>& ids) {
    std::vector<size_t> order(ids.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return ids[a] < ids[b];
    });
    return order;
}

LabelState reorder_labels(const LabelState& input, const std::vector<size_t>& order) {
    LabelState out;
    out.ids.resize(order.size());
    out.live.resize(order.size());
    out.range_uniform.resize(order.size());
    for (const auto& field : kSelectivityFields) {
        out.fields[field].resize(order.size());
    }
    for (size_t i = 0; i < order.size(); ++i) {
        size_t src = order[i];
        out.ids[i] = input.ids[src];
        out.live[i] = input.live[src];
        out.range_uniform[i] = input.range_uniform[src];
        for (const auto& field : kSelectivityFields) {
            out.fields[field][i] = input.fields.at(field)[src];
        }
    }
    return out;
}

LabelState labels_from_csv(const fs::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot read labels csv: " + path.string());
    }
    std::string header_line;
    if (!std::getline(in, header_line)) {
        throw std::runtime_error("empty labels csv: " + path.string());
    }
    auto header = split_csv_line(header_line);
    std::unordered_map<std::string, size_t> col;
    for (size_t i = 0; i < header.size(); ++i) {
        col[header[i]] = i;
    }
    if (!col.count("id")) {
        throw std::runtime_error("labels csv missing id column: " + path.string());
    }
    LabelState labels;
    for (const auto& field : kSelectivityFields) {
        labels.fields[field] = {};
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        auto values = split_csv_line(line);
        auto get = [&](const std::string& name) -> std::string {
            auto it = col.find(name);
            if (it == col.end() || it->second >= values.size()) {
                return "0";
            }
            return values[it->second];
        };
        labels.ids.push_back(std::stoll(get("id")));
        labels.live.push_back(1);
        labels.range_uniform.push_back(std::stof(get(kRangeField)));
        for (const auto& field : kSelectivityFields) {
            labels.fields[field].push_back(static_cast<uint8_t>(std::stoi(get(field)) != 0));
        }
    }
    return reorder_labels(labels, sorted_order(labels.ids));
}

int64_t lower_bound_id(const std::vector<int64_t>& ids, int64_t value) {
    auto it = std::lower_bound(ids.begin(), ids.end(), value);
    if (it == ids.end() || *it != value) {
        return -1;
    }
    return static_cast<int64_t>(it - ids.begin());
}

LabelState labels_for_ids(const LabelState& labels, const std::vector<int64_t>& wanted) {
    LabelState out;
    for (const auto& field : kSelectivityFields) {
        out.fields[field] = {};
    }
    for (int64_t id : wanted) {
        int64_t pos = lower_bound_id(labels.ids, id);
        if (pos < 0) {
            throw std::runtime_error("labels missing id: " + std::to_string(id));
        }
        out.ids.push_back(id);
        out.live.push_back(labels.live[pos]);
        out.range_uniform.push_back(labels.range_uniform[pos]);
        for (const auto& field : kSelectivityFields) {
            out.fields[field].push_back(labels.fields.at(field)[pos]);
        }
    }
    return reorder_labels(out, sorted_order(out.ids));
}

void store_label_state(const fs::path& state_dir, const LabelState& labels) {
    write_binary_vector(state_dir / "label_ids.i64", labels.ids);
    write_binary_vector(state_dir / "label_live.u8", labels.live);
    write_binary_vector(state_dir / "label_range_uniform.f32", labels.range_uniform);
    for (const auto& field : kSelectivityFields) {
        write_binary_vector(state_dir / ("label_" + field + ".u8"), labels.fields.at(field));
    }
}

LabelState load_label_state(const fs::path& state_dir) {
    LabelState labels;
    labels.ids = read_binary_vector<int64_t>(state_dir / "label_ids.i64");
    labels.live = read_binary_vector<uint8_t>(state_dir / "label_live.u8");
    labels.range_uniform = read_binary_vector<float>(state_dir / "label_range_uniform.f32");
    for (const auto& field : kSelectivityFields) {
        labels.fields[field] = read_binary_vector<uint8_t>(state_dir / ("label_" + field + ".u8"));
    }
    if (labels.live.size() != labels.ids.size() || labels.range_uniform.size() != labels.ids.size()) {
        throw std::runtime_error("label state size mismatch");
    }
    return labels;
}

LabelState merge_label_state(const LabelState& existing, const LabelState& incoming) {
    LabelState merged;
    for (const auto& field : kSelectivityFields) {
        merged.fields[field] = {};
    }
    size_t i = 0;
    size_t j = 0;
    while (i < existing.ids.size() || j < incoming.ids.size()) {
        bool take_existing = j == incoming.ids.size() || (i < existing.ids.size() && existing.ids[i] < incoming.ids[j]);
        if (i < existing.ids.size() && j < incoming.ids.size() && existing.ids[i] == incoming.ids[j]) {
            throw std::runtime_error("insert id already exists: " + std::to_string(existing.ids[i]));
        }
        const LabelState& src = take_existing ? existing : incoming;
        size_t pos = take_existing ? i++ : j++;
        merged.ids.push_back(src.ids[pos]);
        merged.live.push_back(src.live[pos]);
        merged.range_uniform.push_back(src.range_uniform[pos]);
        for (const auto& field : kSelectivityFields) {
            merged.fields[field].push_back(src.fields.at(field)[pos]);
        }
    }
    return merged;
}

int64_t live_count(const LabelState& labels) {
    return static_cast<int64_t>(std::count(labels.live.begin(), labels.live.end(), static_cast<uint8_t>(1)));
}

bool matches_selector_at(const LabelState& labels, const json& selector, size_t pos);

std::vector<int64_t> matching_positions(const LabelState& labels, const json& selector) {
    std::vector<int64_t> positions;
    positions.reserve(labels.ids.size());
    for (size_t i = 0; i < labels.ids.size(); ++i) {
        if (labels.live[i] && matches_selector_at(labels, selector, i)) {
            positions.push_back(static_cast<int64_t>(i));
        }
    }
    return positions;
}

bool matches_selector_at(const LabelState& labels, const json& selector, size_t pos) {
    std::string type = selector.value("selector_type", "");
    if (type == "match_all") {
        return true;
    }
    if (type == "equality") {
        std::string field = selector.value("field", "");
        auto it = labels.fields.find(field);
        if (it == labels.fields.end()) {
            throw std::runtime_error("unsupported equality field: " + field);
        }
        return it->second[pos] == static_cast<uint8_t>(selector.value("value", 0));
    }
    if (type == "range") {
        std::string field = selector.value("field", "");
        if (field != kRangeField) {
            throw std::runtime_error("unsupported range field: " + field);
        }
        double lower = selector.value("lower", -std::numeric_limits<double>::infinity());
        double upper = selector.value("upper", std::numeric_limits<double>::infinity());
        double value = labels.range_uniform[pos];
        return value >= lower && value <= upper;
    }
    if (type == "intersect") {
        for (const auto& cond : selector.value("conditions", json::array())) {
            if (!matches_selector_at(labels, cond, pos)) {
                return false;
            }
        }
        return true;
    }
    throw std::runtime_error("unsupported selector type: " + type);
}

void clear_state_payload(const fs::path& state_dir) {
    fs::create_directories(state_dir);
    for (const auto& entry : fs::directory_iterator(state_dir)) {
        if (entry.path().filename() == "adapter.lock") {
            continue;
        }
        fs::remove_all(entry.path());
    }
}

Segment register_segment_reference(const fs::path& state_dir, int64_t segment_id, const fs::path& vectors_path, const std::vector<int64_t>& ids) {
    fs::path segment_dir = state_dir / "segments";
    fs::create_directories(segment_dir);
    fs::path ids_path = segment_dir / ("segment_" + std::to_string(segment_id) + ".ids.i64");
    write_binary_vector(ids_path, ids);
    return Segment{segment_id, fs::absolute(vectors_path), fs::absolute(ids_path), static_cast<int64_t>(ids.size()), true};
}

std::string padded_segment_name(int64_t segment_id, const std::string& suffix) {
    std::ostringstream oss;
    oss << "segment_" << std::setw(6) << std::setfill('0') << segment_id << suffix;
    return oss.str();
}

Segment write_segment_in_dir(const fs::path& segment_dir, int64_t segment_id, const Matrix& vectors, const std::vector<int64_t>& ids) {
    fs::create_directories(segment_dir);
    fs::path vectors_path = segment_dir / padded_segment_name(segment_id, ".vectors.fbin");
    fs::path ids_path = segment_dir / padded_segment_name(segment_id, ".ids.i64");
    write_fbin(vectors_path, vectors);
    write_binary_vector(ids_path, ids);
    return Segment{segment_id, fs::absolute(vectors_path), fs::absolute(ids_path), static_cast<int64_t>(ids.size()), false};
}

void set_search_ids(const fs::path& state_dir, json& state, const std::vector<int64_t>& row_ids) {
    write_binary_vector(state_dir / "search_ids.i64", row_ids);
    std::vector<size_t> order = sorted_order(row_ids);
    std::vector<int64_t> sorted_ids(order.size());
    std::vector<int64_t> rows(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
        sorted_ids[i] = row_ids[order[i]];
        rows[i] = static_cast<int64_t>(order[i]);
    }
    write_binary_vector(state_dir / "search_index_ids.i64", sorted_ids);
    write_binary_vector(state_dir / "search_index_rows.i64", rows);
    state["search_ids_path"] = fs::absolute(state_dir / "search_ids.i64").string();
}

void rebuild_row_index(const fs::path& state_dir, const json& state) {
    std::vector<int64_t> ids;
    std::vector<int32_t> segments_out;
    std::vector<int64_t> offsets;
    std::vector<int64_t> rows;
    int64_t row_base = 0;
    for (const auto& seg : parse_segments(state)) {
        std::vector<int64_t> seg_ids = read_binary_vector<int64_t>(seg.ids_path);
        for (int64_t i = 0; i < static_cast<int64_t>(seg_ids.size()); ++i) {
            ids.push_back(seg_ids[i]);
            segments_out.push_back(static_cast<int32_t>(seg.segment_id));
            offsets.push_back(i);
            rows.push_back(row_base + i);
        }
        row_base += static_cast<int64_t>(seg_ids.size());
    }
    std::vector<size_t> order = sorted_order(ids);
    auto reorder_i64 = [&](const std::vector<int64_t>& input) {
        std::vector<int64_t> out(order.size());
        for (size_t i = 0; i < order.size(); ++i) out[i] = input[order[i]];
        return out;
    };
    std::vector<int32_t> seg_sorted(order.size());
    for (size_t i = 0; i < order.size(); ++i) seg_sorted[i] = segments_out[order[i]];
    write_binary_vector(state_dir / "vector_index_ids.i64", reorder_i64(ids));
    write_binary_vector(state_dir / "vector_index_segments.i32", seg_sorted);
    write_binary_vector(state_dir / "vector_index_offsets.i64", reorder_i64(offsets));
    write_binary_vector(state_dir / "vector_index_rows.i64", reorder_i64(rows));
}

std::vector<int64_t> ids_to_search_rows_in_snapshot(const fs::path& snapshot_dir, const std::vector<int64_t>& ids) {
    std::vector<int64_t> index_ids = read_binary_vector<int64_t>(snapshot_dir / "search_index_ids.i64");
    std::vector<int64_t> index_rows = read_binary_vector<int64_t>(snapshot_dir / "search_index_rows.i64");
    std::vector<int64_t> rows;
    rows.reserve(ids.size());
    for (int64_t id : ids) {
        int64_t pos = lower_bound_id(index_ids, id);
        if (pos < 0) {
            throw std::runtime_error("candidate id missing snapshot search row: " + std::to_string(id));
        }
        rows.push_back(index_rows[pos]);
    }
    return rows;
}

Matrix load_candidate_vectors(const fs::path& state_dir, const json& state, const std::vector<int64_t>& candidate_ids) {
    Matrix out;
    out.dim = state.value("dimension", 0);
    out.rows = static_cast<int64_t>(candidate_ids.size());
    out.data.assign(static_cast<size_t>(out.rows * out.dim), 0.0f);
    if (candidate_ids.empty()) {
        return out;
    }
    std::vector<int64_t> index_ids = read_binary_vector<int64_t>(state_dir / "vector_index_ids.i64");
    std::vector<int32_t> index_segments = read_binary_vector<int32_t>(state_dir / "vector_index_segments.i32");
    std::vector<int64_t> index_offsets = read_binary_vector<int64_t>(state_dir / "vector_index_offsets.i64");
    std::unordered_map<int64_t, Segment> segment_by_id;
    for (const auto& seg : parse_segments(state)) {
        segment_by_id[seg.segment_id] = seg;
    }
    std::unordered_map<int64_t, std::vector<std::pair<size_t, int64_t>>> by_segment;
    for (size_t out_i = 0; out_i < candidate_ids.size(); ++out_i) {
        int64_t pos = lower_bound_id(index_ids, candidate_ids[out_i]);
        if (pos < 0) {
            throw std::runtime_error("candidate id missing vector row: " + std::to_string(candidate_ids[out_i]));
        }
        by_segment[index_segments[pos]].push_back({out_i, index_offsets[pos]});
    }
    for (const auto& [segment_id, items] : by_segment) {
        auto it = segment_by_id.find(segment_id);
        if (it == segment_by_id.end()) {
            throw std::runtime_error("missing segment metadata: " + std::to_string(segment_id));
        }
        Matrix segment_matrix = load_vectors(it->second.vectors_path);
        for (const auto& [out_i, offset] : items) {
            const float* src = segment_matrix.row(offset);
            float* dst = out.data.data() + static_cast<int64_t>(out_i) * out.dim;
            std::copy(src, src + out.dim, dst);
        }
    }
    return out;
}

void refresh_ann_index_in_dir(const fs::path& cache_parent, json& state, const Matrix& vectors, int threads, bool materialized) {
    fs::path cache_dir = cache_parent / kRuntimeCacheDir;
    fs::create_directories(cache_dir);
    fs::path ann_path = cache_dir / kAnnIndexFile;
    state.erase("ann_index_path");
    if (fs::exists(ann_path)) {
        fs::remove(ann_path);
    }
    if (vectors.rows < kAnnMinPoints) {
        state["backend"] = kBackendExact;
        state["ann_materialized_vectors"] = materialized;
        return;
    }

    // FAISS C++ 负责 IVF 训练、add、持久化；业务层只准备连续 live matrix。
    omp_set_num_threads(std::max(1, threads));
    int64_t nlist = std::min<int64_t>(kAnnNlist, std::max<int64_t>(1, vectors.rows / 32));
    int64_t train_count = std::min<int64_t>(vectors.rows, std::max<int64_t>(nlist, kAnnTrainPoints));
    faiss::IndexFlatL2 quantizer(vectors.dim);
    faiss::IndexIVFFlat index(&quantizer, vectors.dim, nlist, faiss::METRIC_L2);
    index.train(train_count, vectors.data.data());
    index.add(vectors.rows, vectors.data.data());
    index.nprobe = std::min<int64_t>(kAnnNprobe, nlist);
    faiss::write_index(&index, ann_path.c_str());
    state["backend"] = kBackendHybrid;
    state["ann_index_path"] = fs::absolute(ann_path).string();
    state["ann_nlist"] = nlist;
    state["ann_nprobe"] = index.nprobe;
    state["ann_train_count"] = train_count;
    state["ann_row_count"] = vectors.rows;
    state["ann_materialized_vectors"] = materialized;
}

Matrix load_search_matrix(const json& state) {
    return load_vectors(state.at("search_vectors_path").get<std::string>());
}

bool should_use_ann(const json& state, int64_t candidate_count) {
    if (candidate_count <= kExactCandidateLimit) {
        return false;
    }
    if (!state.contains("ann_index_path")) {
        return false;
    }
    return fs::exists(state.at("ann_index_path").get<std::string>());
}

std::vector<int64_t> faiss_topk_by_rows(const Matrix& matrix, const float* query, const std::vector<int64_t>& rows, int64_t k) {
    if (rows.empty() || k <= 0) {
        return {};
    }
    int64_t out_k = std::min<int64_t>(k, rows.size());
    std::vector<float> distances(static_cast<size_t>(out_k));
    std::vector<int64_t> labels(static_cast<size_t>(out_k));
    faiss::maxheap_heapify(out_k, distances.data(), labels.data());
    for (int64_t row : rows) {
        float distance = faiss::fvec_L2sqr(query, matrix.row(row), matrix.dim);
        if (distance < distances[0]) {
            faiss::maxheap_replace_top(out_k, distances.data(), labels.data(), distance, row);
        }
    }
    faiss::maxheap_reorder(out_k, distances.data(), labels.data());

    std::vector<int64_t> out;
    out.reserve(out_k);
    for (int64_t row : labels) {
        if (row >= 0) {
            out.push_back(row);
        }
    }
    return out;
}

json result_trace(const std::string& backend, int64_t candidate_count) {
    return {{"candidate_count", candidate_count}, {"backend", backend}};
}

json search_row_json(const SearchRow& row) {
    return {
        {"query_id", row.query_id},
        {"ids", row.ids},
        {"latency_ms", row.latency_ms},
        {"trace", result_trace(row.backend, row.candidate_count)},
    };
}

std::vector<SearchRow> run_exact_subset_search(
        const Matrix& matrix,
        const std::vector<int64_t>& search_ids,
        const Matrix& queries,
        const std::vector<int64_t>& candidate_rows,
        int64_t k,
        const std::string& backend) {
    std::vector<SearchRow> rows;
    rows.reserve(queries.rows);
    for (int64_t qi = 0; qi < queries.rows; ++qi) {
        auto start = std::chrono::steady_clock::now();
        std::vector<int64_t> top_rows = faiss_topk_by_rows(matrix, queries.row(qi), candidate_rows, k);
        std::vector<int64_t> top_ids;
        top_ids.reserve(top_rows.size());
        for (int64_t row : top_rows) {
            if (row >= 0 && row < static_cast<int64_t>(search_ids.size())) {
                top_ids.push_back(search_ids[row]);
            }
        }
        auto end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        rows.push_back(SearchRow{static_cast<int>(qi), top_ids, elapsed, backend, static_cast<int64_t>(candidate_rows.size())});
    }
    return rows;
}

std::vector<int64_t> ann_filtered_rows(
        faiss::Index& index,
        const float* query,
        const std::unordered_set<int64_t>& candidate_lookup,
        int64_t ann_k) {
    std::vector<float> distances(static_cast<size_t>(ann_k));
    std::vector<faiss::Index::idx_t> labels(static_cast<size_t>(ann_k));
    index.search(1, query, ann_k, distances.data(), labels.data());
    std::vector<int64_t> filtered;
    filtered.reserve(labels.size());
    for (faiss::Index::idx_t row : labels) {
        if (row >= 0 && candidate_lookup.find(row) != candidate_lookup.end()) {
            filtered.push_back(static_cast<int64_t>(row));
        }
    }
    return filtered;
}

std::vector<SearchRow> run_ann_rerank_search(
        const fs::path& ann_path,
        const Matrix& matrix,
        const std::vector<int64_t>& search_ids,
        const Matrix& queries,
        std::vector<int64_t> candidate_rows,
        int64_t k) {
    std::unique_ptr<faiss::Index> index(faiss::read_index(ann_path.c_str()));
    auto* ivf = dynamic_cast<faiss::IndexIVF*>(index.get());
    bool single_query = queries.rows == 1;
    int64_t nprobe = single_query ? kSingleQueryNprobe : kAnnNprobe;
    if (!single_query && static_cast<int64_t>(candidate_rows.size()) >= matrix.rows) {
        nprobe = kAnnFullSelectivityNprobe;
    }
    int64_t rerank_k = single_query ? kSingleQueryRerankK : kAnnRerankK;
    int64_t retry_rerank_k = single_query ? kSingleQueryRetryRerankK : kAnnRetryRerankK;
    if (ivf != nullptr) {
        ivf->nprobe = std::min<int64_t>(nprobe, ivf->nlist);
    }
    std::sort(candidate_rows.begin(), candidate_rows.end());
    candidate_rows.erase(std::unique(candidate_rows.begin(), candidate_rows.end()), candidate_rows.end());
    std::unordered_set<int64_t> candidate_lookup(candidate_rows.begin(), candidate_rows.end());
    int64_t ann_k = std::min<int64_t>(matrix.rows, std::max<int64_t>(rerank_k, k));
    if (single_query && queries.rows > 0) {
        (void) ann_filtered_rows(*index, queries.row(0), candidate_lookup, ann_k);
    }
    std::vector<SearchRow> out;
    out.reserve(queries.rows);
    for (int64_t qi = 0; qi < queries.rows; ++qi) {
        auto start = std::chrono::steady_clock::now();
        const float* query = queries.row(qi);
        std::vector<int64_t> filtered = ann_filtered_rows(*index, query, candidate_lookup, ann_k);
        int64_t required = std::min<int64_t>(k, candidate_rows.size());
        if (single_query && static_cast<int64_t>(filtered.size()) < required) {
            std::unordered_set<int64_t> seen(filtered.begin(), filtered.end());
            for (int64_t row : candidate_rows) {
                if (seen.insert(row).second) {
                    filtered.push_back(row);
                    if (static_cast<int64_t>(filtered.size()) >= required) break;
                }
            }
        } else if (static_cast<int64_t>(filtered.size()) < required) {
            int64_t retry_k = std::min<int64_t>(matrix.rows, std::max<int64_t>(ann_k * 4, retry_rerank_k));
            if (retry_k > ann_k) {
                filtered = ann_filtered_rows(*index, query, candidate_lookup, retry_k);
            }
        }
        if (!single_query && static_cast<int64_t>(filtered.size()) < required) {
            filtered = candidate_rows;
        }
        std::vector<int64_t> top_rows = faiss_topk_by_rows(matrix, query, filtered, k);
        std::vector<int64_t> top_ids;
        top_ids.reserve(top_rows.size());
        for (int64_t row : top_rows) {
            top_ids.push_back(search_ids[row]);
        }
        auto end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        out.push_back(SearchRow{static_cast<int>(qi), top_ids, elapsed, kBackendHybrid, static_cast<int64_t>(candidate_rows.size())});
    }
    return out;
}

json latency_summary(const std::vector<SearchRow>& rows) {
    if (rows.empty()) {
        return {
            {"avg_latency_ms", std::numeric_limits<double>::infinity()},
            {"p50_latency_ms", std::numeric_limits<double>::infinity()},
            {"p95_latency_ms", std::numeric_limits<double>::infinity()},
            {"p99_latency_ms", std::numeric_limits<double>::infinity()},
        };
    }
    std::vector<double> values;
    values.reserve(rows.size());
    for (const auto& row : rows) values.push_back(row.latency_ms);
    std::sort(values.begin(), values.end());
    auto pct = [&](double p) {
        if (values.size() == 1) return values[0];
        double rank = (values.size() - 1) * p / 100.0;
        size_t lo = static_cast<size_t>(rank);
        size_t hi = std::min(lo + 1, values.size() - 1);
        double w = rank - lo;
        return values[lo] * (1.0 - w) + values[hi] * w;
    };
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return {
        {"avg_latency_ms", sum / values.size()},
        {"p50_latency_ms", pct(50.0)},
        {"p95_latency_ms", pct(95.0)},
        {"p99_latency_ms", pct(99.0)},
    };
}

std::vector<std::string> state_index_output_paths(const fs::path& state_dir, const fs::path& index_manifest) {
    std::vector<fs::path> paths = {fs::absolute(index_manifest)};
    std::vector<std::string> names = {
        "adapter_state.json", "search_ids.i64", "search_index_ids.i64", "search_index_rows.i64",
        "vector_index_ids.i64", "vector_index_segments.i32", "vector_index_offsets.i64", "vector_index_rows.i64",
        "label_ids.i64", "label_live.u8", "label_range_uniform.f32"};
    for (const auto& field : kSelectivityFields) {
        names.push_back("label_" + field + ".u8");
    }
    for (const auto& name : names) {
        fs::path path = state_dir / name;
        if (fs::exists(path)) {
            paths.push_back(fs::absolute(path));
        }
    }
    fs::path segments = state_dir / "segments";
    if (fs::exists(segments)) {
        for (const auto& entry : fs::directory_iterator(segments)) {
            if (entry.path().extension() == ".i64") {
                paths.push_back(fs::absolute(entry.path()));
            }
        }
    }
    fs::path runtime_cache = state_dir / kRuntimeCacheDir;
    if (fs::exists(runtime_cache)) {
        for (const auto& entry : fs::directory_iterator(runtime_cache)) {
            if (entry.path().extension() == ".index") {
                paths.push_back(fs::absolute(entry.path()));
            }
        }
    }
    fs::path snapshots = state_dir / "snapshots";
    if (fs::exists(snapshots)) {
        for (const auto& entry : fs::recursive_directory_iterator(snapshots)) {
            if (entry.is_regular_file()) {
                paths.push_back(fs::absolute(entry.path()));
            }
        }
    }
    std::sort(paths.begin(), paths.end());
    paths.erase(std::unique(paths.begin(), paths.end()), paths.end());
    std::vector<std::string> out;
    for (const auto& path : paths) out.push_back(path.string());
    return out;
}

std::vector<std::string> state_payload_names() {
    std::vector<std::string> names = {
        "search_ids.i64", "search_index_ids.i64", "search_index_rows.i64",
        "vector_index_ids.i64", "vector_index_segments.i32", "vector_index_offsets.i64", "vector_index_rows.i64",
        "label_ids.i64", "label_live.u8", "label_range_uniform.f32", "search_live_vectors.fbin"};
    for (const auto& field : kSelectivityFields) {
        names.push_back("label_" + field + ".u8");
    }
    return names;
}

void move_if_exists(const fs::path& src, const fs::path& dst) {
    if (!fs::exists(src)) {
        return;
    }
    ensure_parent(dst);
    if (fs::exists(dst)) {
        fs::remove(dst);
    }
    fs::rename(src, dst);
}

void commit_prepared_stage(const fs::path& state_dir, const fs::path& stage_dir, const fs::path& final_snapshot_dir, json& state) {
    fs::path staged_segments = stage_dir / "segments";
    if (fs::exists(staged_segments)) {
        fs::path final_segments = state_dir / "segments";
        fs::create_directories(final_segments);
        for (const auto& entry : fs::directory_iterator(staged_segments)) {
            fs::path dst = final_segments / entry.path().filename();
            if (fs::exists(dst)) {
                fs::remove(dst);
            }
            fs::rename(entry.path(), dst);
        }
    }
    for (const auto& name : state_payload_names()) {
        move_if_exists(stage_dir / name, state_dir / name);
    }
    fs::path staged_snapshot_dir = stage_dir / "snapshot";
    if (fs::exists(staged_snapshot_dir)) {
        fs::create_directories(final_snapshot_dir.parent_path());
        fs::path old_snapshot_dir = final_snapshot_dir;
        old_snapshot_dir += ".old";
        fs::remove_all(old_snapshot_dir);
        if (fs::exists(final_snapshot_dir)) {
            fs::rename(final_snapshot_dir, old_snapshot_dir);
        }
        fs::rename(staged_snapshot_dir, final_snapshot_dir);
        state["snapshot_dir"] = fs::absolute(final_snapshot_dir).string();
        state["search_vectors_path"] = fs::absolute(final_snapshot_dir / "search_live_vectors.fbin").string();
        state["search_ids_path"] = fs::absolute(final_snapshot_dir / "search_ids.i64").string();
        fs::path ann_path = final_snapshot_dir / kRuntimeCacheDir / kAnnIndexFile;
        if (fs::exists(ann_path)) {
            state["ann_index_path"] = fs::absolute(ann_path).string();
        } else {
            state.erase("ann_index_path");
        }
        fs::remove_all(old_snapshot_dir);
    }
    state["needs_materialized_refresh"] = false;
    fs::remove_all(stage_dir);
}

void prepare_snapshot(
        const fs::path& snapshot_dir,
        json& state,
        const Matrix& vectors,
        const std::vector<int64_t>& row_ids,
        int threads) {
    fs::remove_all(snapshot_dir);
    fs::create_directories(snapshot_dir);
    fs::path vectors_path = snapshot_dir / "search_live_vectors.fbin";
    write_fbin(vectors_path, vectors);
    state["search_vectors_path"] = fs::absolute(vectors_path).string();
    set_search_ids(snapshot_dir, state, row_ids);
    refresh_ann_index_in_dir(snapshot_dir, state, vectors, threads, true);
    state["snapshot_dir"] = fs::absolute(snapshot_dir).string();
    state["needs_materialized_refresh"] = false;
}

void prepare_reference_snapshot(
        const fs::path& snapshot_dir,
        json& state,
        const fs::path& vectors_path,
        const Matrix& vectors,
        const std::vector<int64_t>& row_ids,
        int threads) {
    fs::remove_all(snapshot_dir);
    fs::create_directories(snapshot_dir);
    state["search_vectors_path"] = fs::absolute(vectors_path).string();
    set_search_ids(snapshot_dir, state, row_ids);
    refresh_ann_index_in_dir(snapshot_dir, state, vectors, threads, false);
    state["snapshot_dir"] = fs::absolute(snapshot_dir).string();
    state["needs_materialized_refresh"] = false;
}

void command_build(const std::unordered_map<std::string, std::string>& args) {
    fs::path state_dir = require_arg(args, "state-dir");
    fs::path index_dir = require_arg(args, "index-dir");
    fs::create_directories(state_dir);
    fs::create_directories(index_dir);
    FileLock lock(state_dir, true);
    clear_state_payload(state_dir);
    fs::path vectors_path = require_arg(args, "vectors");
    Matrix vectors = load_vectors(vectors_path);
    std::vector<int64_t> ids = read_ids(require_arg(args, "ids"), vectors.rows);
    if (static_cast<int64_t>(ids.size()) != vectors.rows) {
        throw std::runtime_error("id/vector count mismatch");
    }
    LabelState labels = labels_for_ids(labels_from_csv(require_arg(args, "labels")), ids);
    Segment segment = register_segment_reference(state_dir, 0, vectors_path, ids);
    fs::path snapshot_dir = state_dir / "snapshots" / "epoch_0";
    json state = {
        {"version", 2},
        {"implementation", "cpp_faiss"},
        {"backend", kBackendExact},
        {"dimension", vectors.dim},
        {"segments", json::array({segment_to_json(segment)})},
        {"next_segment", 1},
        {"index_dir", fs::absolute(index_dir).string()},
        {"search_vectors_path", ""},
        {"live_count", vectors.rows},
        {"mutation_epoch", 0},
        {"created_at_unix", static_cast<double>(std::time(nullptr))},
    };
    store_label_state(state_dir, labels);
    rebuild_row_index(state_dir, state);
    prepare_reference_snapshot(snapshot_dir, state, vectors_path, vectors, ids, std::max(1, std::stoi(require_arg(args, "threads"))));
    write_state(state_dir, state);
    fs::path index_manifest = index_dir / "adapter_index_manifest.json";
    write_json(index_manifest, {
        {"backend", state["backend"]},
        {"state_dir", fs::absolute(state_dir).string()},
        {"segments", state["segments"]},
        {"dimension", state["dimension"]},
        {"ann_cache_path", state.value("ann_index_path", "")},
        {"implementation", "cpp_faiss"},
    });
    write_json(require_arg(args, "output-manifest"), {
        {"raw_data_paths", json::array({fs::absolute(vectors_path).string()})},
        {"index_output_paths", state_index_output_paths(state_dir, index_manifest)},
    });
}

void command_selectivity(const std::unordered_map<std::string, std::string>& args) {
    fs::path state_dir = require_arg(args, "state-dir");
    json selector = read_json(require_arg(args, "selector"));
    FileLock lock(state_dir, false);
    LabelState labels = load_label_state(state_dir);
    std::vector<int64_t> matched = matching_positions(labels, selector);
    int64_t total = live_count(labels);
    write_json(require_arg(args, "output"), {
        {"matched_count", matched.size()},
        {"total_live_count", total},
        {"selectivity", total == 0 ? 0.0 : static_cast<double>(matched.size()) / static_cast<double>(total)},
    });
}

void command_search(const std::unordered_map<std::string, std::string>& args) {
    fs::path state_dir = require_arg(args, "state-dir");
    json selector = read_json(require_arg(args, "selector"));
    int64_t limit = std::max<int64_t>(0, std::stoll(require_arg(args, "limit")));
    int64_t k = std::stoll(require_arg(args, "k"));
    int threads = std::max(1, std::stoi(require_arg(args, "threads")));
    Matrix queries = load_vectors(require_arg(args, "queries"));
    if (limit < queries.rows) {
        queries.rows = limit;
        queries.data.resize(static_cast<size_t>(queries.rows * queries.dim));
    }

    std::vector<int64_t> candidate_ids;
    std::vector<SearchRow> rows;
    std::string backend = kBackendExact;
    bool consistent_snapshot = false;
    for (int attempt = 0; attempt < 10; ++attempt) {
        candidate_ids.clear();
        json state;
        uint64_t epoch = 0;
        std::vector<int64_t> search_ids;
        std::vector<int64_t> candidate_rows;
        fs::path snapshot_dir;
        {
            FileLock lock(state_dir, false);
            LabelState labels = load_label_state(state_dir);
            std::vector<int64_t> positions = matching_positions(labels, selector);
            candidate_ids.reserve(positions.size());
            for (int64_t pos : positions) {
                candidate_ids.push_back(labels.ids[pos]);
            }
            state = read_state(state_dir);
            epoch = state.value("mutation_epoch", 0);
            snapshot_dir = state.at("snapshot_dir").get<std::string>();
            search_ids = read_binary_vector<int64_t>(snapshot_dir / "search_ids.i64");
            candidate_rows = ids_to_search_rows_in_snapshot(snapshot_dir, candidate_ids);
        }

        Matrix matrix = load_search_matrix(state);
        omp_set_num_threads(threads);
        rows.clear();
        backend = kBackendExact;
        if (candidate_rows.empty()) {
            for (int64_t qi = 0; qi < queries.rows; ++qi) {
                rows.push_back(SearchRow{static_cast<int>(qi), {}, 0.0, backend, 0});
            }
        } else if (should_use_ann(state, static_cast<int64_t>(candidate_rows.size()))) {
            backend = kBackendHybrid;
            rows = run_ann_rerank_search(state.at("ann_index_path").get<std::string>(), matrix, search_ids, queries, candidate_rows, k);
        } else {
            rows = run_exact_subset_search(matrix, search_ids, queries, candidate_rows, k, backend);
        }

        {
            FileLock lock(state_dir, false);
            uint64_t current_epoch = read_state(state_dir).value("mutation_epoch", 0);
            if (current_epoch == epoch) {
                consistent_snapshot = true;
                break;
            }
        }
    }
    if (!consistent_snapshot) {
        throw std::runtime_error("mutation epoch changed repeatedly during search");
    }
    json result_rows = json::array();
    for (const auto& row : rows) {
        result_rows.push_back(search_row_json(row));
    }
    json summary = latency_summary(rows);
    summary["backend"] = backend;
    summary["candidate_count"] = candidate_ids.size();
    summary["selector_type"] = selector.value("selector_type", "");
    write_json(require_arg(args, "output"), {{"results", result_rows}, {"summary", summary}});
}

void command_insert(const std::unordered_map<std::string, std::string>& args) {
    fs::path state_dir = require_arg(args, "state-dir");
    fs::create_directories(state_dir);
    Matrix vectors = load_vectors(require_arg(args, "vectors"));
    std::vector<int64_t> ids = read_ids(require_arg(args, "ids"), vectors.rows);
    if (static_cast<int64_t>(ids.size()) != vectors.rows) {
        throw std::runtime_error("id/vector count mismatch");
    }
    LabelState incoming = labels_for_ids(labels_from_csv(require_arg(args, "labels")), ids);
    json state;
    LabelState merged;
    uint64_t base_epoch = 0;
    {
        FileLock lock(state_dir, false);
        state = read_state(state_dir);
        base_epoch = state.value("mutation_epoch", 0);
        if (!fs::exists(state_dir / "label_ids.i64")) {
            merged = incoming;
        } else {
            if (vectors.dim != state.value("dimension", vectors.dim)) {
                throw std::runtime_error("insert dimension mismatch");
            }
            merged = merge_label_state(load_label_state(state_dir), incoming);
        }
    }
    int64_t segment_id = state.value("next_segment", static_cast<int64_t>(state.value("segments", json::array()).size()));
    fs::path stage_dir = state_dir / (".insert_stage_" + std::to_string(::getpid()));
    fs::remove_all(stage_dir);
    fs::create_directories(stage_dir);
    Segment staged_segment = write_segment_in_dir(stage_dir / "segments", segment_id, vectors, ids);
    Segment final_segment = staged_segment;
    final_segment.vectors_path = fs::absolute(state_dir / "segments" / staged_segment.vectors_path.filename());
    final_segment.ids_path = fs::absolute(state_dir / "segments" / staged_segment.ids_path.filename());
    json staged_state = state;
    json final_state = state;
    if (!staged_state.contains("segments")) {
        staged_state["segments"] = json::array();
    }
    if (!final_state.contains("segments")) {
        final_state["segments"] = json::array();
    }
    staged_state["segments"].push_back(segment_to_json(staged_segment));
    final_state["segments"].push_back(segment_to_json(final_segment));
    staged_state["next_segment"] = segment_id + 1;
    final_state["next_segment"] = segment_id + 1;
    if (!fs::exists(state_dir / "label_ids.i64")) {
        auto initialize_state = [&](json& target) {
            target["version"] = 2;
            target["implementation"] = "cpp_faiss";
            target["dimension"] = vectors.dim;
            target["backend"] = kBackendExact;
            target["index_dir"] = fs::absolute(state_dir / "index").string();
        };
        initialize_state(staged_state);
        initialize_state(final_state);
    }
    store_label_state(stage_dir, merged);
    rebuild_row_index(stage_dir, staged_state);
    std::vector<int64_t> live_ids;
    live_ids.reserve(merged.ids.size());
    for (size_t i = 0; i < merged.ids.size(); ++i) {
        if (merged.live[i]) {
            live_ids.push_back(merged.ids[i]);
        }
    }
    Matrix live_vectors = load_candidate_vectors(stage_dir, staged_state, live_ids);
    fs::path final_snapshot_dir = state_dir / "snapshots" / ("epoch_" + std::to_string(base_epoch + 1));
    prepare_snapshot(stage_dir / "snapshot", staged_state, live_vectors, live_ids, std::max(1, std::stoi(require_arg(args, "threads"))));
    final_state["backend"] = staged_state["backend"];
    final_state["ann_materialized_vectors"] = staged_state.value("ann_materialized_vectors", true);
    for (const auto& key : {"ann_nlist", "ann_nprobe", "ann_train_count", "ann_row_count"}) {
        if (staged_state.contains(key)) {
            final_state[key] = staged_state[key];
        } else {
            final_state.erase(key);
        }
    }
    final_state["live_count"] = live_count(merged);
    final_state["mutation_epoch"] = final_state.value("mutation_epoch", 0) + 1;

    FileLock commit_lock(state_dir, true);
    uint64_t current_epoch = read_state(state_dir).value("mutation_epoch", 0);
    if (current_epoch != base_epoch) {
        fs::remove_all(stage_dir);
        throw std::runtime_error("concurrent mutation detected during insert");
    }
    commit_prepared_stage(state_dir, stage_dir, final_snapshot_dir, final_state);
    write_state(state_dir, final_state);
    write_json(require_arg(args, "output"), {{"live_count", final_state["live_count"]}});
}

void command_delete(const std::unordered_map<std::string, std::string>& args) {
    fs::path state_dir = require_arg(args, "state-dir");
    FileLock lock(state_dir, true);
    json state = read_state(state_dir);
    LabelState labels = load_label_state(state_dir);
    std::vector<int64_t> deleted = read_ids(require_arg(args, "ids"));
    int64_t changed = 0;
    for (int64_t id : deleted) {
        int64_t pos = lower_bound_id(labels.ids, id);
        if (pos >= 0 && labels.live[pos]) {
            labels.live[pos] = 0;
            ++changed;
        }
    }
    store_label_state(state_dir, labels);
    int64_t count = state.value("live_count", live_count(labels));
    state["live_count"] = std::max<int64_t>(0, count - changed);
    state["mutation_epoch"] = state.value("mutation_epoch", 0) + 1;
    write_state(state_dir, state);
    write_json(require_arg(args, "output"), {{"live_count", state["live_count"]}});
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "usage: openharmony_anns_adapter <build|search|selectivity|insert|delete> [options]\n";
            return 2;
        }
        std::string command = argv[1];
        auto args = parse_options(argc, argv, 2);
        if (command == "build") {
            command_build(args);
        } else if (command == "search") {
            command_search(args);
        } else if (command == "selectivity") {
            command_selectivity(args);
        } else if (command == "insert") {
            command_insert(args);
        } else if (command == "delete") {
            command_delete(args);
        } else {
            throw std::runtime_error("unknown command: " + command);
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "openharmony_anns_adapter error: " << exc.what() << "\n";
        return 1;
    }
}
