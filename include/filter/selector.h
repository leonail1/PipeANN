#ifndef SELECTOR_H_
#define SELECTOR_H_

#include "attribute.h"
#include <stdexcept>

namespace pipeann {

  // Selector: Abstract base class for attribute filtering in filtered ANNS.
  // Defines how to filter vectors based on attribute constraints, supporting
  // speculative pre-filtering, speculative in-filtering, and post-filtering.
  // Composite selectors (AndSelector, OrSelector, NotSelector) combine multiple
  // Selectors via Boolean logic for complex multi-attribute queries.
  struct Selector {
    virtual ~Selector() = default;
    // copy() always returns a deep copy of the selector tree. Callers either
    // borrow an existing selector or take ownership of a copied selector.
    virtual Selector *copy() const {
      throw std::runtime_error("copy() is only implemented for native selectors");
    }

    // Estimate the fraction of dataset vectors satisfying this constraint.
    virtual double estimate_selectivity(const Attributes &query_attrs) = 0;
    // Estimate the precision of is_member_approx: TP / (TP + FP). 1.0 = strict filter.
    virtual double estimate_precision(const Attributes &query_attrs) = 0;
    // Estimate the SSD pages read during speculative pre-filtering.
    virtual uint32_t estimate_prefilter_reads(const Attributes &query_attrs) = 0;

    // Scan on-SSD attribute indexes to return a superset of valid vector IDs.
    // May speculatively skip high-selectivity branches (e.g., in AndSelector),
    // deferring exact verification to is_member() during re-ranking.
    virtual VectorIDList pre_filter(const Attributes &query_attrs, AlignedFileReader *reader) = 0;

    // Exact membership check using full attributes stored in the on-SSD record.
    // Used during re-ranking to verify candidates from speculative filtering.
    virtual bool is_member(uint32_t target_id, const Attributes &query_attrs, const Attributes &target_attrs) = 0;

    // Estimate the SSD pages read during in-filter preparation (prepare_in_filter).
    virtual uint32_t estimate_infilter_reads(const Attributes &query_attrs) = 0;

    // Pre-scan rare/cold attribute entries from SSD before graph traversal starts.
    // The prepared state is stored on this selector instance and consumed by
    // is_member_approx() during traversal.
    virtual void prepare_in_filter(const Attributes &query_attrs, AlignedFileReader *reader) = 0;

    // Fast in-memory approximate membership check during graph traversal.
    // No false negatives: returns false only if the vector is definitely invalid.
    virtual bool is_member_approx(uint32_t target_id, const Attributes &query_attrs) = 0;
  };

  struct LabelOrSelector : public Selector {
    uint32_t key_;
    uint32_t base_key_;
    AttrIndex *attr_index_;
    VectorIDList cold_list_;
    LabelOrSelector(uint32_t key, uint32_t base_key, AttrIndex *attr_index)
        : key_(key), base_key_(base_key), attr_index_(attr_index) {
    }

    // Leaf selectors only need a shallow copy because AttrIndex ownership is
    // managed outside the selector tree.
    Selector *copy() const override {
      return new LabelOrSelector(key_, base_key_, attr_index_);
    }

    virtual double estimate_selectivity(const Attributes &query_attrs) override {
      return (double) attr_index_->estimate_count(query_attrs.get(key_)) / attr_index_->n_vectors;
    }

    virtual double estimate_precision(const Attributes &query_attrs) override {
      return attr_index_->estimate_precision(query_attrs.get(key_));
    }

    virtual uint32_t estimate_prefilter_reads(const Attributes &query_attrs) override {
      return attr_index_->estimate_prefilter_reads(query_attrs.get(key_));
    }

    virtual VectorIDList pre_filter(const Attributes &query_attrs, AlignedFileReader *reader) override {
      return attr_index_->pre_filter(query_attrs.get(key_), reader);
    }

    virtual bool is_member(uint32_t target_id, const Attributes &query_attrs, const Attributes &target_attrs) override {
      if (!query_attrs.find(key_) || !target_attrs.find(base_key_)) {
        return false;
      }
      Attribute query_attr = query_attrs.get(key_);
      Attribute target_attr = target_attrs.get(base_key_);
      for (auto &label : query_attr) {
        if (std::find(target_attr.begin(), target_attr.end(), label) != target_attr.end()) {
          return true;
        }
      }
      return false;
    }

    virtual uint32_t estimate_infilter_reads(const Attributes &query_attrs) override {
      return attr_index_->estimate_infilter_reads(query_attrs.get(key_));
    }

    virtual void prepare_in_filter(const Attributes &query_attrs, AlignedFileReader *reader) override {
      cold_list_ = attr_index_->prepare_in_filter(query_attrs.get(key_), reader);
    }

    virtual bool is_member_approx(uint32_t target_id, const Attributes &query_attrs) override {
      return attr_index_->is_member_approx(target_id, query_attrs.get(key_), cold_list_);
    }
  };

  struct LabelAndSelector : public Selector {
    uint32_t key_;
    uint32_t base_key_;
    AttrIndex *attr_index_;
    VectorIDList cold_list_;
    LabelAndSelector(uint32_t key, uint32_t base_key, AttrIndex *attr_index)
        : key_(key), base_key_(base_key), attr_index_(attr_index) {
    }

    // Leaf selectors only need a shallow copy because AttrIndex ownership is
    // managed outside the selector tree.
    Selector *copy() const override {
      return new LabelAndSelector(key_, base_key_, attr_index_);
    }

    InvertedLabelAttrIndex *inv_store() {
      return static_cast<InvertedLabelAttrIndex *>(attr_index_);
    }

    virtual double estimate_selectivity(const Attributes &query_attrs) override {
      return (double) inv_store()->estimate_count_and(query_attrs.get(key_)) / attr_index_->n_vectors;
    }

    virtual double estimate_precision(const Attributes &query_attrs) override {
      return inv_store()->estimate_precision_and(query_attrs.get(key_));
    }

    virtual uint32_t estimate_prefilter_reads(const Attributes &query_attrs) override {
      return inv_store()->estimate_prefilter_reads_and(query_attrs.get(key_));
    }

    virtual VectorIDList pre_filter(const Attributes &query_attrs, AlignedFileReader *reader) override {
      return inv_store()->pre_filter_and(query_attrs.get(key_), reader);
    }

    virtual bool is_member(uint32_t target_id, const Attributes &query_attrs, const Attributes &target_attrs) override {
      if (!query_attrs.find(key_) || !target_attrs.find(base_key_)) {
        return false;
      }
      Attribute query_attr = query_attrs.get(key_);
      Attribute target_attr = target_attrs.get(base_key_);
      for (auto &label : query_attr) {
        if (std::find(target_attr.begin(), target_attr.end(), label) == target_attr.end()) {
          return false;
        }
      }
      return true;
    }

    virtual uint32_t estimate_infilter_reads(const Attributes &query_attrs) override {
      return inv_store()->estimate_infilter_reads_and(query_attrs.get(key_));
    }

    virtual void prepare_in_filter(const Attributes &query_attrs, AlignedFileReader *reader) override {
      cold_list_ = inv_store()->prepare_in_filter_and(query_attrs.get(key_), reader);
    }

    virtual bool is_member_approx(uint32_t target_id, const Attributes &query_attrs) override {
      return inv_store()->is_member_approx_and(target_id, query_attrs.get(key_), cold_list_);
    }
  };

  struct RangeSelector : public Selector {
    uint32_t key_;
    uint32_t base_key_;
    AttrIndex *attr_index_;
    VectorIDList prepared_list_;

    RangeSelector(uint32_t key, uint32_t base_key, AttrIndex *attr_index)
        : key_(key), base_key_(base_key), attr_index_(attr_index) {
    }

    // Leaf selectors only need a shallow copy because AttrIndex ownership is
    // managed outside the selector tree.
    Selector *copy() const override {
      return new RangeSelector(key_, base_key_, attr_index_);
    }

    virtual double estimate_selectivity(const Attributes &query_attrs) override {
      return (double) attr_index_->estimate_count(query_attrs.get(key_)) / attr_index_->n_vectors;
    }

    virtual double estimate_precision(const Attributes &query_attrs) override {
      return attr_index_->estimate_precision(query_attrs.get(key_));
    }

    virtual uint32_t estimate_prefilter_reads(const Attributes &query_attrs) override {
      return attr_index_->estimate_prefilter_reads(query_attrs.get(key_));
    }

    virtual VectorIDList pre_filter(const Attributes &query_attrs, AlignedFileReader *reader) override {
      return attr_index_->pre_filter(query_attrs.get(key_), reader);
    }

    virtual bool is_member(uint32_t target_id, const Attributes &query_attrs, const Attributes &target_attrs) override {
      if (!query_attrs.find(key_) || !target_attrs.find(key_)) {
        return false;
      }
      Attribute query_attr = query_attrs.get(key_);
      Attribute target_attr = target_attrs.get(key_);
      uint32_t l = query_attr[0], r = query_attr.size() > 1 ? query_attr[1] : l + 1;
      return target_attr[0] >= l && target_attr[0] < r;
    }

    virtual uint32_t estimate_infilter_reads(const Attributes &query_attrs) override {
      return attr_index_->estimate_infilter_reads(query_attrs.get(key_));
    }

    virtual void prepare_in_filter(const Attributes &query_attrs, AlignedFileReader *reader) override {
      prepared_list_ = attr_index_->prepare_in_filter(query_attrs.get(key_), reader);
    }

    virtual bool is_member_approx(uint32_t target_id, const Attributes &query_attrs) override {
      return attr_index_->is_member_approx(target_id, query_attrs.get(key_), prepared_list_);
    }
  };

  struct AndSelector : public Selector {
    std::vector<std::pair<Selector *, double>> selectors_;    // selector + selectivity
    static constexpr double kHighSelectivityThreshold = 0.1;  // Skip branches with selectivity > this in pre_filter.

    AndSelector(std::vector<Selector *> selectors) {
      for (auto &selector : selectors) {
        selectors_.push_back(std::make_pair(selector, 1.0));
      }
    }

    // The parent AndSelector owns and deletes its children, so we deep-copy
    // the child selector tree before handing ownership to the new parent.
    Selector *copy() const override {
      std::vector<Selector *> selectors;
      selectors.reserve(selectors_.size());
      for (const auto &selector : selectors_) {
        selectors.push_back(selector.first->copy());
      }
      return new AndSelector(std::move(selectors));
    }

    ~AndSelector() override {
      for (auto &selector : selectors_) {
        delete selector.first;
      }
    }

    virtual double estimate_selectivity(const Attributes &query_attrs) override {
      double selectivity = 1.0;
      for (auto &ss : selectors_) {
        ss.second = ss.first->estimate_selectivity(query_attrs);
        selectivity *= ss.second;
      }
      // from low selectivity to high selectivity.
      std::sort(selectors_.begin(), selectors_.end(), [](const auto &a, const auto &b) { return a.second < b.second; });
      return selectivity;
    }

    // For AND, precision is the product of individual precisions.
    virtual double estimate_precision(const Attributes &query_attrs) override {
      double precision = 1.0;
      for (auto &ss : selectors_) {
        precision *= ss.first->estimate_precision(query_attrs);
      }
      return precision;
    }

    virtual uint32_t estimate_prefilter_reads(const Attributes &query_attrs) override {
      uint64_t reads = 0;
      for (size_t i = 0; i < selectors_.size() && selectors_[i].second <= kHighSelectivityThreshold; i++) {
        reads += selectors_[i].first->estimate_prefilter_reads(query_attrs);
      }
      return reads;
    }

    // For pre_filter, skip high-selectivity branches to avoid expensive scans.
    // High-selectivity branches will be verified during exact is_member check later.
    virtual VectorIDList pre_filter(const Attributes &query_attrs, AlignedFileReader *reader) override {
      VectorIDList ret = selectors_[0].first->pre_filter(query_attrs, reader);
      for (size_t i = 1; i < selectors_.size(); i++) {
        if (selectors_[i].second <= kHighSelectivityThreshold) {
          auto cur_set = selectors_[i].first->pre_filter(query_attrs, reader);
          ret = and_sorted_unique(ret, cur_set);
        } else {
          LOG(INFO) << "Skip high-selectivity branch: " << selectors_[i].second;
        }
      }
      return ret;
    }

    virtual bool is_member(uint32_t target_id, const Attributes &query_attrs, const Attributes &target_attrs) override {
      for (auto &ss : selectors_) {
        if (!ss.first->is_member(target_id, query_attrs, target_attrs)) {
          return false;
        }
      }
      return true;
    }

    virtual uint32_t estimate_infilter_reads(const Attributes &query_attrs) override {
      uint64_t reads = 0;
      for (auto &ss : selectors_) {
        reads += ss.first->estimate_infilter_reads(query_attrs);
      }
      return reads;
    }

    virtual void prepare_in_filter(const Attributes &query_attrs, AlignedFileReader *reader) override {
      for (auto &ss : selectors_) {
        ss.first->prepare_in_filter(query_attrs, reader);
      }
    }

    virtual bool is_member_approx(uint32_t target_id, const Attributes &query_attrs) override {
      for (auto &ss : selectors_) {
        if (!ss.first->is_member_approx(target_id, query_attrs)) {
          return false;
        }
      }
      return true;
    }
  };

  struct OrSelector : public Selector {
    std::vector<Selector *> selectors_;
    OrSelector(std::vector<Selector *> selectors) : selectors_(std::move(selectors)) {
    }

    // The parent OrSelector owns and deletes its children, so we deep-copy
    // the child selector tree before handing ownership to the new parent.
    Selector *copy() const override {
      std::vector<Selector *> selectors;
      selectors.reserve(selectors_.size());
      for (const auto &selector : selectors_) {
        selectors.push_back(selector->copy());
      }
      return new OrSelector(std::move(selectors));
    }

    ~OrSelector() override {
      for (auto *selector : selectors_) {
        delete selector;
      }
    }

    // The selectors are mostly independent. We should use IEP to estimate the selectivity.
    // We simplify IEP by using the union of the selectivities.
    // This is because we only care about low-selectivity cases, where sel1 * sel2 is close to 0.
    virtual double estimate_selectivity(const Attributes &query_attrs) override {
      double selectivity = 0.0;
      for (auto &selector : selectors_) {
        selectivity += selector->estimate_selectivity(query_attrs);
      }
      return selectivity;
    }

    // For OR, precision = union(TP) / union(TP + FP).
    // union(TP) ≈ sum(sel_i * N) (simplified IEP for low selectivity).
    // union(TP + FP) ≈ sum(sel_i * N / p_i).
    virtual double estimate_precision(const Attributes &query_attrs) override {
      double tp_sum = 0.0, total_sum = 0.0;
      for (auto &selector : selectors_) {
        double sel = selector->estimate_selectivity(query_attrs);
        double p = selector->estimate_precision(query_attrs);
        tp_sum += sel;
        total_sum += sel / p;
      }
      return std::min(tp_sum / total_sum, 1.0);
    }

    virtual uint32_t estimate_prefilter_reads(const Attributes &query_attrs) override {
      uint64_t reads = 0;
      for (auto &selector : selectors_) {
        reads += selector->estimate_prefilter_reads(query_attrs);
      }
      return reads;
    }

    virtual uint32_t estimate_infilter_reads(const Attributes &query_attrs) override {
      uint64_t reads = 0;
      for (auto &selector : selectors_) {
        reads += selector->estimate_infilter_reads(query_attrs);
      }
      return reads;
    }

    virtual VectorIDList pre_filter(const Attributes &query_attrs, AlignedFileReader *reader) override {
      VectorIDList vector_id_set;
      for (auto &selector : selectors_) {
        auto ret = selector->pre_filter(query_attrs, reader);
        vector_id_set = or_sorted_unique(vector_id_set, ret);
      }
      return vector_id_set;
    }

    virtual bool is_member(uint32_t target_id, const Attributes &query_attrs, const Attributes &target_attrs) override {
      for (auto &selector : selectors_) {
        if (selector->is_member(target_id, query_attrs, target_attrs)) {
          return true;
        }
      }
      return false;
    }

    virtual void prepare_in_filter(const Attributes &query_attrs, AlignedFileReader *reader) override {
      for (auto &selector : selectors_) {
        selector->prepare_in_filter(query_attrs, reader);
      }
    }

    // OR semantics: any branch returning true means the vector is a member.
    virtual bool is_member_approx(uint32_t target_id, const Attributes &query_attrs) override {
      for (auto &selector : selectors_) {
        if (selector->is_member_approx(target_id, query_attrs)) {
          return true;
        }
      }
      return false;
    }
  };

  struct NotSelector : public Selector {
    Selector *selector_;
    uint32_t n_vectors_;
    VectorIDList vector_id_set_;
    NotSelector(Selector *selector, uint32_t n_vectors) : selector_(selector), n_vectors_(n_vectors) {
    }

    // Deep-copy the child tree so each query-local selector has its own state.
    Selector *copy() const override {
      return new NotSelector(selector_->copy(), n_vectors_);
    }

    ~NotSelector() override {
      delete selector_;
    }

    virtual double estimate_selectivity(const Attributes &query_attrs) override {
      return 1.0 - selector_->estimate_selectivity(query_attrs);
    }

    // NOT degrades to pre-filter (exact), so precision is 1.0.
    virtual double estimate_precision(const Attributes &query_attrs) override {
      return 1.0;
    }

    virtual uint32_t estimate_prefilter_reads(const Attributes &query_attrs) override {
      return selector_->estimate_prefilter_reads(query_attrs);
    }

    virtual VectorIDList pre_filter(const Attributes &query_attrs, AlignedFileReader *reader) override {
      VectorIDList not_vector_ids = selector_->pre_filter(query_attrs, reader);
      VectorIDList vector_ids;
      vector_ids.reserve(n_vectors_ - not_vector_ids.size());

      uint32_t start_id = 0;

      for (uint32_t exclude_id : not_vector_ids) {
        for (uint32_t id = start_id; id < exclude_id; ++id) {
          vector_ids.push_back(id);
        }
        start_id = exclude_id + 1;
      }

      for (uint32_t id = start_id; id < n_vectors_; ++id) {
        vector_ids.push_back(id);
      }
      return vector_ids;
    }

    virtual bool is_member(uint32_t target_id, const Attributes &query_attrs, const Attributes &target_attrs) override {
      return !selector_->is_member(target_id, query_attrs, target_attrs);
    }

    // degrade to pre-filter (as NOT-supersets have false negatives).
    virtual uint32_t estimate_infilter_reads(const Attributes &query_attrs) override {
      return this->estimate_prefilter_reads(query_attrs);
    }

    virtual void prepare_in_filter(const Attributes &query_attrs, AlignedFileReader *reader) override {
      vector_id_set_ = this->pre_filter(query_attrs, reader);
    }

    virtual bool is_member_approx(uint32_t target_id, const Attributes &) override {
      return std::binary_search(vector_id_set_.begin(), vector_id_set_.end(), target_id);
    }
  };

  // Recursively parse query config to build Selector tree and collect query attrs.
  // query_attrs: out param to collect all loaded attrs by key
  inline Selector *parse_selector_from_json(const picojson::value &node,
                                            const std::map<uint32_t, AttrIndex *> &base_stores,
                                            std::vector<Attributes> &query_attrs) {
    const auto &obj = node.get<picojson::object>();
    std::string type = obj.at("type").get<std::string>();

    if (type == "label" || type == "label_and" || type == "range") {
      uint32_t key = static_cast<uint32_t>(obj.at("key").get<double>());
      uint32_t base_key = static_cast<uint32_t>(obj.at("base_key").get<double>());
      std::string file = obj.at("file").get<std::string>();
      std::string load_type = (type == "label_and") ? "label" : type;
      load_query_attrs_from_file(file, load_type, key, query_attrs);

      auto it = base_stores.find(base_key);
      if (it == base_stores.end()) {
        LOG(ERROR) << "Base attr index not found for base_key: " << base_key;
        crash();
      }

      if (type == "label") {
        return new LabelOrSelector(key, base_key, it->second);
      } else if (type == "label_and") {
        return new LabelAndSelector(key, base_key, it->second);
      } else /* type == "range" */ {
        return new RangeSelector(key, base_key, it->second);
      }
    } else if (type == "and" || type == "or") {
      const auto &children = obj.at("children").get<picojson::array>();
      std::vector<Selector *> selectors;
      for (const auto &child : children) {
        selectors.push_back(parse_selector_from_json(child, base_stores, query_attrs));
      }
      if (type == "and") {
        return new AndSelector(std::move(selectors));
      } else /* type == "or" */ {
        return new OrSelector(std::move(selectors));
      }
    } else if (type == "not") {
      const auto &child = obj.at("children").get<picojson::array>()[0];
      return new NotSelector(parse_selector_from_json(child, base_stores, query_attrs),
                             base_stores.begin()->second->n_vectors);
    } else {
      LOG(ERROR) << "Unknown selector type: " << type;
      return nullptr;
    }
  }

  // Load selector and query attrs from config JSON.
  inline std::pair<Selector *, std::vector<Attributes>> load_selector_from_config(
      const std::string &config_path, const std::map<uint32_t, AttrIndex *> &base_stores) {
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
      LOG(ERROR) << "Failed to open config file: " << config_path;
      crash();
    }

    picojson::value config;
    std::string err = picojson::parse(config, config_file);
    config_file.close();
    if (!err.empty()) {
      LOG(ERROR) << "Failed to parse config JSON: " << err;
      crash();
    }

    const auto &root = config.get<picojson::object>();
    const auto &query_node = root.at("query");

    std::vector<Attributes> query_attrs;
    auto selector = parse_selector_from_json(query_node, base_stores, query_attrs);
    LOG(INFO) << "Loaded selector from config with " << query_attrs.size() << " queries";
    return std::make_pair(selector, std::move(query_attrs));
  }
}  // namespace pipeann

#endif  // SELECTOR_H_
