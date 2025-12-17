// 索引构建时间统计工具
// 用于细粒度统计索引构建过程中各部分的时间占比，支持多线程环境
// 提供两种统计视角：
//   1. CPU时间（所有线程累加）- 反映计算量占比
//   2. Wall时间（真实世界时间）- 反映端到端耗时占比
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace pipeann {

  // 单个统计项的数据结构
  struct StatEntry {
    double total_us = 0;      // 累计时间（微秒）
    uint64_t count = 0;       // 调用次数
    double start_time = 0;    // 当前计时的起始时间
    bool running = false;     // 是否正在计时
  };

  // 单线程的构建统计类（用于CPU时间统计）
  class BuildStats {
   public:
    BuildStats() = default;

    void start(const std::string &label) {
      auto &entry = stats_[label];
      if (!entry.running) {
        entry.start_time = get_time_us();
        entry.running = true;
      }
    }

    void stop(const std::string &label) {
      auto it = stats_.find(label);
      if (it != stats_.end() && it->second.running) {
        double elapsed = get_time_us() - it->second.start_time;
        it->second.total_us += elapsed;
        it->second.count++;
        it->second.running = false;
      }
    }

    void add(const std::string &label, double time_us, uint64_t cnt = 1) {
      auto &entry = stats_[label];
      entry.total_us += time_us;
      entry.count += cnt;
    }

    void merge(const BuildStats &other) {
      for (const auto &kv : other.stats_) {
        auto &entry = stats_[kv.first];
        entry.total_us += kv.second.total_us;
        entry.count += kv.second.count;
      }
    }

    void clear() {
      stats_.clear();
    }

    double get_total_us(const std::string &label) const {
      auto it = stats_.find(label);
      return it != stats_.end() ? it->second.total_us : 0;
    }

    uint64_t get_count(const std::string &label) const {
      auto it = stats_.find(label);
      return it != stats_.end() ? it->second.count : 0;
    }

    double get_total_time_us() const {
      double total = 0;
      for (const auto &kv : stats_) {
        total += kv.second.total_us;
      }
      return total;
    }

    const std::unordered_map<std::string, StatEntry> &get_stats() const {
      return stats_;
    }

    // 生成CPU时间汇总报告
    std::string report(const std::string &title = "CPU Time Statistics (All Threads)") const {
      std::ostringstream oss;
      oss << "\n========== " << title << " ==========\n";

      double total_time = 0;
      double distance_calc_time = 0;
      uint64_t distance_calc_count = 0;
      std::vector<std::pair<std::string, StatEntry>> sorted_stats;
      
      for (const auto &kv : stats_) {
        if (kv.first == "distance_calc") {
          distance_calc_time = kv.second.total_us;
          distance_calc_count = kv.second.count;
        } else {
          sorted_stats.emplace_back(kv);
          total_time += kv.second.total_us;
        }
      }
      
      std::sort(sorted_stats.begin(), sorted_stats.end(),
                [](const auto &a, const auto &b) { return a.second.total_us > b.second.total_us; });

      oss << std::left << std::setw(20) << "Label" << std::right << std::setw(15) << "CPU Time(s)" << std::setw(12)
          << "Percent" << std::setw(15) << "Count" << std::setw(12) << "Avg(us)\n";
      oss << std::string(74, '-') << "\n";

      for (const auto &kv : sorted_stats) {
        double pct = total_time > 0 ? 100.0 * kv.second.total_us / total_time : 0;
        double avg_us = kv.second.count > 0 ? kv.second.total_us / kv.second.count : 0;
        oss << std::left << std::setw(20) << kv.first << std::right << std::fixed << std::setprecision(3)
            << std::setw(15) << (kv.second.total_us / 1e6) << std::setprecision(1) << std::setw(11) << pct << "%"
            << std::setw(15) << kv.second.count << std::setprecision(1) << std::setw(12) << avg_us << "\n";
      }

      oss << std::string(74, '-') << "\n";
      oss << std::left << std::setw(20) << "TOTAL" << std::right << std::fixed << std::setprecision(3) << std::setw(15)
          << (total_time / 1e6) << std::setprecision(1) << std::setw(11) << "100.0%\n";
      
      if (distance_calc_count > 0) {
        double dist_pct = total_time > 0 ? 100.0 * distance_calc_time / total_time : 0;
        double avg_us = distance_calc_time / distance_calc_count;
        oss << "\n[Distance Calculation Breakdown]\n";
        oss << "  Time: " << std::fixed << std::setprecision(3) << (distance_calc_time / 1e6) << "s"
            << " (" << std::setprecision(1) << dist_pct << "% of total CPU time)\n";
        oss << "  Count: " << distance_calc_count << ", Avg: " << std::setprecision(3) << avg_us << " us\n";
      }
      
      return oss.str();
    }

   private:
    std::unordered_map<std::string, StatEntry> stats_;

    static double get_time_us() {
      return std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now().time_since_epoch())
          .count();
    }
  };

  // 多线程CPU时间统计类（单例模式）
  // 每个线程有独立的 BuildStats，避免锁竞争
  class ThreadBuildStats {
   public:
    static ThreadBuildStats &instance() {
      static ThreadBuildStats inst;
      return inst;
    }

    void init(int num_threads) {
      std::lock_guard<std::mutex> lock(mutex_);
      thread_stats_.clear();
      thread_stats_.resize(num_threads);
    }

    BuildStats &get_local() {
      int tid = omp_get_thread_num();
      return thread_stats_[tid];
    }

    void start(const std::string &label) {
      get_local().start(label);
    }

    void stop(const std::string &label) {
      get_local().stop(label);
    }

    BuildStats aggregate() const {
      BuildStats result;
      for (const auto &ts : thread_stats_) {
        result.merge(ts);
      }
      return result;
    }

    void clear() {
      for (auto &ts : thread_stats_) {
        ts.clear();
      }
    }

   private:
    ThreadBuildStats() = default;
    std::vector<BuildStats> thread_stats_;
    std::mutex mutex_;
  };

  // Wall Time 统计项（线程安全）
  // 通过记录时间区间并合并重叠区间来计算真实wall time
  struct WallStatEntry {
    std::mutex mutex;
    std::vector<std::pair<double, double>> intervals;  // (start_us, end_us)
    std::atomic<uint64_t> count{0};

    void add_interval(double start_us, double end_us) {
      {
        std::lock_guard<std::mutex> lock(mutex);
        intervals.emplace_back(start_us, end_us);
      }
      count.fetch_add(1, std::memory_order_relaxed);
    }

    // 合并重叠区间，计算真实wall time
    double compute_merged_time() {
      std::lock_guard<std::mutex> lock(mutex);
      if (intervals.empty()) return 0;

      std::sort(intervals.begin(), intervals.end());

      double total = 0;
      double cur_start = intervals[0].first;
      double cur_end = intervals[0].second;

      for (size_t i = 1; i < intervals.size(); ++i) {
        if (intervals[i].first <= cur_end) {
          cur_end = std::max(cur_end, intervals[i].second);
        } else {
          total += cur_end - cur_start;
          cur_start = intervals[i].first;
          cur_end = intervals[i].second;
        }
      }
      total += cur_end - cur_start;
      return total;
    }
  };

  // Wall Time 统计类（线程安全，用于统计真实世界时间）
  // 通过记录时间区间并合并来正确处理多线程并行执行的情况
  class WallTimeStats {
   public:
    static WallTimeStats &instance() {
      static WallTimeStats inst;
      return inst;
    }

    void clear() {
      std::lock_guard<std::mutex> lock(mutex_);
      stats_.clear();
    }

    // 记录一个时间区间 [start_us, end_us]
    void add_interval(const std::string &label, double start_us, double end_us) {
      get_entry(label).add_interval(start_us, end_us);
    }

    std::string report(double total_wall_time_us, const std::string &title = "Wall Time Statistics") {
      std::ostringstream oss;
      oss << "\n========== " << title << " ==========\n";

      std::vector<std::tuple<std::string, double, uint64_t>> sorted_stats;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto &kv : stats_) {
          double merged_time = kv.second->compute_merged_time();
          uint64_t cnt = kv.second->count.load();
          sorted_stats.emplace_back(kv.first, merged_time, cnt);
        }
      }
      std::sort(sorted_stats.begin(), sorted_stats.end(),
                [](const auto &a, const auto &b) { return std::get<1>(a) > std::get<1>(b); });

      oss << std::left << std::setw(20) << "Label" << std::right << std::setw(15) << "Wall Time(s)" << std::setw(12)
          << "Percent" << std::setw(15) << "Count" << std::setw(12) << "Avg CPU(us)\n";
      oss << std::string(74, '-') << "\n";

      for (const auto &item : sorted_stats) {
        const std::string &label = std::get<0>(item);
        double time = std::get<1>(item);
        uint64_t cnt = std::get<2>(item);
        double pct = total_wall_time_us > 0 ? 100.0 * time / total_wall_time_us : 0;
        // 从CPU统计获取平均时间，更有意义
        double avg_cpu_us = cnt > 0 ? ThreadBuildStats::instance().aggregate().get_total_us(label) / cnt : 0;
        oss << std::left << std::setw(20) << label << std::right << std::fixed << std::setprecision(3)
            << std::setw(15) << (time / 1e6) << std::setprecision(1) << std::setw(11) << pct << "%" << std::setw(15)
            << cnt << std::setprecision(1) << std::setw(12) << avg_cpu_us << "\n";
      }

      oss << std::string(74, '-') << "\n";
      oss << std::left << std::setw(20) << "Total Wall Time" << std::right << std::fixed << std::setprecision(3)
          << std::setw(15) << (total_wall_time_us / 1e6) << "s\n";
      return oss.str();
    }

   private:
    WallTimeStats() = default;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<WallStatEntry>> stats_;

    WallStatEntry &get_entry(const std::string &label) {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = stats_.find(label);
      if (it == stats_.end()) {
        stats_[label] = std::make_unique<WallStatEntry>();
        return *stats_[label];
      }
      return *it->second;
    }
  };

  class ScopedTimer {
   public:
    ScopedTimer(const std::string &label) : label_(label) {
      start_time_ = get_time_us();
    }
    ~ScopedTimer() {
      double elapsed = get_time_us() - start_time_;
      ThreadBuildStats::instance().get_local().add(label_, elapsed);
    }
   private:
    std::string label_;
    double start_time_;
    static double get_time_us() {
      return std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now().time_since_epoch())
          .count();
    }
  };

#ifdef ENABLE_BUILD_STATS
// 便捷宏定义
#define BUILD_STATS_INIT(n) pipeann::ThreadBuildStats::instance().init(n)
#define BUILD_STATS_START(label) pipeann::ThreadBuildStats::instance().start(label)
#define BUILD_STATS_STOP(label) pipeann::ThreadBuildStats::instance().stop(label)
#define BUILD_STATS_SCOPED(label) pipeann::ScopedTimer _scoped_timer_##__LINE__(label)
#define BUILD_STATS_REPORT() pipeann::ThreadBuildStats::instance().aggregate().report()
#define BUILD_STATS_CLEAR() pipeann::ThreadBuildStats::instance().clear()

#define WALL_STATS_CLEAR() pipeann::WallTimeStats::instance().clear()
#define WALL_STATS_ADD(label, start_us, end_us) pipeann::WallTimeStats::instance().add_interval(label, start_us, end_us)
#define WALL_STATS_REPORT(total_us) pipeann::WallTimeStats::instance().report(total_us)

#define TIMED_COMPARE(dist_ptr, a, b, dim) \
  [&]() { \
    thread_local uint64_t _sample_counter = 0; \
    constexpr uint64_t _sample_rate = 1024; \
    float _result; \
    if ((_sample_counter++ & (_sample_rate - 1)) == 0) { \
      double _start = pipeann::get_time_us(); \
      _result = (dist_ptr)->compare((a), (b), (dim)); \
      double _elapsed = pipeann::get_time_us() - _start; \
      pipeann::ThreadBuildStats::instance().get_local().add("distance_calc", _elapsed * _sample_rate, _sample_rate); \
    } else { \
      _result = (dist_ptr)->compare((a), (b), (dim)); \
    } \
    return _result; \
  }()

#else
// 禁用统计时的空宏
#define BUILD_STATS_INIT(n) ((void)0)
#define BUILD_STATS_START(label) ((void)0)
#define BUILD_STATS_STOP(label) ((void)0)
#define BUILD_STATS_SCOPED(label) ((void)0)
#define BUILD_STATS_REPORT() std::string()
#define BUILD_STATS_CLEAR() ((void)0)

#define WALL_STATS_CLEAR() ((void)0)
#define WALL_STATS_ADD(label, start_us, end_us) ((void)0)
#define WALL_STATS_REPORT(total_us) std::string()

#define TIMED_COMPARE(dist_ptr, a, b, dim) ((dist_ptr)->compare((a), (b), (dim)))

#endif // ENABLE_BUILD_STATS

  // 获取当前时间（微秒）
  inline double get_time_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
  }

}  // namespace pipeann