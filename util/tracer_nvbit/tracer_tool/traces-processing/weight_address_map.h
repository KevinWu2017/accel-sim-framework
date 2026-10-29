#ifndef ACCEL_SIM_WEIGHT_ADDRESS_MAP_H
#define ACCEL_SIM_WEIGHT_ADDRESS_MAP_H

#include <algorithm>
#include <cstdint>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// Address ranges emitted by the Qwen WeightAddress capture script.
class WeightAddressMap {
 public:
  bool load(const std::string &path) {
    std::ifstream input(path);
    if (!input) return false;

    ranges_.clear();
    std::string line;
    while (std::getline(input, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream fields(line);
      std::string first, last;
      if (!(fields >> first >> last)) return false;
      try {
        size_t first_end = 0, last_end = 0;
        uint64_t start = std::stoull(first, &first_end, 16);
        uint64_t end = std::stoull(last, &last_end, 16);
        if (first_end != first.size() || last_end != last.size() || end <= start)
          return false;
        ranges_.push_back({start, end});
      } catch (const std::exception &) {
        return false;
      }
    }
    std::sort(ranges_.begin(), ranges_.end(),
              [](const Range &a, const Range &b) { return a.start < b.start; });
    return !ranges_.empty();
  }

  bool contains(uint64_t address) const {
    auto next = std::upper_bound(
        ranges_.begin(), ranges_.end(), address,
        [](uint64_t value, const Range &range) { return value < range.start; });
    if (next == ranges_.begin()) return false;
    --next;
    return address < next->end;
  }

  size_t size() const { return ranges_.size(); }

 private:
  struct Range {
    uint64_t start, end;
  };
  std::vector<Range> ranges_;
};

#endif
