#pragma once

#include <unordered_map>
#include <shared_mutex>
#include <optional>

template <typename Key, typename Value>
class ThreadSafeDict {
private:
    std::unordered_map<Key, Value> data;
    mutable std::shared_mutex rw_mutex; // 读写锁

public:
    // 写入操作：独占锁
    void set(const Key& key, const Value& value) {
        std::unique_lock lock(rw_mutex);
        data[key] = value;
    }

    // 使用前检查key存在
    Value get(const Key& key) const {
        std::shared_lock lock(rw_mutex);    // 共享锁
        return data.at(key);
    }

    bool has(const Key& key) const {
        std::shared_lock lock(rw_mutex);
        auto it = data.find(key);
        return (it != data.end());
    }

    void remove(const Key& key) {
        std::unique_lock lock(rw_mutex);
        data.erase(key);
    }


    std::vector<Key> keys_copy() const {
        std::shared_lock lock(rw_mutex);
        std::vector<Key> result;
        result.reserve(data.size());
        for (const auto& kv : data) {
            result.push_back(kv.first);
        }
        return result;
    }
};