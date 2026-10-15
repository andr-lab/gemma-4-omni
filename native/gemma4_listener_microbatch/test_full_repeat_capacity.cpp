#include <cassert>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "full_repeat_capacity.h"

using gemma4_full_repeat_staging::initial_capacity;
using gemma4_full_repeat_staging::next_capacity;
using gemma4_full_repeat_staging::repeat_buffer_bytes;
using gemma4_full_repeat_staging::replace_pair_transactionally;

static void test_capacity_boundaries() {
    uint32_t capacity = 0;
    assert(initial_capacity(16384, &capacity) && capacity == 2048);
    assert(initial_capacity(1024, &capacity) && capacity == 1024);
    assert(!initial_capacity(0, &capacity));
    assert(!initial_capacity(16384, nullptr));

    assert(next_capacity(2048, 2048, 16384, &capacity) && capacity == 2048);
    assert(next_capacity(2048, 2049, 16384, &capacity) && capacity == 4096);
    assert(next_capacity(4096, 4097, 16384, &capacity) && capacity == 8192);
    assert(next_capacity(8192, 16384, 16384, &capacity) && capacity == 16384);
    assert(!next_capacity(8192, 16385, 16384, &capacity));
    assert(!next_capacity(16385, 16385, 16384, &capacity));
    assert(!next_capacity(2048, 0, 16384, &capacity));
    assert(next_capacity(0, 1, 16384, &capacity) && capacity == 2048);
}

static void test_checked_bytes() {
    size_t bytes = 0;
    assert(repeat_buffer_bytes(8, 2048, 512, 2, &bytes));
    assert(bytes == 16u * 1024u * 1024u);
    assert(!repeat_buffer_bytes(0, 2048, 512, 2, &bytes));
    assert(!repeat_buffer_bytes(8, 2048, 0, 2, &bytes));
    assert(!repeat_buffer_bytes(std::numeric_limits<uint64_t>::max(), 2, 512, 2, &bytes));
    assert(!repeat_buffer_bytes(8, 2048, 512, 0, &bytes));
    assert(!repeat_buffer_bytes(8, 2048, 512, 2, nullptr));
}

static void test_pair_allocation_failures_are_transactional() {
    for (int fail_at : {1, 2}) {
        void *key = reinterpret_cast<void *>(std::uintptr_t(0x1010));
        void *value = reinterpret_cast<void *>(std::uintptr_t(0x2020));
        uint32_t live_capacity = 2048;
        int calls = 0;
        std::vector<void *> freed;
        auto allocate = [&](void **out, size_t, const char *) {
            ++calls;
            *out = reinterpret_cast<void *>(std::uintptr_t(0x3000 + calls));
            return calls != fail_at;
        };
        auto deallocate = [&](void *ptr) { freed.push_back(ptr); };
        auto fence = [] { return true; };
        assert(!replace_pair_transactionally(
            key, value, live_capacity, 4096, 8, "key", "value",
            allocate, deallocate, fence));
        assert(key == reinterpret_cast<void *>(std::uintptr_t(0x1010)));
        assert(value == reinterpret_cast<void *>(std::uintptr_t(0x2020)));
        assert(live_capacity == 2048);
        assert(freed.size() == static_cast<size_t>(fail_at));
    }
}

static void test_fence_failure_and_successful_publication() {
    void *key = reinterpret_cast<void *>(std::uintptr_t(0x1010));
    void *value = reinterpret_cast<void *>(std::uintptr_t(0x2020));
    uint32_t live_capacity = 2048;
    int calls = 0;
    std::vector<void *> freed;
    auto allocate = [&](void **out, size_t, const char *) {
        *out = reinterpret_cast<void *>(std::uintptr_t(0x3000 + ++calls));
        return true;
    };
    auto deallocate = [&](void *ptr) { freed.push_back(ptr); };
    auto failed_fence = [] { return false; };
    assert(!replace_pair_transactionally(
        key, value, live_capacity, 4096, 8, "key", "value",
        allocate, deallocate, failed_fence));
    assert(key == reinterpret_cast<void *>(std::uintptr_t(0x1010)));
    assert(value == reinterpret_cast<void *>(std::uintptr_t(0x2020)));
    assert(live_capacity == 2048);
    assert(freed.size() == 2);

    freed.clear();
    auto successful_fence = [] { return true; };
    assert(replace_pair_transactionally(
        key, value, live_capacity, 4096, 8, "key", "value",
        allocate, deallocate, successful_fence));
    assert(key == reinterpret_cast<void *>(std::uintptr_t(0x3003)));
    assert(value == reinterpret_cast<void *>(std::uintptr_t(0x3004)));
    assert(live_capacity == 4096);
    assert((freed == std::vector<void *> {
        reinterpret_cast<void *>(std::uintptr_t(0x1010)),
        reinterpret_cast<void *>(std::uintptr_t(0x2020)),
    }));
}

int main() {
    test_capacity_boundaries();
    test_checked_bytes();
    test_pair_allocation_failures_are_transactional();
    test_fence_failure_and_successful_publication();
    std::puts("full repeat capacity CPU contract: PASS");
}
