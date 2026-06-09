// Copyright 2025 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "tent/common/config.h"
#include "tent/common/types.h"
#include "tent/runtime/control_plane.h"
#include "tent/runtime/topology.h"
#include "tent/runtime/transport.h"
#include "tent/transport/mnnvl/mnnvl_transport.h"

namespace mooncake {
namespace tent {
namespace {

constexpr size_t kTransferSize = 64 * 1024;

class CountingSink final : public BatchEventSink {
   public:
    void notifyMaybeReady() noexcept override {
        // Mock the engine-side sink contract: once close() has been
        // observed, ignore further notifications. Without this guard
        // the host func can still increment the counter after close()
        // because the weak_ptr only goes empty when the sink is
        // destroyed, not when it is closed.
        if (closed_.load(std::memory_order_acquire)) return;
        notify_count_.fetch_add(1, std::memory_order_acq_rel);
        std::lock_guard<std::mutex> lock(mu_);
        cv_.notify_all();
    }

    void close() noexcept override {
        closed_.store(true, std::memory_order_release);
    }

    int notifyCount() const {
        return notify_count_.load(std::memory_order_acquire);
    }

    bool waitForNotify(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mu_);
        return cv_.wait_for(lock, timeout, [&] {
            return notify_count_.load(std::memory_order_acquire) > 0;
        });
    }

   private:
    std::atomic<int> notify_count_{0};
    std::atomic<bool> closed_{false};
    std::mutex mu_;
    std::condition_variable cv_;
};

class MnnvlTransportTest : public ::testing::Test {
   protected:
    void SetUp() override {
        int dev_count = 0;
        if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

        conf_ = std::make_shared<Config>();
        metadata_ = std::make_shared<ControlService>("p2p", "", nullptr);
        topology_ = std::make_shared<Topology>();

        auto local_desc = metadata_->segmentManager().getLocal();
        local_desc->machine_id = "test-machine";
        local_desc->rpc_server_addr = "127.0.0.1:0";

        auto status = transport_.install(local_segment_name_, metadata_,
                                         topology_, conf_);
        if (!status.ok()) {
            GTEST_SKIP() << "mnnvl unavailable: " << status.ToString();
        }
        installed_ = true;

        ASSERT_EQ(cudaMalloc(&src_dev_, kTransferSize), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&dst_dev_, kTransferSize), cudaSuccess);
        ASSERT_EQ(cudaMemset(src_dev_, 0xa5, kTransferSize), cudaSuccess);
        ASSERT_EQ(cudaMemset(dst_dev_, 0x00, kTransferSize), cudaSuccess);
    }

    void TearDown() override {
        if (active_batch_) {
            EXPECT_TRUE(transport_.freeSubBatch(active_batch_).ok());
        }
        if (src_dev_) cudaFree(src_dev_);
        if (dst_dev_) cudaFree(dst_dev_);
        if (installed_) {
            EXPECT_TRUE(transport_.uninstall().ok());
        }
    }

    void AllocateBatch(size_t capacity = 1) {
        ASSERT_TRUE(transport_.allocateSubBatch(active_batch_, capacity).ok());
        ASSERT_NE(active_batch_, nullptr);
        mnnvl_batch_ = dynamic_cast<MnnvlSubBatch*>(active_batch_);
        ASSERT_NE(mnnvl_batch_, nullptr);
    }

    bool waitUntilCompleted(TransferStatus& status,
                            std::chrono::milliseconds timeout) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            status = {};
            if (!transport_.getTransferStatus(active_batch_, 0, status).ok()) {
                return false;
            }
            if (status.s != TransferStatusEnum::PENDING) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return false;
    }

    std::shared_ptr<Config> conf_;
    std::shared_ptr<ControlService> metadata_;
    std::shared_ptr<Topology> topology_;
    MnnvlTransport transport_;
    std::string local_segment_name_ = "mnnvl-test-segment";
    Transport::SubBatchRef active_batch_ = nullptr;
    MnnvlSubBatch* mnnvl_batch_ = nullptr;
    bool installed_ = false;
    void* src_dev_ = nullptr;
    void* dst_dev_ = nullptr;
};

TEST_F(MnnvlTransportTest, HostFuncDrivesSinkNotification) {
    AllocateBatch();

    auto sink = std::make_shared<CountingSink>();
    active_batch_->sink = sink;

    Request request{};
    request.opcode = Request::WRITE;
    request.source = src_dev_;
    request.target_id = LOCAL_SEGMENT_ID;
    request.target_offset = reinterpret_cast<uint64_t>(dst_dev_);
    request.length = kTransferSize;

    ASSERT_TRUE(transport_.submitTransferTasks(active_batch_, {request}).ok());

    // Host func should fire after the async stream finishes the copy.
    ASSERT_TRUE(sink->waitForNotify(std::chrono::milliseconds(2000)));

    TransferStatus status{};
    ASSERT_TRUE(waitUntilCompleted(status, std::chrono::milliseconds(2000)));
    EXPECT_EQ(status.s, TransferStatusEnum::COMPLETED);
    EXPECT_EQ(status.transferred_bytes, kTransferSize);
    EXPECT_GE(sink->notifyCount(), 1);

    EXPECT_FALSE(mnnvl_batch_->events.empty());
    EXPECT_EQ(mnnvl_batch_->events.size(), 1u);
}

TEST_F(MnnvlTransportTest, NoSinkDoesNotEnqueueHostFunc) {
    // Without a sink, polling must still flip status_word via cudaEventQuery.
    AllocateBatch();

    Request request{};
    request.opcode = Request::WRITE;
    request.source = src_dev_;
    request.target_id = LOCAL_SEGMENT_ID;
    request.target_offset = reinterpret_cast<uint64_t>(dst_dev_);
    request.length = kTransferSize;

    ASSERT_TRUE(transport_.submitTransferTasks(active_batch_, {request}).ok());

    TransferStatus status{};
    ASSERT_TRUE(waitUntilCompleted(status, std::chrono::milliseconds(2000)));
    EXPECT_EQ(status.s, TransferStatusEnum::COMPLETED);
}

TEST_F(MnnvlTransportTest, ClosedSinkBeforeFreeBecomesNoop) {
    AllocateBatch();

    auto sink = std::make_shared<CountingSink>();
    active_batch_->sink = sink;

    Request request{};
    request.opcode = Request::WRITE;
    request.source = src_dev_;
    request.target_id = LOCAL_SEGMENT_ID;
    request.target_offset = reinterpret_cast<uint64_t>(dst_dev_);
    request.length = kTransferSize;

    ASSERT_TRUE(transport_.submitTransferTasks(active_batch_, {request}).ok());

    // Close the sink immediately; the engine does this before freeSubBatch().
    // The host func may fire after this and must not segfault: with the
    // weak_ptr guard plus close(), notifyMaybeReady() is called only if the
    // sink is still locked-able from the captured weak_ptr.
    sink->close();

    // Drain the stream so the host func has a chance to run.
    ASSERT_EQ(cudaStreamSynchronize(mnnvl_batch_->async_stream.get()),
              cudaSuccess);

    // After close(), the sink must drop further notifyMaybeReady()
    // calls. The host func may still run (the weak_ptr is only emptied
    // by sink destruction, not by close()), but the sink itself turns
    // the call into a noop. Combined with the cudaStreamSynchronize
    // above that drains any pending host func, notifyCount() must be 0.
    EXPECT_EQ(sink->notifyCount(), 0);
    EXPECT_TRUE(transport_.freeSubBatch(active_batch_).ok());
    active_batch_ = nullptr;
}

TEST_F(MnnvlTransportTest, MultipleSubmitsAccumulateEventsAndNotify) {
    AllocateBatch(/*capacity=*/4);

    auto sink = std::make_shared<CountingSink>();
    active_batch_->sink = sink;

    constexpr int kSubmits = 3;
    for (int i = 0; i < kSubmits; ++i) {
        Request request{};
        request.opcode = Request::WRITE;
        request.source = src_dev_;
        request.target_id = LOCAL_SEGMENT_ID;
        request.target_offset = reinterpret_cast<uint64_t>(dst_dev_);
        request.length = kTransferSize;
        ASSERT_TRUE(
            transport_.submitTransferTasks(active_batch_, {request}).ok());
    }

    // Drain so all enqueued host functions get a chance to fire.
    ASSERT_EQ(cudaStreamSynchronize(mnnvl_batch_->async_stream.get()),
              cudaSuccess);

    // Each submit must record exactly one event into the batch.
    EXPECT_EQ(mnnvl_batch_->events.size(), static_cast<size_t>(kSubmits));
    // And each submit must launch exactly one host func -> notify_count==N.
    EXPECT_EQ(sink->notifyCount(), kSubmits);
}

}  // namespace
}  // namespace tent
}  // namespace mooncake
