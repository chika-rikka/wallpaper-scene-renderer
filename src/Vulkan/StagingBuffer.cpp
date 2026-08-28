#include "StagingBuffer.hpp"
#include <algorithm>
#include <cstring>
#include "Util.hpp"
#include "Device.hpp"

using namespace wallpaper::vulkan;

#define CHECK_REF(ref, act)                                                  \
    if (! ref) {                                                             \
        LOG_ERROR("stage ref not available, index %d", ref.m_virtual_index); \
        { act; }                                                             \
    }

StagingBuffer::StagingBuffer(const Device& d, VkDeviceSize size, VkBufferUsageFlags usage)
    : m_device(d), m_size_step(size), m_usage(usage) {}
StagingBuffer::~StagingBuffer() {}

namespace
{
std::optional<VmaBufferParameters> CreateGpuBuffer(VmaAllocator allocator, VkBufferUsageFlags usage,
                                                   std::size_t size) {
    do {
        VmaBufferParameters buffer;
        VkBufferCreateInfo  ci {
             .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
             .pNext = nullptr,
             .size  = size,
             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
        };
        buffer.req_size                  = ci.size;
        VmaAllocationCreateInfo vma_info = {};
        vma_info.usage                   = VMA_MEMORY_USAGE_GPU_ONLY;
        VVK_CHECK_ACT(break, vvk::CreateBuffer(allocator, ci, vma_info, buffer.handle));
        return buffer;
    } while (false);
    return std::nullopt;
}

void RecordCopyBufferRange(const BufferParameters& dst_buf, const BufferParameters& src_buf,
                           VkDeviceSize offset, VkDeviceSize size, vvk::CommandBuffer& cmd) {
    // The CPU staging allocation is authoritative, but dynamic buffers may reserve far more bytes
    // than a frame actually mutates. Copying a precise range keeps the GPU mirror synchronized
    // without turning every particle tick into a full-buffer transfer.
    VkBufferCopy copy {
        .srcOffset = offset,
        .dstOffset = offset,
        .size      = size,
    };
    cmd.CopyBuffer(src_buf.handle, dst_buf.handle, copy);

    VkBufferMemoryBarrier in_bar {
        .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .pNext         = nullptr,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT,
        .buffer        = dst_buf.handle,
        .offset        = offset,
        .size          = size,
    };
    cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                        VK_DEPENDENCY_BY_REGION_BIT,
                        in_bar);
}
} // namespace

void StagingBuffer::markDirty(VkDeviceSize offset, VkDeviceSize size) {
    if (size == 0) return;

    // Shader uniform updates write hundreds of tiny std140 slots per frame. Merging those ranges
    // on every write is quadratic in the live dirty list. Record the interval here and collapse
    // once in recordUpload() so the write path stays O(1).
    m_dirty_ranges.push_back(DirtyRange { .offset = offset, .size = size });
}

void StagingBuffer::coalesceDirtyRanges() {
    if (m_dirty_ranges.size() <= 1) return;

    std::sort(m_dirty_ranges.begin(),
              m_dirty_ranges.end(),
              [](const DirtyRange& lhs, const DirtyRange& rhs) {
                  if (lhs.offset != rhs.offset) return lhs.offset < rhs.offset;
                  return lhs.size > rhs.size;
              });

    // std140 float arrays store one value every 16 bytes, so adjacent uniform writes leave 12-byte
    // holes. Absorbing a small gap keeps GPU copies proportional to the UBO or vertex live prefix
    // instead of one vkCmdCopyBuffer per member. 256 bytes is well below a typical uniform block
    // and will not swallow a neighbouring 2 MiB particle mesh unless that mesh is already dirty.
    constexpr VkDeviceSize kMergeGap = 256;

    std::vector<DirtyRange> merged;
    merged.reserve(m_dirty_ranges.size());
    DirtyRange current = m_dirty_ranges.front();
    for (size_t i = 1; i < m_dirty_ranges.size(); ++i) {
        const DirtyRange& next        = m_dirty_ranges[i];
        const VkDeviceSize current_end = current.offset + current.size;
        if (next.offset <= current_end + kMergeGap) {
            const VkDeviceSize next_end = next.offset + next.size;
            if (next_end > current_end) current.size = next_end - current.offset;
            continue;
        }
        merged.push_back(current);
        current = next;
    }
    merged.push_back(current);
    m_dirty_ranges.swap(merged);
}

void StagingBuffer::markAllDirty() {
    m_dirty_ranges.clear();
    if (m_stage_buf.handle && m_stage_buf.req_size > 0) {
        // Growing the CPU staging allocation also drops the GPU mirror. The newly-created mirror
        // has none of the old CPU contents, so the next upload must replay the whole staging image
        // even if individual writes were already clean before the resize.
        m_dirty_ranges.push_back(DirtyRange { .offset = 0, .size = m_stage_buf.req_size });
    }
}

StagingBuffer::VirtualBlock* StagingBuffer::newVirtualBlock(VkDeviceSize nsize) {
    auto it = std::find_if(m_virtual_blocks.begin(), m_virtual_blocks.end(), [nsize](auto& b) {
        return ! b.enabled && b.size >= nsize;
    });
    if (it == std::end(m_virtual_blocks)) {
        VkDeviceSize offset = m_virtual_blocks.empty()
                                  ? 0
                                  : m_virtual_blocks.back().offset + m_virtual_blocks.back().size;

        m_virtual_blocks.push_back({});
        it         = m_virtual_blocks.end() - 1;
        it->size   = nsize > m_size_step ? nsize : m_size_step;
        it->index  = (size_t)std::distance(m_virtual_blocks.begin(), it);
        it->offset = offset;
    }
    auto& block = *it;

    VmaVirtualBlockCreateInfo blockCreateInfo = {};
    blockCreateInfo.size                      = block.size;

    VVK_CHECK_ACT(return nullptr, vmaCreateVirtualBlock(&blockCreateInfo, &block.handle));
    block.enabled = true;

    LOG_INFO("new buffer block(%p), size: %d, index: %d / %d",
             this,
             block.size,
             block.index,
             m_virtual_blocks.size());
    return &block;
}
bool StagingBuffer::increaseBuf(VkDeviceSize nsize) {
    if (m_stage_raw == nullptr) {
        VVK_CHECK_BOOL_RE(mapStageBuf());
    }
    auto newsize = m_stage_buf.req_size + nsize;
    // do double copy
    std::vector<uint8_t> tmp;
    tmp.resize(newsize);
    memcpy(tmp.data(), m_stage_raw, m_stage_buf.req_size);

    m_stage_raw = nullptr;
    m_stage_buf.handle.UnMapMemory();
    m_stage_buf.handle = nullptr;

    if (! CreateStagingBuffer(m_device.vma_allocator(), newsize, m_stage_buf)) return false;
    VVK_CHECK_BOOL_RE(mapStageBuf());
    memcpy(m_stage_raw, tmp.data(), newsize);

    m_gpu_buf.handle = nullptr;
    markAllDirty();
    LOG_INFO("increase buffer size: %d", nsize);
    return true;
}

bool StagingBuffer::allocate() {
    if (! CreateStagingBuffer(m_device.vma_allocator(), m_size_step, m_stage_buf)) return false;
    VVK_CHECK_BOOL_RE(m_stage_buf.handle.MapMemory(&m_stage_raw));
    auto* block = newVirtualBlock(m_size_step);
    m_dirty_ranges.reserve(2048);
    return block != nullptr;
}

void StagingBuffer::destroy() {
    if (m_stage_raw != nullptr) {
        m_stage_buf.handle.UnMapMemory();
    }
    for (auto& block : m_virtual_blocks) {
        if (block.enabled) {
            vmaClearVirtualBlock(block.handle);
            vmaDestroyVirtualBlock(block.handle);
        }
    }
    m_virtual_blocks.clear();

    m_stage_buf = {};
    m_gpu_buf   = {};
    m_dirty_ranges.clear();
}

bool StagingBuffer::allocateSubRef(VkDeviceSize size, StagingBufferRef& ref,
                                   VkDeviceSize alignment) {
    VkResult                       result;
    VmaVirtualAllocationCreateInfo allocCreateInfo = {};
    allocCreateInfo.size                           = size;
    allocCreateInfo.alignment                      = alignment;

    VmaVirtualAllocation allocation;
    VkDeviceSize         offset;

    auto setRef = [&offset, &allocation, size](StagingBufferRef& ref, VirtualBlock& block) {
        ref.size   = size;
        ref.offset = offset + block.offset;

        ref.m_allocation    = allocation;
        ref.m_virtual_index = block.index;
    };

    for (auto& block : m_virtual_blocks) {
        if (block.enabled && block.size >= size) {
            if (auto res = vmaVirtualAllocate(block.handle, &allocCreateInfo, &allocation, &offset);
                res == VK_SUCCESS) {
                setRef(ref, block);
                return true;
            }
        }
    }

    auto  old_block_num = m_virtual_blocks.size();
    auto* p_block       = newVirtualBlock(size);
    if (p_block == nullptr) return false;

    auto& block = *p_block;
    if (old_block_num < m_virtual_blocks.size()) {
        if (! increaseBuf(block.size)) {
            auto& block = m_virtual_blocks.back();
            vmaClearVirtualBlock(block.handle);
            vmaDestroyVirtualBlock(block.handle);
            m_virtual_blocks.pop_back();
            LOG_ERROR("increase buf failed, pop_back block, current: %d", m_virtual_blocks.size());
            return false;
        }
    }
    VVK_CHECK_BOOL_RE(vmaVirtualAllocate(block.handle, &allocCreateInfo, &allocation, &offset));
    setRef(ref, block);
    return true;
}
void StagingBuffer::unallocateSubRef(StagingBufferRef& ref) {
    if (!ref) return;
    if (ref.m_virtual_index < m_virtual_blocks.size()) {
        auto& block = m_virtual_blocks[ref.m_virtual_index];
        vmaVirtualFree(block.handle, ref.m_allocation);
        if (block.enabled && vmaIsVirtualBlockEmpty(block.handle)) {
            vmaDestroyVirtualBlock(block.handle);
            block.handle  = VK_NULL_HANDLE;
            block.enabled = false;
        }
    } else {
        LOG_ERROR("unallocate stagingbuffer failed: wrong index %d", ref.m_virtual_index);
    }
    // Resource-only refreshes can tear down and immediately retry a pass when text-backed effect
    // geometry changes. Clearing the ref after the virtual allocation is released makes repeated
    // cleanup idempotent instead of allowing a stale VMA handle to be freed twice.
    ref.size = 0;
    ref.offset = 0;
    ref.m_allocation = VK_NULL_HANDLE;
    ref.m_virtual_index = 0;
}

VkResult StagingBuffer::mapStageBuf() { return m_stage_buf.handle.MapMemory(&m_stage_raw); }

bool StagingBuffer::writeToBuf(const StagingBufferRef& ref, std::span<uint8_t> data,
                               size_t offset) {
    CHECK_REF(ref, return false);

    if (m_stage_raw == nullptr) {
        mapStageBuf();
    }
    VkDeviceSize size = std::min(ref.size - offset, data.size());
    uint8_t*     raw  = (uint8_t*)m_stage_raw;
    std::copy(data.begin(), data.begin() + size, raw + ref.offset + offset);
    markDirty(ref.offset + offset, size);
    return true;
}

bool StagingBuffer::peekBytes(const StagingBufferRef& ref, std::span<uint8_t> out) {
    CHECK_REF(ref, return false);
    if (out.empty()) return true;
    if (m_stage_raw == nullptr) {
        mapStageBuf();
    }
    if (m_stage_raw == nullptr) return false;
    const size_t n = std::min(out.size(), static_cast<size_t>(ref.size));
    const uint8_t* raw = static_cast<const uint8_t*>(m_stage_raw) + ref.offset;
    std::copy(raw, raw + n, out.begin());
    return true;
}

bool StagingBuffer::fillBuf(const StagingBufferRef& ref, size_t offset, size_t size, uint8_t c) {
    CHECK_REF(ref, return false);

    if (m_stage_raw == nullptr) {
        mapStageBuf();
    }
    VkDeviceSize size_     = std::min(ref.size - offset, size);
    uint8_t*     raw       = (uint8_t*)m_stage_raw;
    uint8_t*     raw_begin = raw + ref.offset + offset;
    std::fill(raw_begin, raw_begin + size_, c);
    markDirty(ref.offset + offset, size_);
    return true;
}

bool StagingBuffer::recordUpload(vvk::CommandBuffer& cmd) {
    if (! m_gpu_buf.handle) {
        if (auto opt = CreateGpuBuffer(m_device.vma_allocator(), m_usage, m_stage_buf.req_size);
            opt.has_value()) {
            m_gpu_buf = std::move(opt.value());
        } else
            return false;
    }
    if (m_dirty_ranges.empty()) {
        return true;
    }
    coalesceDirtyRanges();
    if (m_stage_raw != nullptr) {
        m_stage_buf.handle.UnMapMemory();
        m_stage_raw = nullptr;
    }
    for (const auto& range : m_dirty_ranges) {
        VVK_CHECK_BOOL_RE(vmaFlushAllocation(m_device.vma_allocator(),
                                             m_stage_buf.handle.Allocation(),
                                             range.offset,
                                             range.size));
        RecordCopyBufferRange(m_gpu_buf, m_stage_buf, range.offset, range.size, cmd);
    }
    m_dirty_ranges.clear();
    return true;
}

VkBuffer StagingBuffer::gpuBuf() const { return *m_gpu_buf.handle; }

VkDeviceSize StagingBuffer::stageBytes() const {
    return m_stage_buf.handle ? m_stage_buf.handle.AllocationSize() : 0;
}

VkDeviceSize StagingBuffer::gpuBytes() const {
    return m_gpu_buf.handle ? m_gpu_buf.handle.AllocationSize() : 0;
}

VkDeviceSize StagingBuffer::trackedBytes() const {
    return stageBytes() + gpuBytes();
}

size_t StagingBuffer::blockCount() const {
    return m_virtual_blocks.size();
}
