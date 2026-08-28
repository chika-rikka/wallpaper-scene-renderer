#pragma once
#include "Core/NoCopyMove.hpp"
#include "Instance.hpp"
#include "Parameters.hpp"
#include "vk_mem_alloc.h"

#include <cstdint>
#include <span>
#include <vector>

namespace wallpaper
{
namespace vulkan
{

class Device;
class StagingBuffer;

class StagingBufferRef {
public:
    VkDeviceSize size { 0 };
    VkDeviceSize offset { 0 };

    operator bool() const { return m_allocation != VK_NULL_HANDLE; }

private:
    friend class StagingBuffer;
    VmaVirtualAllocation m_allocation {};
    size_t               m_virtual_index { 0 };
};

class StagingBuffer : NoCopy, NoMove {
public:
    StagingBuffer(const Device&, VkDeviceSize size, VkBufferUsageFlags);
    ~StagingBuffer();

    bool allocate();
    void destroy();

    bool allocateSubRef(VkDeviceSize size, StagingBufferRef&, VkDeviceSize alignment = 1);
    void unallocateSubRef(StagingBufferRef&);
    bool writeToBuf(const StagingBufferRef&, std::span<uint8_t>, size_t offset = 0);
    bool fillBuf(const StagingBufferRef& ref, size_t offset, size_t size, uint8_t c);
    bool peekBytes(const StagingBufferRef&, std::span<uint8_t>);

    bool recordUpload(vvk::CommandBuffer&);

    VkBuffer gpuBuf() const;
    VkDeviceSize stageBytes() const;
    VkDeviceSize gpuBytes() const;
    VkDeviceSize trackedBytes() const;
    size_t       blockCount() const;

private:
    struct DirtyRange {
        VkDeviceSize offset { 0 };
        VkDeviceSize size { 0 };
    };

    struct VirtualBlock {
        VmaVirtualBlock handle {};
        bool            enabled { false };
        size_t          index { 0 };
        VkDeviceSize    offset { 0 };
        VkDeviceSize    size { 0 };
    };

    VkResult      mapStageBuf();
    VirtualBlock* newVirtualBlock(VkDeviceSize);
    bool          increaseBuf(VkDeviceSize);
    void          markDirty(VkDeviceSize offset, VkDeviceSize size);
    void          markAllDirty();
    void          coalesceDirtyRanges();

    const Device& m_device;
    VkDeviceSize  m_size_step;

    VkBufferUsageFlags m_usage;

    void*                     m_stage_raw { nullptr };
    std::vector<VirtualBlock> m_virtual_blocks {};

    VmaBufferParameters m_stage_buf;
    VmaBufferParameters m_gpu_buf;
    // Dynamic staging buffers can become much larger than the bytes touched in a frame. Tracking
    // merged dirty ranges keeps per-frame flush/copy work proportional to real writes instead of
    // to the total backing allocation reserved for possible future particle growth.
    std::vector<DirtyRange> m_dirty_ranges;
};

} // namespace vulkan
} // namespace wallpaper
