#include "Scene.h"

#include "Image.hpp"
#include "SceneCamera.h"

#include "Fs/VFS.h"
#include "Interface/IImageParser.h"
#include "Interface/IShaderValueUpdater.h"
#include "Particle/ParticleSystem.h"
#include "Utils/Logging.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <string_view>
#include <unordered_set>

namespace wallpaper 
{

namespace
{
std::size_t EstimateParsedImageBytes(const std::shared_ptr<Image>& image) {
    if (image == nullptr) return 0;

    std::size_t total = 0;
    for (const auto& slot : image->slots) {
        for (const auto& mipmap : slot.mipmaps) {
            if (mipmap.size > 0) {
                total += static_cast<std::size_t>(mipmap.size);
                continue;
            }
            total += static_cast<std::size_t>(std::max(mipmap.width, 0)) *
                     static_cast<std::size_t>(std::max(mipmap.height, 0)) * 4u;
        }
    }
    return total;
}

// GFX_ORTHO18 0x14009a630. Column-major: m00=2/(r-l), m11=2/(t-b),
// m22=-1/(f-n), T=(-(r+l)/(r-l), -(t+b)/(t-b), -(f+n)/(f-n)).
Eigen::Matrix4f GfxOrtho18(float l, float r, float b, float t, float n, float f) {
    if (!(r > l) || !(t > b)) return Eigen::Matrix4f::Identity();
    Eigen::Matrix4f cam = Eigen::Matrix4f::Zero();
    cam(0, 0)           = 2.0f / (r - l);
    cam(1, 1)           = 2.0f / (t - b);
    cam(2, 2)           = -1.0f / (f - n);
    cam(0, 3)           = -(r + l) / (r - l);
    cam(1, 3)           = -(t + b) / (t - b);
    cam(2, 3)           = -(f + n) / (f - n);
    cam(3, 3)           = 1.0f;
    return cam;
}

bool IsLayerVisibleImpl(const Scene& scene, int32_t layer_id) {
    if (layer_id == 0) return true;
    // SceneObject::flags bit0 plus the object parent walk is the only
    // visibility source (0x140185010). Layers with no authored object
    // default to visible.
    if (const auto* object = scene.FindSceneObject(layer_id)) {
        return object->EffectiveVisible();
    }
    return true;
}

Eigen::Vector3d ToVector3d(const std::array<float, 3>& value) {
    return Eigen::Vector3d(value[0], value[1], value[2]);
}

std::array<float, 3> LerpArray3(const std::array<float, 3>& lhs,
                                const std::array<float, 3>& rhs,
                                double ratio) {
    const auto t = static_cast<float>(std::clamp(ratio, 0.0, 1.0));
    return {
        lhs[0] + (rhs[0] - lhs[0]) * t,
        lhs[1] + (rhs[1] - lhs[1]) * t,
        lhs[2] + (rhs[2] - lhs[2]) * t,
    };
}

bool ResolveCameraPathSample(const Scene::CameraPathSegment& segment,
                             double local_time,
                             Scene::CameraPathKeyframe& out) {
    if (segment.keyframes.empty()) return false;
    if (segment.keyframes.size() == 1) {
        out = segment.keyframes.front();
        return true;
    }

    const auto clamped_time = std::clamp(local_time, 0.0, std::max(0.0, segment.duration));
    const auto& first = segment.keyframes.front();
    const auto& last = segment.keyframes.back();
    if (clamped_time <= first.timestamp) {
        out = first;
        return true;
    }
    if (clamped_time >= last.timestamp) {
        out = last;
        return true;
    }

    for (size_t index = 1; index < segment.keyframes.size(); index++) {
        const auto& lhs = segment.keyframes[index - 1];
        const auto& rhs = segment.keyframes[index];
        if (clamped_time > rhs.timestamp) continue;

        const auto span = rhs.timestamp - lhs.timestamp;
        const auto ratio = span > 1e-9 ? (clamped_time - lhs.timestamp) / span : 0.0;
        out.timestamp = clamped_time;
        out.eye = LerpArray3(lhs.eye, rhs.eye, ratio);
        out.center = LerpArray3(lhs.center, rhs.center, ratio);
        out.up = LerpArray3(lhs.up, rhs.up, ratio);
        return true;
    }

    out = last;
    return true;
}

void CollectLayerEffectNodes(const Scene& scene, int32_t layer_id, std::vector<SceneNode*>& nodes) {
    auto camera_names_it = scene.objectRuntimeCameraNames.find(layer_id);
    if (camera_names_it == scene.objectRuntimeCameraNames.end()) return;

    for (const auto& camera_name : camera_names_it->second) {
        auto camera_it = scene.cameras.find(camera_name);
        if (camera_it == scene.cameras.end() || !camera_it->second->HasImgEffect()) continue;

        auto* effect_layer = camera_it->second->GetImgEffect().get();
        if (effect_layer == nullptr) continue;

        if (effect_layer->HasFinalComposite()) {
            // Final composite nodes are owned by the image-effect bridge instead of the authored
            // scene tree, so they will not be reached by normal parent/child propagation. Treat
            // them as layer-owned runtime nodes here to keep layer visibility authoritative while
            // preserving effect-local visibility on the internal shader nodes.
            nodes.push_back(&effect_layer->FinalNode());
        }

        for (size_t effect_index = 0; effect_index < effect_layer->EffectCount(); effect_index++) {
            auto& effect = effect_layer->GetEffect(effect_index);
            for (auto& effect_node : effect->nodes) {
                if (effect_node.sceneNode) nodes.push_back(effect_node.sceneNode.get());
            }
        }
    }
}

void ApplyLayerVisibilityRecursive(Scene& scene, int32_t layer_id, std::unordered_set<int32_t>& visited) {
    if (layer_id == 0 || !visited.insert(layer_id).second) return;

    const bool effective_visible = IsLayerVisibleImpl(scene, layer_id);

    if (auto runtime_nodes_it = scene.objectRuntimeNodes.find(layer_id);
        runtime_nodes_it != scene.objectRuntimeNodes.end()) {
        for (auto* node : runtime_nodes_it->second) {
            if (node != nullptr) {
                // Layer visibility propagation must not overwrite a node's own local visibility
                // contract. Runtime-owned support nodes may intentionally stay hidden even while
                // their authored layer is visible, so the scene system only updates the
                // layer-level flag.
                node->SetLayerVisible(effective_visible);
            }
        }
    }

    std::vector<SceneNode*> effect_nodes;
    CollectLayerEffectNodes(scene, layer_id, effect_nodes);
    for (auto* node : effect_nodes) {
        if (node != nullptr) {
            // Effect nodes are also owned by the layer-visibility system, but they still need to
            // preserve any explicit local visibility decisions that the effect pipeline may make.
            node->SetLayerVisible(effective_visible);
        }
    }

    // Child layers are the SceneObject children; the object tree is the only
    // parent/child source.
    if (const auto* object = scene.FindSceneObject(layer_id)) {
        for (const auto* child : object->children()) {
            if (child != nullptr) ApplyLayerVisibilityRecursive(scene, child->id(), visited);
        }
    }
}

std::pair<int32_t, Scene::CameraLayerRuntimeState*> FindActiveCameraLayer(Scene& scene) {
    // Wallpaper Engine uses the bottom-most visible camera layer as the active view. Scene JSON is
    // parsed in layer order, so walking the recorded camera layer order backwards gives later
    // camera layers precedence while still letting user/script visibility changes disable them.
    for (auto it = scene.cameraLayerOrder.rbegin(); it != scene.cameraLayerOrder.rend(); ++it) {
        auto layer_it = scene.cameraLayers.find(*it);
        if (layer_it == scene.cameraLayers.end() || !layer_it->second.node) continue;
        if (!scene.IsLayerVisible(*it)) continue;
        return { *it, &layer_it->second };
    }
    return { 0, nullptr };
}

constexpr uint64_t kOfficialTextureResolutionAutoArea = 1969920ull;

const char* TextureResolutionRequestedName(int quality) {
    switch (quality) {
    case 1:
        return "half";
    case 2:
        return "auto";
    default:
        return "full";
    }
}

bool TextureResolutionShouldDropMip0(int quality, uint32_t width, uint32_t height) {
    if (quality == 1) return true;
    if (quality != 2) return false;
    // Official Wallpaper Engine 2.8.42 auto: one global bool for the whole
    // wallpaper, not a per-texture 1920x1080 test. Compare output pixel area
    // (floatA * floatB) against 1969920 (1920 * 1080 * 0.95). Below → half.
    const uint64_t area =
        static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    return area < kOfficialTextureResolutionAutoArea;
}

void DropImageMip0(Image& image) {
    for (auto& slot : image.slots) {
        if (slot.mipmaps.size() <= 1) continue;
        slot.mipmaps.erase(slot.mipmaps.begin());
        if (!slot.mipmaps.empty()) {
            slot.width  = slot.mipmaps.front().width;
            slot.height = slot.mipmaps.front().height;
        }
    }
}

double SanitizeCameraZoom(double zoom, int32_t layer_id) {
    if (std::isfinite(zoom) && zoom > 0.0001) return zoom;

    // Invalid authored/user zoom values would collapse the orthographic projection to infinity.
    // Log the offending camera layer and keep a neutral zoom so the wallpaper remains visible.
    LOG_ERROR("SceneCameraLayer: invalid zoom %.6f on layer=%d, using 1.0", zoom, layer_id);
    return 1.0;
}

void ApplyCameraProjectionState(Scene& scene,
                                const std::string& camera_name,
                                SceneCamera& camera,
                                double zoom,
                                float fov,
                                int32_t layer_id) {
    if (camera.IsPerspective()) {
        camera.SetFov(fov);
    } else {
        const double safe_zoom = SanitizeCameraZoom(zoom, layer_id);
        camera.SetWidth(std::max(1.0, static_cast<double>(scene.ortho[0]) / safe_zoom));
        camera.SetHeight(std::max(1.0, static_cast<double>(scene.ortho[1]) / safe_zoom));
    }

    camera.Update();
    scene.UpdateLinkedCamera(camera_name);
}
} // namespace

Scene::Scene(): sceneGraph(std::make_shared<SceneNode>()) ,paritileSys(std::make_unique<ParticleSystem>(*this)) {
    DestStackResetIdentity();
}

void Scene::DestStackResetIdentity() {
    // DEST_IDENTITY_CTOR 0x14017d5a2: *dest = 4x4 identity, pointer stays base.
    m_dest_slots[0] = Eigen::Matrix4f::Identity();
    m_dest_index    = 0;
}

void Scene::DestStackPushCopy() {
    // PATH_B 0x14018b01e: dest pointer += 0x40, copy 0x40 bytes from previous.
    if (m_dest_index + 1 >= kDestStackSlots) return;
    m_dest_slots[m_dest_index + 1] = m_dest_slots[m_dest_index];
    ++m_dest_index;
}

void Scene::DestStackPop() {
    // PATH_B 0x14018b17a: dest pointer -= 0x40.
    if (m_dest_index == 0) return;
    --m_dest_index;
}

void Scene::DestStackApplyPathB(SceneObject& object, float lookat_x, float lookat_y,
                               float amount) {
    const SceneObject* root = object.Root();
    // PATH_B 0x14018b062: walk +0x180 to ROOT; ox/oy from ROOT +0x128/+0x170.
    const float ox =
        (root->origin().x() - lookat_x) * amount * root->parallax_depth().x();
    const float oy =
        (root->origin().y() - lookat_y) * amount * root->parallax_depth().y();
    object.set_leftover_parallax(ox, oy);
    auto& dest = m_dest_slots[m_dest_index];
    // PATH_B 0x14018b118: T += ox·col0 + oy·col1. Dest 3x3 at entry is identity.
    dest.col(3) += ox * dest.col(0) + oy * dest.col(1);
}

const Eigen::Matrix4f& Scene::DestStackTop() const {
    return m_dest_slots[m_dest_index];
}

bool Scene::DestStackAtBase() const {
    return m_dest_index == 0;
}

void Scene::SetWindowSize(int32_t w, int32_t h) {
    m_window_w = w;
    m_window_h = h;
}

Eigen::Matrix4f Scene::FitOrthoCamera() const {
    // VIEW_ORTHO_LR 0x140183b75 / GFX_ORTHO18 0x14009a630. Default cover,
    // pad 0.5, fliph 0. Do not use TREE Ortho() (Vulkan Z remap).
    const float cw = static_cast<float>(ortho[0]);
    const float ch = static_cast<float>(ortho[1]);
    const float ww = static_cast<float>(m_window_w);
    const float wh = static_cast<float>(m_window_h);
    if (!(cw > 0.0f) || !(ch > 0.0f) || !(ww > 0.0f) || !(wh > 0.0f)) {
        return Eigen::Matrix4f::Identity();
    }
    float l = 0.0f;
    float r = cw;
    float b = 0.0f;
    float t = ch;
    if (cw / ch > ww / wh) {
        const float vw = ch * ww / wh;
        l              = (cw - vw) * 0.5f;
        r              = (cw + vw) * 0.5f;
    } else {
        const float vh = cw * wh / ww;
        b              = (ch - vh) * 0.5f;
        t              = (ch + vh) * 0.5f;
    }
    return GfxOrtho18(l, r, b, t, -2000.0f, 2000.0f);
}

Eigen::Matrix4f Scene::DestOrthoCamera(float width, float height) const {
    // DEST_ORTHO_TNF Date leftover: l=0 r=W b=0 t=H n=-1000 f=1000.
    // LIVE_LASTPASS_930 leftover PRE +0x930 is 2/W, 2/H, Tx=Ty=-1.
    if (!(width > 0.0f) || !(height > 0.0f)) return Eigen::Matrix4f::Identity();
    return GfxOrtho18(0.0f, width, 0.0f, height, -1000.0f, 1000.0f);
}

Eigen::Matrix4f Scene::LeftoverDestOrthoMvp(const SceneObject& object) const {
    // IMAGE_2D8_NOFULLFB / EFFECT_FBO_SIZE: named-RT is max(4,AABB). dest=I.
    const int32_t dest_w = static_cast<int32_t>(object.dest_size().x());
    const int32_t dest_h = static_cast<int32_t>(object.dest_size().y());
    return DestOrthoCamera(static_cast<float>(std::max(4, dest_w)),
                           static_cast<float>(std::max(4, dest_h)));
}

Eigen::Matrix4f Scene::LeftoverTextDestOrthoMvp(const SceneObject& object) const {
    const int32_t dest_w = std::max(4, static_cast<int32_t>(object.dest_size().x()));
    const int32_t dest_h = std::max(4, static_cast<int32_t>(object.dest_size().y()));
    Eigen::Matrix4f to_dest_center = Eigen::Matrix4f::Identity();
    to_dest_center(0, 3)           = 0.5f * static_cast<float>(dest_w);
    to_dest_center(1, 3)           = 0.5f * static_cast<float>(dest_h);
    return LeftoverDestOrthoMvp(object) * to_dest_center;
}

void Scene::FlushLastPassMvp() {
    // ENGINE_FLUSH 0x1400d4264: +0x930 = *camera * *dest. camera is
    // LASTPASS_CAM_ORTHO fit-ortho; dest is Path B dest-STACK.
    m_last_pass_mvp = FitOrthoCamera() * DestStackTop();
}

Eigen::Matrix4f Scene::LastPassDrawMvp(SceneObject& object) const {
    // DEST_BLIT 0x1401e9dd5 / LASTPASS_8F0_T: I*=FetchDest, then
    // +0x8f0 = I * +0x930. Date last-pass upload is LastPassMvp
    // (VERTICAL_MVP_ID id 0xb). IMAGE leftover +0x320==0 (IMAGE_VT_F0)
    // and IMAGE last-pass (LASTPASS_IMAGE_ID) upload this +0x8f0
    // stand-in. Do not copy it into +0x930.
    return LastPassMvp() * object.FetchDest();
}

Scene::~Scene() {
    ClearParsedImageCache();
}

std::shared_ptr<Image> Scene::CacheParsedImageResultLocked(
    const std::string& texture_key,
    std::shared_ptr<Image> image,
    std::chrono::steady_clock::time_point started_at,
    const char* success_event,
    const char* failure_event) {
    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - started_at)
                                .count();
    if (image != nullptr) {
        PrepareParsedImageForGpu(*image);
        m_parsed_image_cache[texture_key] = image;
        LOG_INFO("%s: key='%s' bytes=%zu duration=%.2fms",
                 success_event,
                 texture_key.c_str(),
                 EstimateParsedImageBytes(image),
                 static_cast<double>(elapsed_us) / 1000.0);
        return image;
    }

    m_failed_parsed_images.insert(texture_key);
    LOG_ERROR("%s: key='%s' duration=%.2fms",
              failure_event,
              texture_key.c_str(),
              static_cast<double>(elapsed_us) / 1000.0);
    return {};
}

std::shared_ptr<Image> Scene::GetParsedImageIfReady(const std::string& texture_key) {
    if (texture_key.empty()) return {};

    std::lock_guard lock(m_parsed_image_mutex);
    if (const auto cached_it = m_parsed_image_cache.find(texture_key);
        cached_it != m_parsed_image_cache.end()) {
        return cached_it->second;
    }
    if (m_failed_parsed_images.count(texture_key) != 0) return {};

    const auto pending_it = m_pending_parsed_images.find(texture_key);
    if (pending_it == m_pending_parsed_images.end()) return {};
    if (pending_it->second.future.wait_for(std::chrono::milliseconds(0)) !=
        std::future_status::ready) {
        return {};
    }

    const auto started_at = pending_it->second.started_at;
    auto       image      = pending_it->second.future.get();
    m_pending_parsed_images.erase(pending_it);

    return CacheParsedImageResultLocked(texture_key,
                                        std::move(image),
                                        started_at,
                                        "SceneImageAsyncParseComplete",
                                        "SceneImageAsyncParseFailed");
}

std::shared_ptr<Image> Scene::ParseImageBlockingCached(const std::string& texture_key) {
    if (texture_key.empty() || imageParser == nullptr) return {};

    {
        std::lock_guard lock(m_parsed_image_mutex);
        if (const auto cached_it = m_parsed_image_cache.find(texture_key);
            cached_it != m_parsed_image_cache.end()) {
            return cached_it->second;
        }
        if (m_failed_parsed_images.count(texture_key) != 0) return {};

        const auto pending_it = m_pending_parsed_images.find(texture_key);
        if (pending_it != m_pending_parsed_images.end()) {
            const auto started_at = pending_it->second.started_at;
            auto       image      = pending_it->second.future.get();
            m_pending_parsed_images.erase(pending_it);
            return CacheParsedImageResultLocked(texture_key,
                                                std::move(image),
                                                started_at,
                                                "SceneImageAsyncParseJoined",
                                                "SceneImageAsyncParseFailed");
        }
    }

    const auto started_at = std::chrono::steady_clock::now();
    auto       image      = imageParser->Parse(texture_key);

    std::lock_guard lock(m_parsed_image_mutex);
    return CacheParsedImageResultLocked(texture_key,
                                        std::move(image),
                                        started_at,
                                        "SceneImageParseBlocking",
                                        "SceneImageParseBlockingFailed");
}

Scene::ParsedImageRequest Scene::RequestParsedImageAsync(const std::string& texture_key) {
    if (texture_key.empty() || imageParser == nullptr) {
        return { ParsedImageRequestState::Failed, {} };
    }

    if (auto image = GetParsedImageIfReady(texture_key); image != nullptr) {
        return { ParsedImageRequestState::Ready, image };
    }

    {
        std::lock_guard lock(m_parsed_image_mutex);
        if (const auto cached_it = m_parsed_image_cache.find(texture_key);
            cached_it != m_parsed_image_cache.end()) {
            return { ParsedImageRequestState::Ready, cached_it->second };
        }
        if (m_failed_parsed_images.count(texture_key) != 0) {
            return { ParsedImageRequestState::Failed, {} };
        }
        if (m_pending_parsed_images.count(texture_key) != 0) {
            return { ParsedImageRequestState::Pending, {} };
        }

        auto*       parser   = imageParser.get();
        std::string key_copy = texture_key;
        PendingParsedImageRequest pending;
        pending.started_at = std::chrono::steady_clock::now();
        pending.future     = std::async(std::launch::async, [parser, key_copy]() {
            return parser != nullptr ? parser->Parse(key_copy) : std::shared_ptr<Image> {};
        });
        m_pending_parsed_images.emplace(texture_key, std::move(pending));
    }

    LOG_INFO("SceneImageAsyncParseQueued: key='%s'", texture_key.c_str());
    return { ParsedImageRequestState::Pending, {} };
}

void Scene::DropParsedImageCache(std::string_view texture_key) {
    if (texture_key.empty()) return;

    const std::string key(texture_key);
    std::future<std::shared_ptr<Image>> pending_future;
    std::size_t dropped_bytes = 0;
    bool dropped_cached_image = false;
    bool dropped_pending_parse = false;
    {
        std::lock_guard lock(m_parsed_image_mutex);
        if (auto cached_it = m_parsed_image_cache.find(key);
            cached_it != m_parsed_image_cache.end()) {
            dropped_bytes = EstimateParsedImageBytes(cached_it->second);
            m_parsed_image_cache.erase(cached_it);
            dropped_cached_image = true;
        }
        if (auto pending_it = m_pending_parsed_images.find(key);
            pending_it != m_pending_parsed_images.end()) {
            pending_future = std::move(pending_it->second.future);
            m_pending_parsed_images.erase(pending_it);
            dropped_pending_parse = true;
        }
        m_failed_parsed_images.erase(key);
    }
    if (dropped_cached_image || dropped_pending_parse) {
        LOG_INFO("SceneImageCacheDrop: key='%s' cached=%s bytes=%zu pending=%s",
                 key.c_str(),
                 dropped_cached_image ? "true" : "false",
                 dropped_bytes,
                 dropped_pending_parse ? "true" : "false");
    }
    if (pending_future.valid()) pending_future.wait();
}

void Scene::ClearParsedImageCache() {
    std::vector<std::future<std::shared_ptr<Image>>> pending_futures;
    {
        std::lock_guard lock(m_parsed_image_mutex);
        pending_futures.reserve(m_pending_parsed_images.size());
        for (auto& [_, request] : m_pending_parsed_images) {
            if (request.future.valid()) pending_futures.emplace_back(std::move(request.future));
        }
        m_parsed_image_cache.clear();
        m_pending_parsed_images.clear();
        m_failed_parsed_images.clear();
    }

    if (!pending_futures.empty()) {
        LOG_INFO("SceneImageAsyncParseJoin: pending=%zu", pending_futures.size());
        for (auto& future : pending_futures) {
            if (future.valid()) future.wait();
        }
    }
}

void Scene::ApplyTextureResolution(int quality, uint32_t output_width, uint32_t output_height) {
    quality = std::clamp(quality, 0, 2);
    const bool next_drop = TextureResolutionShouldDropMip0(quality, output_width, output_height);
    const bool quality_changed = textureResolution.quality != quality;
    const bool drop_changed    = textureResolution.drop_mip0 != next_drop;
    const bool first_apply     = textureResolution.output_width == 0 && output_width != 0;

    textureResolution.quality       = quality;
    textureResolution.drop_mip0     = next_drop;
    textureResolution.output_width  = output_width;
    textureResolution.output_height = output_height;

    if (quality_changed || drop_changed || first_apply) {
        const uint64_t area =
            static_cast<uint64_t>(output_width) * static_cast<uint64_t>(output_height);
        LOG_INFO("texture-resolution requested=%s drop-mip0=%s output=%ux%u area=%llu",
                 TextureResolutionRequestedName(quality),
                 next_drop ? "true" : "false",
                 output_width,
                 output_height,
                 static_cast<unsigned long long>(area));
    }

    if (!drop_changed) return;

    textureResolution.epoch++;
    for (auto& [key, texture] : textures) {
        texture.gpuWidth  = 0;
        texture.gpuHeight = 0;
        if (texture.isVideo || key.empty()) continue;
        // 1-mip / video / synthetic images cannot drop mip0. Unknown mip
        // counts are refreshed so a later parse can apply the new policy.
        if (texture.mipmapCount == 1) continue;
        dirtyImportedTextureKeys.insert(key);
    }
    ClearParsedImageCache();
    MarkRenderGraphResourcesDirty();
}

void Scene::ApplyTextureResolutionForCurrentOutput() {
    if (physicalOutputExtent[0] == 0 || physicalOutputExtent[1] == 0) return;
    ApplyTextureResolution(textureResolution.quality,
                           physicalOutputExtent[0],
                           physicalOutputExtent[1]);
}

void Scene::PrepareParsedImageForGpu(Image& image) {
    image.textureResolutionEpoch = textureResolution.epoch;
    const auto texture_it = textures.find(image.key);
    const bool video =
        image.header.isVideoTexture ||
        (texture_it != textures.end() && texture_it->second.isVideo);
    if (!video && textureResolution.drop_mip0) {
        DropImageMip0(image);
    }
    if (texture_it == textures.end() || image.slots.empty()) return;
    texture_it->second.gpuWidth  = image.slots[0].width;
    texture_it->second.gpuHeight = image.slots[0].height;
}

std::array<i32, 4>
Scene::EffectiveImportedTextureResolution(const SceneTexture& texture) const {
    const bool drop = textureResolution.drop_mip0 && !texture.isVideo &&
                      texture.mipmapCount > 1;
    if (texture.gpuWidth > 0 && texture.gpuHeight > 0) {
        // Official bind path: g_TextureNResolution follows the uploaded GPU
        // extent, not the authored .tex header, once mip0 has been dropped.
        if (!drop) {
            if (texture.mipmap_larger) {
                return { texture.width, texture.height, texture.mapWidth, texture.mapHeight };
            }
            return { texture.mapWidth, texture.mapHeight, texture.mapWidth, texture.mapHeight };
        }
        if (texture.mipmap_larger) {
            return { texture.gpuWidth,
                     texture.gpuHeight,
                     std::max<i32>(1, texture.mapWidth / 2),
                     std::max<i32>(1, texture.mapHeight / 2) };
        }
        return { texture.gpuWidth, texture.gpuHeight, texture.gpuWidth, texture.gpuHeight };
    }
    if (!drop) {
        if (texture.mipmap_larger) {
            return { texture.width, texture.height, texture.mapWidth, texture.mapHeight };
        }
        return { texture.mapWidth, texture.mapHeight, texture.mapWidth, texture.mapHeight };
    }
    if (texture.mipmap_larger) {
        return { std::max<i32>(1, texture.width / 2),
                 std::max<i32>(1, texture.height / 2),
                 std::max<i32>(1, texture.mapWidth / 2),
                 std::max<i32>(1, texture.mapHeight / 2) };
    }
    const i32 half_w = std::max<i32>(1, texture.mapWidth / 2);
    const i32 half_h = std::max<i32>(1, texture.mapHeight / 2);
    return { half_w, half_h, half_w, half_h };
}

void Scene::SetLayerParentBinding(int32_t layer_id, int32_t parent_id, std::string attachment) {
    if (layer_id == 0) return;
    if (parent_id == 0 && attachment.empty()) {
        layerParentBindings.erase(layer_id);
        if (auto* object = FindSceneObject(layer_id)) object->set_parent(nullptr, -1);
        return;
    }
    layerParentBindings[layer_id] = LayerParentBinding {
        .parent_id = parent_id,
        .attachment = std::move(attachment),
    };
    BindSceneObjectParent(layer_id, parent_id, layerParentBindings[layer_id].attachment);
}

Scene::LayerParentBinding Scene::GetLayerParentBinding(int32_t layer_id) const {
    auto it = layerParentBindings.find(layer_id);
    return it == layerParentBindings.end() ? LayerParentBinding {} : it->second;
}

void Scene::ClearLayerParentBinding(int32_t layer_id) {
    layerParentBindings.erase(layer_id);
}

std::vector<int32_t> Scene::GetLayerChildren(int32_t layer_id) const {
    if (const auto* object = FindSceneObject(layer_id); object != nullptr) {
        std::vector<int32_t> children;
        children.reserve(object->children().size());
        for (auto* child : object->children()) {
            if (child != nullptr) children.push_back(child->id());
        }
        if (! children.empty()) return children;
    }
    std::vector<int32_t> children;
    for (const auto& [child_id, binding] : layerParentBindings) {
        if (binding.parent_id == layer_id) children.push_back(child_id);
    }
    return children;
}

SceneObject* Scene::FindSceneObject(int32_t layer_id) {
    auto it = sceneObjects.find(layer_id);
    return it == sceneObjects.end() ? nullptr : it->second.get();
}

const SceneObject* Scene::FindSceneObject(int32_t layer_id) const {
    auto it = sceneObjects.find(layer_id);
    return it == sceneObjects.end() ? nullptr : it->second.get();
}

SceneObject* Scene::FindSceneObjectForNode(const SceneNode* node) {
    return const_cast<SceneObject*>(
        static_cast<const Scene*>(this)->FindSceneObjectForNode(node));
}

const SceneObject* Scene::FindSceneObjectForNode(const SceneNode* node) const {
    if (node == nullptr) return nullptr;
    auto owner = nodeOwners.find(const_cast<SceneNode*>(node));
    // owner 0 is bloom/post handles, not an objects[] SceneObject (SO1 0x140190837).
    if (owner != nodeOwners.end() && owner->second != 0) return FindSceneObject(owner->second);
    const int32_t node_id = const_cast<SceneNode*>(node)->ID();
    if (node_id != 0 && sceneObjects.count(node_id) != 0) return FindSceneObject(node_id);
    return nullptr;
}

SceneObject& Scene::EnsureSceneObject(int32_t layer_id) {
    if (auto* existing = FindSceneObject(layer_id)) return *existing;
    auto object = std::make_unique<SceneObject>(this, layer_id);
    auto* raw   = object.get();
    sceneObjects.emplace(layer_id, std::move(object));
    objectList.push_back(raw);
    return *raw;
}

void Scene::BindSceneObjectParent(int32_t layer_id, int32_t parent_id,
                                  std::string_view attachment) {
    auto* child = FindSceneObject(layer_id);
    if (child == nullptr) return;
    auto* parent = parent_id != 0 ? FindSceneObject(parent_id) : nullptr;
    // 0x1401de931 / 0x14018802c: +0x180 parent*, +0x190 attach index or -1,
    // r8 adjustTransforms=0 so 0x1401de928 skips attach-zero 0x1401de962.
    child->set_parent(parent, attachment.empty() ? -1 : 0);
}

void Scene::SetLayerLocalVisibility(int32_t layer_id, bool visible) {
    if (layer_id == 0) return;

    // The SceneObject is the only owner of layer visibility. Scripts can set
    // visibility on a layer before its runtime nodes materialize, so create
    // the object when it does not exist yet.
    EnsureSceneObject(layer_id).set_local_visible(visible);
}

bool Scene::GetLayerLocalVisibility(int32_t layer_id) const {
    const auto* object = FindSceneObject(layer_id);
    return object == nullptr ? true : object->local_visible();
}

bool Scene::IsLayerVisible(int32_t layer_id) const {
    return IsLayerVisibleImpl(*this, layer_id);
}

void Scene::ApplyLayerVisibility(int32_t layer_id) {
    std::unordered_set<int32_t> visited;
    ApplyLayerVisibilityRecursive(*this, layer_id, visited);
    if (!cameraLayers.empty()) UpdateActiveCameraLayer();
}

void Scene::ApplyAllLayerVisibility() {
    std::unordered_set<int32_t> visited;
    for (const auto layer_id : layerOrder) {
        ApplyLayerVisibilityRecursive(*this, layer_id, visited);
    }
    for (const auto& [layer_id, _] : layerNodes) {
        ApplyLayerVisibilityRecursive(*this, layer_id, visited);
    }
    if (!cameraLayers.empty()) UpdateActiveCameraLayer();
}

void Scene::UpdateModelCameraPath() {
    if (!modelCameraPathEnabled || modelCameraPathSegments.empty() ||
        modelPerspectiveCameraName.empty()) {
        return;
    }

    auto camera_it = cameras.find(modelPerspectiveCameraName);
    if (camera_it == cameras.end() || !camera_it->second) return;

    double total_duration = 0.0;
    for (const auto& segment : modelCameraPathSegments) {
        total_duration += std::max(0.0, segment.duration);
    }
    if (total_duration <= 1e-9) return;

    double path_time = std::fmod(std::max(0.0, elapsingTime), total_duration);
    if (path_time < 0.0) path_time += total_duration;

    int32_t active_segment = -1;
    double local_time = path_time;
    for (size_t index = 0; index < modelCameraPathSegments.size(); index++) {
        const auto duration = std::max(0.0, modelCameraPathSegments[index].duration);
        if (local_time <= duration || index + 1 == modelCameraPathSegments.size()) {
            active_segment = static_cast<int32_t>(index);
            break;
        }
        local_time -= duration;
    }
    if (active_segment < 0 ||
        active_segment >= static_cast<int32_t>(modelCameraPathSegments.size())) {
        return;
    }

    Scene::CameraPathKeyframe sample;
    if (!ResolveCameraPathSample(modelCameraPathSegments[active_segment], local_time, sample)) {
        return;
    }

    // Camera path playback is bound to the model-only camera name installed by WPModelObject
    // parsing. This deliberately avoids `global_perspective`, which is a legacy 2D particle camera.
    camera_it->second->SetExplicitView(ToVector3d(sample.eye),
                                       ToVector3d(sample.center),
                                       ToVector3d(sample.up));
    UpdateLinkedCamera(modelPerspectiveCameraName);

    if (activeModelCameraPathSegment != active_segment) {
        const auto& segment = modelCameraPathSegments[active_segment];
        LOG_INFO("Scene3DModelCameraPathActive: previous=%d active=%d duration=%.3f "
                 "local-time=%.3f eye=[%.3f, %.3f, %.3f] center=[%.3f, %.3f, %.3f]",
                 activeModelCameraPathSegment,
                 active_segment,
                 segment.duration,
                 local_time,
                 sample.eye[0],
                 sample.eye[1],
                 sample.eye[2],
                 sample.center[0],
                 sample.center[1],
                 sample.center[2]);
        activeModelCameraPathSegment = active_segment;
    }
}

namespace
{

// The same translation is added to eye and center so look direction stays unchanged.
constexpr float kCameraShakeYFrequency     = 1.3329999446868896f;
constexpr float kCameraShakeAmplitudeScale = 0.1f;
constexpr float kCameraShakeRoughnessPow   = 3.0f;
constexpr float kCameraShakeRoughnessEps   = 0.001f;

Eigen::Vector3d ComputeSceneCameraShakeOffset(bool enabled, bool orthographic, float amplitude,
                                              float roughness, float speed, double time_seconds,
                                              int32_t ortho_height) {
    if (!enabled) return Eigen::Vector3d::Zero();

    const float p     = std::pow(roughness, kCameraShakeRoughnessPow);
    const float t     = speed * speed * static_cast<float>(time_seconds);
    float       x     = std::sin(t);
    float       y     = std::cos(t * kCameraShakeYFrequency);
    float       z     = std::cos(t);
    float       scale = amplitude * kCameraShakeAmplitudeScale;
    if (orthographic) {
        z = 0.0f;
        scale *= static_cast<float>(ortho_height) * kCameraShakeAmplitudeScale;
    }
    if (p > kCameraShakeRoughnessEps && p != 1.0f) {
        const float len2 = x * x + y * y + z * z;
        if (len2 > 0.0f) {
            const float len = std::sqrt(len2);
            const float n   = std::pow(len, p) / len;
            x *= n;
            y *= n;
            z *= n;
        }
    }
    return Eigen::Vector3d(static_cast<double>(x * scale),
                           static_cast<double>(y * scale),
                           static_cast<double>(z * scale));
}

void ApplyCameraShakeOffset(Scene& scene, std::string_view camera_name,
                            const Eigen::Vector3d& offset) {
    auto camera_it = scene.cameras.find(std::string(camera_name));
    if (camera_it == scene.cameras.end() || !camera_it->second) return;
    camera_it->second->SetShakeOffset(offset);
    camera_it->second->Update();
}

} // namespace

void Scene::UpdateCameraShake() {
    const Eigen::Vector3d offset =
        ComputeSceneCameraShakeOffset(cameraShake,
                                      cameraOrthographic,
                                      cameraShakeAmplitude,
                                      cameraShakeRoughness,
                                      cameraShakeSpeed,
                                      elapsingTime,
                                      ortho[1]);
    ApplyCameraShakeOffset(*this, "global", offset);
    ApplyCameraShakeOffset(*this, "global_perspective", offset);
    if (activeCamera != nullptr) {
        activeCamera->SetShakeOffset(offset);
        activeCamera->Update();
    }
    if (!cameraOrthographic && !modelPerspectiveCameraName.empty()) {
        ApplyCameraShakeOffset(*this, modelPerspectiveCameraName, offset);
    }
}

Eigen::Vector3f Scene::ResolveCameraLayerNodeTranslation(
    const std::array<float, 3>& authored_origin) const {
    // WE 2D camera origins are authored around the static camera origin, where 0/0 means the
    // default centered wallpaper view. Hanabi's orthographic camera node is centered in render
    // coordinates, so add the canvas half-size before attaching the SceneCamera to this layer.
    return Eigen::Vector3f {
        static_cast<float>(ortho[0]) * 0.5f + authored_origin[0],
        static_cast<float>(ortho[1]) * 0.5f + authored_origin[1],
        authored_origin[2],
    };
}

void Scene::UpdateActiveCameraLayer() {
    auto [next_layer_id, camera_layer] = FindActiveCameraLayer(*this);

    std::string camera_name = "global";
    std::shared_ptr<SceneNode> camera_node = defaultGlobalCameraNode;
    double zoom = defaultGlobalCameraZoom;
    float fov = 50.0f;

    if (camera_layer != nullptr) {
        camera_name = camera_layer->camera_name.empty() ? "global" : camera_layer->camera_name;
        camera_node = camera_layer->node;
        zoom = camera_layer->zoom;
        fov = camera_layer->fov;
    }

    auto camera_it = cameras.find(camera_name);
    if (camera_it == cameras.end() || !camera_it->second) {
        LOG_ERROR("SceneCameraLayer: target camera '%s' for layer=%d is missing",
                  camera_name.c_str(),
                  next_layer_id);
        camera_it = cameras.find("global");
        camera_name = "global";
    }
    if (camera_it == cameras.end() || !camera_it->second || !camera_node) return;

    camera_it->second->AttatchNode(camera_node);
    ApplyCameraProjectionState(*this,
                               camera_name,
                               *camera_it->second,
                               zoom,
                               fov,
                               next_layer_id);
    activeCamera = camera_it->second.get();

    if (activeCameraLayerId != next_layer_id) {
        // This transition log is intentionally sparse: it proves which authored camera layer owns
        // the view without flooding frame logs while keyframed zoom/origin values animate.
        LOG_INFO("SceneCameraLayerActive: previous=%d active=%d camera='%s' zoom=%.3f origin=[%.3f, %.3f, %.3f]",
                 activeCameraLayerId,
                 next_layer_id,
                 camera_name.c_str(),
                 zoom,
                 camera_layer != nullptr ? camera_layer->origin[0] : 0.0f,
                 camera_layer != nullptr ? camera_layer->origin[1] : 0.0f,
                 camera_layer != nullptr ? camera_layer->origin[2] : 0.0f);
        activeCameraLayerId = next_layer_id;
    }
}

SceneImageEffect* Scene::FindImageEffect(int32_t owner_layer_id, uint32_t effect_index) {
    auto camera_names_it = objectRuntimeCameraNames.find(owner_layer_id);
    if (camera_names_it == objectRuntimeCameraNames.end()) return nullptr;

    for (const auto& camera_name : camera_names_it->second) {
        auto camera_it = cameras.find(camera_name);
        if (camera_it == cameras.end() || !camera_it->second->HasImgEffect()) continue;

        auto* effect_layer = camera_it->second->GetImgEffect().get();
        if (effect_layer == nullptr || effect_index >= effect_layer->EffectCount()) continue;
        return effect_layer->GetEffect(effect_index).get();
    }

    return nullptr;
}

const SceneImageEffect* Scene::FindImageEffect(int32_t owner_layer_id,
                                               uint32_t effect_index) const {
    return const_cast<Scene*>(this)->FindImageEffect(owner_layer_id, effect_index);
}

SceneImageEffect* Scene::FindImageEffectById(int32_t owner_layer_id, int32_t effect_id) {
    auto camera_names_it = objectRuntimeCameraNames.find(owner_layer_id);
    if (camera_names_it == objectRuntimeCameraNames.end()) return nullptr;

    for (const auto& camera_name : camera_names_it->second) {
        auto camera_it = cameras.find(camera_name);
        if (camera_it == cameras.end() || !camera_it->second->HasImgEffect()) continue;

        auto* effect_layer = camera_it->second->GetImgEffect().get();
        if (effect_layer == nullptr) continue;

        for (std::size_t effect_index = 0; effect_index < effect_layer->EffectCount();
             effect_index++) {
            auto& effect = effect_layer->GetEffect(effect_index);
            if (effect != nullptr && effect->EffectId() == effect_id) return effect.get();
        }
    }

    return nullptr;
}

const SceneImageEffect* Scene::FindImageEffectById(int32_t owner_layer_id,
                                                   int32_t effect_id) const {
    return const_cast<Scene*>(this)->FindImageEffectById(owner_layer_id, effect_id);
}

bool Scene::SetEffectLocalVisibility(int32_t owner_layer_id, uint32_t effect_index,
                                     bool visible) {
    auto* effect = FindImageEffect(owner_layer_id, effect_index);
    if (effect == nullptr) return false;

    // Only the effect-local bit changes here. The render graph topology remains valid because
    // hidden effects are handled by conditional execution and a bypass copy, while layer visibility
    // propagation still owns parent/child effective visibility.
    effect->SetLocalVisible(visible);
    ApplyLayerVisibility(owner_layer_id);
    return true;
}

bool Scene::SetEffectLocalVisibilityById(int32_t owner_layer_id, int32_t effect_id,
                                         bool visible) {
    auto* effect = FindImageEffectById(owner_layer_id, effect_id);
    if (effect == nullptr) return false;

    effect->SetLocalVisible(visible);
    ApplyLayerVisibility(owner_layer_id);
    return true;
}

}
