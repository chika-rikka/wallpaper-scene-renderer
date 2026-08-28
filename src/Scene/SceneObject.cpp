#include "SceneObject.h"

#include "Scene.h"
#include "SceneMaterial.h"
#include "SceneMesh.h"
#include "SceneNode.h"
#include "SpecTexs.hpp"
#include "WPPuppet.hpp"

#include <algorithm>
#include <array>
#include <unordered_set>

using namespace wallpaper;
using namespace Eigen;

SceneObject::SceneObject(Scene* scene, int32_t id): m_scene(scene), m_id(id) {
    RebuildBasisFromAngles();
}

SceneObject::~SceneObject() = default;

WPPuppetLayer* SceneObject::puppet_layer() {
    return m_puppet_layer.get();
}

const WPPuppetLayer* SceneObject::puppet_layer() const {
    return m_puppet_layer.get();
}

void SceneObject::set_origin(const Vector3f& value) {
    m_origin = value;
    MarkDestDirty();
}

void SceneObject::set_scale(const Vector3f& value) {
    m_scale = value;
    MarkDestDirty();
}

void SceneObject::set_angles(const Vector3f& value) {
    m_angles = value;
    RebuildBasisFromAngles();
    MarkDestDirty();
}

void SceneObject::set_parallax_depth(const Vector2f& value) {
    m_parallax_depth = value;
}

void SceneObject::set_local_visible(bool visible) {
    if (visible) m_flags |= kFlagVisible;
    else m_flags &= ~kFlagVisible;
}

bool SceneObject::EffectiveVisible() const {
    std::unordered_set<const SceneObject*> visiting;
    const SceneObject* cur = this;
    while (cur != nullptr) {
        if (! visiting.insert(cur).second) return true;
        if (! cur->local_visible()) return false;
        cur = cur->parent();
    }
    return true;
}

void SceneObject::set_parent(SceneObject* parent, int32_t attach_index) {
    if (m_parent == parent && m_attach_index == attach_index) return;
    if (m_parent != nullptr) {
        auto& siblings = m_parent->m_children;
        siblings.erase(std::remove(siblings.begin(), siblings.end(), this), siblings.end());
    }
    m_parent       = parent;
    m_attach_index = attach_index;
    if (m_parent != nullptr) m_parent->m_children.push_back(this);
    // 0x1401de92a zeros +0xd0 on setParent
    MarkDestDirty();
}

SceneObject* SceneObject::Root() {
    std::unordered_set<SceneObject*> visiting;
    SceneObject* cur = this;
    while (cur->m_parent != nullptr) {
        if (! visiting.insert(cur).second) break;
        cur = cur->m_parent;
    }
    return cur;
}

const SceneObject* SceneObject::Root() const {
    return const_cast<SceneObject*>(this)->Root();
}

void SceneObject::ApplyAttachZeroOrigin() {
    m_angles = Vector3f::Zero();
    m_origin = Vector3f::Zero();
    RebuildBasisFromAngles();
    MarkDestDirty();
}

void SceneObject::SetAttachBinding(uint32_t bone_index, const Affine3f& bind) {
    m_attach_bone = bone_index;
    m_attach_bind = bind;
    if (m_attach_index < 0) m_attach_index = 0;
    MarkDestDirty();
}

void SceneObject::MarkDestDirty() {
    // 0x1401de92a / 0x1401dd652 / 0x1401f3245: +0xd0 = 0
    m_dest_stamp = 0;
}

bool SceneObject::AncestorDestCacheValid(int32_t incoming_attach) const {
    // 0x140185040. First call from FetchDest passes the child's +0x190 in edx
    // so an unattached child does not apply the scene-stamp compare to an
    // attached ancestor. Recurse replaces edx with this object's +0x190.
    if (m_dest_stamp == 0) return false;
    if (incoming_attach >= 0) {
        const uint32_t scene_stamp = m_scene != nullptr ? m_scene->destStamp : 0;
        if (m_dest_stamp != scene_stamp) return false;
    }
    if (m_parent == nullptr) return true;
    if (m_dest_stamp < m_parent->m_dest_stamp) return false;
    return m_parent->AncestorDestCacheValid(m_attach_index);
}

void SceneObject::set_puppet_layer(const WPPuppetLayer& layer) {
    m_puppet_layer = std::make_unique<WPPuppetLayer>(layer);
    ApplyPuppet304Bit4();
}

void SceneObject::CaptureImage490Mesh(const SceneMesh& mesh) {
    // PUPPET_490 0x14020b15e: gfx vt+0x40 uploads puppet+0x18 CPU verts
    // into +0x490. MESH_FACTORY does not rewrite those floats.
    m_image_490_mesh = std::make_unique<SceneMesh>();
    m_image_490_mesh->ChangeMeshDataFrom(mesh);
}

SceneNode* SceneObject::EnsureLeftoverMvpNode(SceneNode* leftover_owner) {
    if (m_leftover_mvp_node != nullptr) return m_leftover_mvp_node.get();
    // IMAGE_VT_F8_PUPPET Draws +0x490. IMAGE_VT_F8 0x1402090fd no-puppet
    // Draws +0x2e8. Puppet with no +0x490 is 0x1402090d1 ret (no Draw).
    const SceneMesh* src = m_image_490_mesh.get();
    if (src == nullptr) {
        if (m_puppet_layer != nullptr && m_puppet_layer->hasPuppet()) return nullptr;
        src = m_lastpass_mesh.get();
    }
    if (src == nullptr || leftover_owner == nullptr || leftover_owner->Mesh() == nullptr ||
        leftover_owner->Mesh()->Material() == nullptr) {
        return nullptr;
    }
    auto mesh = std::make_shared<SceneMesh>();
    mesh->ChangeMeshDataFrom(*src);
    mesh->AddMaterial(SceneMaterial(*leftover_owner->Mesh()->Material()));
    m_leftover_mvp_node = std::make_unique<SceneNode>();
    m_leftover_mvp_node->AddMesh(mesh);
    m_leftover_mvp_node->SetName(leftover_owner->Name());
    m_leftover_mvp_node->ID() = leftover_owner->ID();
    return m_leftover_mvp_node.get();
}

void SceneObject::ApplyParse304Bit4(int32_t color_blend_mode) {
    // PARSE_304_BIT4 0x1401e6fa2: +0x32c!=0 && !=0x1f ors +0x304 bit4.
    // colorBlendMode 0x1f stays additive (IMAGE_DEST_BLEND), not bit4.
    if (color_blend_mode != 0 && color_blend_mode != 0x1f) m_flag304_bit4 = true;
}

void SceneObject::ApplyPuppet304Bit4() {
    // PUPPET_304_BIT4 0x14020af31 / 0x14020b20d: leftover-card path skipped
    // when +0x320>0 (xor r13b), then [+0x4b8+8]!=0 ors bit4.
    if (m_effect_count > 0 && m_puppet_layer != nullptr && m_puppet_layer->hasPuppet()) {
        m_flag304_bit4 = true;
    }
}

void SceneObject::RebuildBasisFromAngles() {
    // 0x1401df110: cz/sz +0x148, cy/sy +0x144, cx/sx +0x140 → +0x14c…+0x16c
    Affine3f trans = Affine3f::Identity();
    trans.prerotate(AngleAxisf(m_angles.x(), Vector3f::UnitX()));
    trans.prerotate(AngleAxisf(m_angles.y(), Vector3f::UnitY()));
    trans.prerotate(AngleAxisf(m_angles.z(), Vector3f::UnitZ()));
    m_basis = trans.linear();
}

Matrix4f SceneObject::ComposeLocalDest() const {
    // 0x140185150: +0x110 ← +0x128 raw; 3×3 columns = scale * basis
    Matrix4f dest = Matrix4f::Identity();
    dest.block<3, 3>(0, 0) = m_basis * m_scale.asDiagonal();
    dest(0, 3)             = m_origin.x();
    dest(1, 3)             = m_origin.y();
    dest(2, 3)             = m_origin.z();
    return dest;
}

const Matrix4f& SceneObject::FetchDest() {
    // 0x1401850a9: +0xd0 == 0 recomputes. 0x1401850b8 js skips scene+0x144
    // when +0x190 < 0. 0x1401850e7 walks 0x140185040. 0x14018512e copies
    // [scene+0x144] as-is (ctor 0x1401ddc10 zeros +0xd0; no destStamp++).
    const uint32_t scene_stamp = m_scene != nullptr ? m_scene->destStamp : 0;
    if (m_dest_stamp != 0) {
        const bool scene_ok = m_attach_index < 0 || m_dest_stamp == scene_stamp;
        if (scene_ok) {
            if (m_parent == nullptr) return m_dest;
            if (m_dest_stamp >= m_parent->m_dest_stamp &&
                m_parent->AncestorDestCacheValid(m_attach_index)) {
                return m_dest;
            }
        }
    }

    m_dest_stamp = scene_stamp;
    m_dest       = ComposeLocalDest();
    if (m_parent != nullptr) {
        if (m_attach_index >= 0) {
            // 0x140185296: +0x190 < 0 skips vt+0x78. Factory image 0x1401902b6
            // uses vt 0x1404911a8: vt+0x70 0x1401fd510 (MDAT name → index),
            // vt+0x78 0x1401fd5c0: dest = (live_bone * bind) * dest_local.
            // live_bone is helper+0x2c8 (64-byte stride). Model class uses the
            // same multiply at 0x140224970 from parent+0x2d8.
            auto* parent_puppet = m_parent->puppet_layer();
            if (parent_puppet != nullptr && parent_puppet->hasPuppet()) {
                const auto* puppet = parent_puppet->Puppet();
                if (puppet != nullptr) {
                    const Affine3f bone_bind =
                        puppet->BoneModelTransform(m_attach_bone) * m_attach_bind;
                    m_dest = bone_bind.matrix() * m_dest;
                }
            }
        }
        // 0x14005ecb0: dest = parent_dest * dest
        m_dest = m_parent->FetchDest() * m_dest;
    }
    return m_dest;
}

void SceneObject::set_dest_size(float width, float height) {
    // IMAGE_SIZE_PROP 0x1401a4046 / TEXT_2F0 0x140258916.
    m_dest_size = Vector2f(width, height);
}

void SceneObject::set_leftover_uv(float u, float v) {
    // IMAGE_VT_128 0x140209290 / 0x1402092ac.
    m_leftover_uv = Vector2f(u, v);
    if (m_leftover_mesh != nullptr) PublishDestDrawMeshes();
}

namespace
{

void FillCenterFlagsCard(SceneMesh& mesh, float width, float height, uint32_t flags,
                         float uv_u = 1.0f, float uv_v = 1.0f) {
    // CENTER_FLAGS 0x1401ede30: bit0 set → ±extent/2; bit0 clear → 0..extent
    // (IMAGE_2D8_NOFULLFB 0x1401eb19f flags=0). DEST_DRAW_VERTS 0x1401edf4d:
    // (xmm9,xmm12,0), (xmm11,xmm12,0), (xmm9,xmm10,0), (xmm11,xmm10,0).
    // Leftover UV max is IMAGE_VT_128 mapRate, not 1,1.
    float left   = 0.0f;
    float right  = width;
    float bottom = 0.0f;
    float top    = height;
    if ((flags & 1u) != 0) {
        const float half_w = width * 0.5f;
        const float half_h = height * 0.5f;
        left               = -half_w;
        right              = half_w;
        bottom             = -half_h;
        top                = half_h;
    }
    const std::array<float, 12> pos {
        left, top, 0.0f, right, top, 0.0f, left, bottom, 0.0f, right, bottom, 0.0f,
    };
    const std::array<float, 8> texCoord {
        0.0f, 0.0f, uv_u, 0.0f, 0.0f, uv_v, uv_u, uv_v,
    };
    SceneVertexArray vertex(
        {
            { std::string(WE_IN_POSITION), VertexType::FLOAT3 },
            { std::string(WE_IN_TEXCOORD), VertexType::FLOAT2 },
        },
        4);
    vertex.SetVertex(WE_IN_POSITION, pos);
    vertex.SetVertex(WE_IN_TEXCOORD, texCoord);
    // TEXT_2F0 / IMAGE_VT_B0 republish +0x2e8 in place. Replace verts on
    // the existing Data so dest-draw nodes that ChangeMeshDataFrom this
    // mesh keep the live AABB, not a parse-time snapshot.
    if (mesh.VertexCount() > 0) {
        mesh.GetVertexArray(0) = std::move(vertex);
    } else {
        mesh.AddVertexArray(std::move(vertex));
    }
    mesh.SetDirty();
}

} // namespace

void SceneObject::ApplyTextDestSize(float layout_w, float layout_h, float pad_x, float pad_y) {
    // TEXT_2F0 / VERTICAL_2E8_SIZE: +0x5a8==0 uses 2.0; Date +0x320>0
    // adds 2*min(pad, 512).
    float width  = layout_w;
    float height = layout_h;
    if (m_effect_count > 0) {
        width += 2.0f * std::min(pad_x, 512.0f);
        height += 2.0f * std::min(pad_y, 512.0f);
    }
    set_dest_size(width, height);
    PublishDestDrawMeshes();
}

void SceneObject::PublishDestDrawMeshes() {
    // Leftover +0x2d8: IMAGE_2D8_NOFULLFB 0x1401eb180 flags=0, size
    // max(4,AABB) from EFFECT_FBO_SIZE. PostFx +0x2e0: POSTFX_MESH
    // {2,2} bit0 → ±1. Last-pass +0x2e8: IMAGE_VT_B0 / VERTICAL_2E8_SIZE
    // cvtdq2ps (int)+0x2f0 flags bit0 → ±half. Do not copy leftover
    // +0x2d8 into +0x2e8.
    const int32_t dest_w = static_cast<int32_t>(m_dest_size.x());
    const int32_t dest_h = static_cast<int32_t>(m_dest_size.y());
    const float leftover_w = static_cast<float>(std::max(4, dest_w));
    const float leftover_h = static_cast<float>(std::max(4, dest_h));
    if (m_leftover_mesh == nullptr) m_leftover_mesh = std::make_unique<SceneMesh>();
    FillCenterFlagsCard(*m_leftover_mesh, leftover_w, leftover_h, 0, m_leftover_uv.x(),
                        m_leftover_uv.y());
    // POSTFX_MESH 0x1401ea3f7 / CARD_CENTER 0x14020679c: +0x2e0 {2,2}
    // flags bit0 → ±1. HORIZONTAL esi!=+0x144 uses this, not leftover
    // +0x2d8 and not last-pass +0x2e8.
    if (m_postfx_mesh == nullptr) m_postfx_mesh = std::make_unique<SceneMesh>();
    FillCenterFlagsCard(*m_postfx_mesh, 2.0f, 2.0f, 1);
    if (m_lastpass_mesh == nullptr) m_lastpass_mesh = std::make_unique<SceneMesh>();
    // IMAGE_490_MESH leftover +0x320==0 Draw uses +0x490 ±half. Last-pass
    // +0x2e8 UV stays 1,1 (TEXT_2F0 / IMAGE_VT_128). Leftover-only has no
    // +0x2e8 last-pass; IMAGE_VT_128 leftover a_TexCoord max is mapRate.
    const float lastpass_u = m_effect_count <= 0 ? m_leftover_uv.x() : 1.0f;
    const float lastpass_v = m_effect_count <= 0 ? m_leftover_uv.y() : 1.0f;
    FillCenterFlagsCard(*m_lastpass_mesh, static_cast<float>(dest_w),
                        static_cast<float>(dest_h), 1, lastpass_u, lastpass_v);
    SizeDestDrawNamedRts();
}

void SceneObject::SizeDestDrawNamedRts() {
    // DEST_ORTHO_TNF leftover named-RT is max(4,(int)+0x2f0). FullCompo
    // scale-1 FBO is the same AABB (NAMED_RT_VPSIZE r9d=1). Not FetchDest.
    if (m_scene == nullptr || m_effect_count <= 0) return;
    const auto rt_it = m_scene->objectRuntimeRenderTargets.find(m_id);
    if (rt_it == m_scene->objectRuntimeRenderTargets.end()) return;
    const int32_t named_w = std::max(4, static_cast<int32_t>(m_dest_size.x()));
    const int32_t named_h = std::max(4, static_cast<int32_t>(m_dest_size.y()));
    for (const auto& name : rt_it->second) {
        const bool leftover_named = sstart_with(name, WE_EFFECT_PPONG_PREFIX);
        const bool full_compo     = sstart_with(name, WE_FULL_COMPO_BUFFER_PREFIX);
        if (!leftover_named && !full_compo) continue;
        auto target = m_scene->renderTargets.find(name);
        if (target == m_scene->renderTargets.end()) continue;
        // NAMED_RT_VPSIZE leftover +0x2c8 and scale-1 FullCompo are
        // max(4,AABB), not window FullFB. TREE parse may mark them
        // bind.screen; official leftover dest-ortho is layer pixels.
        if (target->second.bind.enable && target->second.bind.screen) {
            target->second.bind.enable = false;
            target->second.bind.screen = false;
        }
        // EFFECT_FBO_SIZE 0x140258a02: TEXT_2F0 AABB write then vt+0xb8
        // recreates leftover +0x2c8 / scale-1 FullCompo. TREE Query keys
        // leftover TextPass and HORIZONTAL tex[0] together; a silent
        // resize leaves HORIZONTAL on the old white/empty image.
        if (target->second.width != named_w || target->second.height != named_h ||
            target->second.mapWidth != named_w || target->second.mapHeight != named_h) {
            target->second.width     = named_w;
            target->second.height    = named_h;
            target->second.mapWidth  = named_w;
            target->second.mapHeight = named_h;
            m_scene->MarkRenderTargetResourcesDirty(name);
        }
    }
}

void SceneObject::DestDraw() {
    // PATH_B 0x14018b170 / IMAGE_VT50 jmp 0x1401e8aa0. ENGINE_FLUSH writes
    // +0x930 while dest-STACK is live (LASTPASS_DEST_STACK). Leftover
    // IMAGE_VT_E8 then last-pass 0x1401ea151→0x1401ebf60 go through
    // DestDrawGfx (engine+0x1518 / LEFTOVER_VS_DESTDRAW).
    if (m_scene == nullptr) return;
    // TEXT_2F0 0x140258a02 vt+0xb8 after +0x2f0: leftover +0x2c8 and
    // scale-1 FullCompo are max(4,AABB) (EFFECT_FBO_SIZE / NAMED_RT_VPSIZE)
    // before leftover Draw. TREE letter-box backing must not stay.
    SizeDestDrawNamedRts();
    m_scene->FlushLastPassMvp();
    if (DestDrawGfx* gfx = m_scene->dest_draw_gfx(); gfx != nullptr) gfx->Record(*this);
}
