#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace wallpaper
{

class Scene;
class SceneMesh;
class SceneNode;
class SceneObject;
class WPPuppetLayer;

// Official 2.8.42 layer identity. RTTI names are ImageLayer / IPropertyObject;
// ownership matches the base object at factory 0x14018ff60 / ctor 0x1401ddbb0.
enum class SceneObjectKind
{
    Empty,
    Image,
    Particle,
    Text,
    Light,
    Sound,
    Camera,
    Model,
    Shape,
};

// Official dest-draw phases on one +0x158 object (LEFTOVER_VS_DESTDRAW).
// Leftover is IMAGE_VT_E8 / IMAGE_VT_F0 / TEXT_VT_F0. PostFx is type-0
// inside 0x1401ebf60. LastPass is VERTICAL no-target FullFB (POSTFX_OMSET)
// when +0x304 bit4 is clear. LeftoverMvp is IMAGE_VT_F8 / TEXT_VT_F8 when
// bit4 is set (DRAW_FLAG304 / PUPPET_304_BIT4).
enum class DestDrawPhase
{
    None,
    Leftover,
    LeftoverMvp,
    PostFx,
    LastPass,
};

class DestDrawGfx {
public:
    virtual ~DestDrawGfx() = default;
    // PATH_B 0x14018b170 vt+0x50 before dest pop. Leftover then last-pass
    // (IMAGE_VT_E8 0x1401e98dc, then 0x1401ea151→0x1401ebf60).
    virtual void Record(SceneObject& object) = 0;
};

class SceneObject {
public:
    // +0x120 ctor 0x2001; hide inherit reads bit0 (0x140185014).
    static constexpr uint32_t kFlagVisible  = 0x1u;
    static constexpr uint32_t kDefaultFlags = 0x2001u;

    SceneObject(Scene* scene, int32_t id);
    ~SceneObject();

    Scene*  scene() const { return m_scene; }
    int32_t id() const { return m_id; }

    SceneObjectKind kind() const { return m_kind; }
    void            set_kind(SceneObjectKind kind) { m_kind = kind; }

    const std::string& name() const { return m_name; }
    void               set_name(std::string name) { m_name = std::move(name); }

    // +0x128 / +0x12c / +0x130
    const Eigen::Vector3f& origin() const { return m_origin; }
    void                   set_origin(const Eigen::Vector3f& value);

    // +0x134 / +0x138 / +0x13c
    const Eigen::Vector3f& scale() const { return m_scale; }
    void                   set_scale(const Eigen::Vector3f& value);

    // +0x140 / +0x144 / +0x148
    const Eigen::Vector3f& angles() const { return m_angles; }
    void                   set_angles(const Eigen::Vector3f& value);

    // +0x170 / +0x174 ctor {1,1} (0x1401ddce1)
    const Eigen::Vector2f& parallax_depth() const { return m_parallax_depth; }
    void                   set_parallax_depth(const Eigen::Vector2f& value);
    bool                   parallax_depth_authored() const { return m_parallax_depth_authored; }
    void                   set_parallax_depth_authored(bool authored) {
        m_parallax_depth_authored = authored;
    }
    // PATH_B 0x14018b0ab / 0x14018b0b9 leftover ox/oy on the walked object.
    const Eigen::Vector2f& leftover_parallax() const { return m_leftover_parallax; }
    void                   set_leftover_parallax(float ox, float oy) {
        m_leftover_parallax = Eigen::Vector2f(ox, oy);
    }

    uint32_t flags() const { return m_flags; }
    bool     local_visible() const { return (m_flags & kFlagVisible) != 0; }
    void     set_local_visible(bool visible);
    // 0x140185010: bit0 then walk +0x180
    bool EffectiveVisible() const;

    // +0x180 / +0x190 (0x1401de931)
    SceneObject* parent() const { return m_parent; }
    int32_t      attach_index() const { return m_attach_index; }
    void         set_parent(SceneObject* parent, int32_t attach_index);
    const std::vector<SceneObject*>& children() const { return m_children; }
    // Authored bone-attachment name (scene.json "attachment"). The resolved
    // bone index lives in attach_bone_index(); this keeps the authored string
    // so scripts can round-trip it.
    const std::string& attachment() const { return m_attachment; }
    void               set_attachment(std::string name) { m_attachment = std::move(name); }

    SceneObject*       Root();
    const SceneObject* Root() const;

    // 0x1401df0cd: zero +0x140/+0x148, copy to origin, keep scale, rebuild basis.
    // Official setParent reaches this only when r8 adjustTransforms != 0
    // (0x1401de925). Scene JSON bind uses r8=0 and leaves +0x128.
    void ApplyAttachZeroOrigin();
    void SetAttachBinding(uint32_t bone_index, const Eigen::Affine3f& bind);
    uint32_t attach_bone_index() const { return m_attach_bone; }
    const Eigen::Affine3f& attach_bind() const { return m_attach_bind; }

    void MarkDestDirty();
    // 0x1401850a0: origin → +0x110, 3×3 = scale * basis, bone then parent * dest
    const Eigen::Matrix4f& FetchDest();
    Eigen::Matrix4f        ComposeLocalDest() const;
    // PATH_B 0x14018b170 vt+0x50 before dest pop. Leftover IMAGE_VT_E8 then
    // last-pass 0x1401ea151→0x1401ebf60 (LEFTOVER_VS_DESTDRAW).
    void DestDraw();

    // IMAGE_SIZE_PROP / TEXT_2F0 +0x2f0/+0x2f4. Ctor 0x1401e69be writes 1.0.
    const Eigen::Vector2f& dest_size() const { return m_dest_size; }
    void                   set_dest_size(float width, float height);

    // EFFECT_COUNT +0x320. Shared ctor zeros it.
    int32_t effect_count() const { return m_effect_count; }
    void    set_effect_count(int32_t count) { m_effect_count = count; }
    // PARSE_304_BIT4 0x1401e6fa2: +0x32c!=0 && !=0x1f ors +0x304 bit4.
    void ApplyParse304Bit4(int32_t color_blend_mode);
    // PUPPET_304_BIT4 0x14020b20d: +0x320>0 && [+0x4b8+8]!=0 ors bit4.
    void ApplyPuppet304Bit4();
    bool Flag304Bit4() const { return m_flag304_bit4; }
    // IMAGE_DRAW_PASS +0x320>0 dest-draw leftover + in-loop dest-draw.
    bool DestDrawHasEffects() const {
        return (m_kind == SceneObjectKind::Image || m_kind == SceneObjectKind::Text) &&
               m_effect_count > 0;
    }
    // Official POSTFX last-pass 0x1401ea151 when +0x320>0 and bit4 clear
    // (LEFTOVER_VS_DESTDRAW / DRAW_FLAG304). bit4 takes IMAGE_VT_F8 leftover-MVP.
    bool DestDrawPublishesDefault() const {
        return DestDrawHasEffects() && !m_flag304_bit4;
    }

    // IMAGE_2D8_NOFULLFB vt+0xb8 0x1401eb180 flags=0 → 0..max(4,AABB).
    const SceneMesh* leftover_mesh() const { return m_leftover_mesh.get(); }
    // POSTFX_MESH / CARD_CENTER: +0x2e0 size {2,2} flags bit0 → ±1.
    const SceneMesh* postfx_mesh() const { return m_postfx_mesh.get(); }
    // IMAGE_VT_B0 / POSTFX_MESH / VERTICAL_2E8_SIZE: +0x2e8 ±half (int)+0x2f0.
    SceneMesh* lastpass_mesh() { return m_lastpass_mesh.get(); }
    const SceneMesh* lastpass_mesh() const { return m_lastpass_mesh.get(); }
    // PUPPET_490 0x14020b165: leftover-card writes +0x490 from puppet+0x18
    // verts even when +0x320>0 skips leftover-card draw. IMAGE_VT_F8_PUPPET
    // leftover-MVP Draws that mesh, not +0x2e8 dest AABB.
    const SceneMesh* image_490_mesh() const { return m_image_490_mesh.get(); }
    void             CaptureImage490Mesh(const SceneMesh& mesh);
    // Official leftover dest-ortho Draws +0x2d8 and leftover-MVP Draws
    // +0x490 on the same object (IMAGE_2D8_NOFULLFB / PUPPET_490). TREE
    // leftover owner Mesh() is leftover dest-ortho; leftover-MVP bind is
    // this node so prepare/upload see +0x490, not leftover dest card.
    SceneNode* leftover_mvp_node() const { return m_leftover_mvp_node.get(); }
    SceneNode* EnsureLeftoverMvpNode(SceneNode* leftover_owner);
    // IMAGE_VT_128 0x140209290: leftover a_TexCoord max is content/physical
    // (tex+0x2c)/(tex+0x20). Last-pass +0x2e8 stays 1,1 (TEXT_2F0).
    const Eigen::Vector2f& leftover_uv() const { return m_leftover_uv; }
    void                   set_leftover_uv(float u, float v);
    // TEXT_2F0 calls vt+0xb0 then vt+0xb8 after +0x2f0. Image vt+0x110 → vt+0xb0.
    void PublishDestDrawMeshes();
    // NAMED_RT_VPSIZE / EFFECT_FBO_SIZE: leftover +0x2c8 and scale-1
    // FullCompo FBOs are max(4,AABB). Clears TREE bind.screen so
    // setRenderTargetSize cannot restore window FullFB.
    void SizeDestDrawNamedRts();
    // TEXT_2F0 0x140258916: +0x2f0 = layout AABB, or 2.0 if no +0x5a8.
    // +0x320>0 adds 2*min(+0x4e8/+0x4ec, 512). Then publish +0x2d8/+0x2e8.
    void ApplyTextDestSize(float layout_w, float layout_h, float pad_x, float pad_y);

    SceneNode* source_node() const { return m_source_node; }
    void       set_source_node(SceneNode* node) { m_source_node = node; }

    WPPuppetLayer*       puppet_layer();
    const WPPuppetLayer* puppet_layer() const;
    void                 set_puppet_layer(const WPPuppetLayer& layer);

private:
    void RebuildBasisFromAngles();
    // 0x140185040: rcx=this, edx=incoming +0x190 from the caller.
    bool AncestorDestCacheValid(int32_t incoming_attach) const;

    Scene*          m_scene { nullptr };
    int32_t         m_id { 0 };
    SceneObjectKind m_kind { SceneObjectKind::Empty };
    std::string     m_name;
    uint32_t        m_flags { kDefaultFlags };
    Eigen::Vector3f m_origin { Eigen::Vector3f::Zero() };
    Eigen::Vector3f m_scale { 1.0f, 1.0f, 1.0f };
    Eigen::Vector3f m_angles { Eigen::Vector3f::Zero() };
    Eigen::Matrix3f m_basis { Eigen::Matrix3f::Identity() };
    Eigen::Vector2f m_parallax_depth { 1.0f, 1.0f };
    bool            m_parallax_depth_authored { false };
    Eigen::Vector2f m_leftover_parallax { 0.0f, 0.0f };
    SceneObject*    m_parent { nullptr };
    int32_t         m_attach_index { -1 };
    std::string     m_attachment;
    uint32_t        m_attach_bone { 0xFFFFFFFFu };
    Eigen::Affine3f m_attach_bind { Eigen::Affine3f::Identity() };
    std::vector<SceneObject*> m_children;
    uint32_t        m_dest_stamp { 0 };
    Eigen::Matrix4f m_dest { Eigen::Matrix4f::Identity() };
    SceneNode*      m_source_node { nullptr };
    Eigen::Vector2f m_dest_size { 1.0f, 1.0f };
    Eigen::Vector2f m_leftover_uv { 1.0f, 1.0f };
    int32_t         m_effect_count { 0 };
    bool            m_flag304_bit4 { false };
    std::unique_ptr<SceneMesh> m_leftover_mesh;
    std::unique_ptr<SceneMesh> m_postfx_mesh;
    std::unique_ptr<SceneMesh> m_lastpass_mesh;
    std::unique_ptr<SceneMesh> m_image_490_mesh;
    std::unique_ptr<SceneNode> m_leftover_mvp_node;
    std::unique_ptr<WPPuppetLayer> m_puppet_layer;
};

} // namespace wallpaper
