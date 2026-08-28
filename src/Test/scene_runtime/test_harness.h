#pragma once

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Independent Path B / g_ParallaxPosition expectations. They are copied from the current
// WPNodeTransformResolver / WPShaderValueUpdater formulas so a later edit to those functions
// fails this harness instead of the test silently tracking the implementation.

// This binary does not sample a GPU framebuffer. SCENE_CHECK / QUANT rows are parse,
// Path B numbers, graph inventory, and updater matrices. Test names must not claim
// "draw" or "framebuffer" unless they call sceneToRenderGraph or read a buffer.

struct SceneFailRecord {
    std::string case_name;
    std::string file;
    int         line;
    std::string message;
};

struct SceneQuantRow {
    std::string name;
    float       got_x;
    float       got_y;
    float       expected_x;
    float       expected_y;
};

inline int                        g_scene_test_failures = 0;
inline int                        g_scene_test_checks   = 0;
inline std::string                g_scene_test_name;
inline std::vector<SceneFailRecord> g_scene_test_fail_log;
inline std::vector<SceneQuantRow> g_scene_quant_rows;

inline void SceneRecordFail(const char* file, int line, const std::string& message) {
    ++g_scene_test_failures;
    g_scene_test_fail_log.push_back(SceneFailRecord { g_scene_test_name, file, line, message });
    std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, message.c_str());
    std::fflush(stderr);
}

#define SCENE_CHECK(cond)                                                                          \
    do {                                                                                           \
        ++g_scene_test_checks;                                                                     \
        if (!(cond)) {                                                                             \
            SceneRecordFail(__FILE__, __LINE__, #cond);                                            \
        }                                                                                          \
    } while (0)

#define SCENE_CHECK_NEAR(actual, expected, eps)                                                    \
    do {                                                                                           \
        ++g_scene_test_checks;                                                                     \
        const auto scene_check_a = (actual);                                                       \
        const auto scene_check_e = (expected);                                                     \
        const auto scene_check_eps = (eps);                                                        \
        if (!(std::fabs(static_cast<double>(scene_check_a) -                                       \
                        static_cast<double>(scene_check_e)) <=                                     \
              static_cast<double>(scene_check_eps))) {                                             \
            char scene_check_buf[256];                                                             \
            std::snprintf(scene_check_buf,                                                         \
                          sizeof(scene_check_buf),                                                 \
                          "%s got=%.6f expected=%.6f eps=%.6g",                                    \
                          #actual " ≈ " #expected,                                                 \
                          static_cast<double>(scene_check_a),                                      \
                          static_cast<double>(scene_check_e),                                      \
                          static_cast<double>(scene_check_eps));                                   \
            SceneRecordFail(__FILE__, __LINE__, scene_check_buf);                                  \
        }                                                                                          \
    } while (0)

#define SCENE_CHECK_STREQ(actual, expected)                                                        \
    do {                                                                                           \
        ++g_scene_test_checks;                                                                     \
        const std::string scene_check_a = (actual);                                                \
        const std::string scene_check_e = (expected);                                              \
        if (scene_check_a != scene_check_e) {                                                      \
            SceneRecordFail(__FILE__,                                                              \
                            __LINE__,                                                              \
                            std::string(#actual " == " #expected) + " got=\"" + scene_check_a +    \
                                "\" expected=\"" + scene_check_e + "\"");                          \
        }                                                                                          \
    } while (0)

inline void SceneTestBegin(const char* name) {
    g_scene_test_name = name;
    std::fprintf(stderr, "== %s\n", name);
    std::fflush(stderr);
}

inline void SceneReportVec2(const char* name, float got_x, float got_y, float expected_x,
                            float expected_y) {
    g_scene_quant_rows.push_back(SceneQuantRow { name, got_x, got_y, expected_x, expected_y });
    std::fprintf(stderr,
                 "  QUANT %s got=(%.6f, %.6f) expected=(%.6f, %.6f)\n",
                 name,
                 static_cast<double>(got_x),
                 static_cast<double>(got_y),
                 static_cast<double>(expected_x),
                 static_cast<double>(expected_y));
    std::fflush(stderr);
}

inline int SceneTestSummary() {
    std::fprintf(stderr, "\n=== QUANT TABLE (%zu rows) ===\n", g_scene_quant_rows.size());
    for (const auto& row : g_scene_quant_rows) {
        const bool ok = std::fabs(static_cast<double>(row.got_x) - static_cast<double>(row.expected_x)) <=
                            1e-4 &&
                        std::fabs(static_cast<double>(row.got_y) - static_cast<double>(row.expected_y)) <=
                            1e-4;
        std::fprintf(stderr,
                     "%s  %s got=(%.6f, %.6f) expected=(%.6f, %.6f)\n",
                     ok ? "OK  " : "FAIL",
                     row.name.c_str(),
                     static_cast<double>(row.got_x),
                     static_cast<double>(row.got_y),
                     static_cast<double>(row.expected_x),
                     static_cast<double>(row.expected_y));
    }
    if (! g_scene_test_fail_log.empty()) {
        std::fprintf(stderr, "\n=== FAIL LEDGER (%d/%d) ===\n", g_scene_test_failures,
                     g_scene_test_checks);
        for (const auto& rec : g_scene_test_fail_log) {
            std::fprintf(stderr,
                         "FAIL [%s] %s:%d: %s\n",
                         rec.case_name.c_str(),
                         rec.file.c_str(),
                         rec.line,
                         rec.message.c_str());
        }
    }
    if (g_scene_test_failures == 0) {
        std::fprintf(stderr, "OK %d checks\n", g_scene_test_checks);
        std::fflush(stderr);
        return 0;
    }
    std::fprintf(stderr, "FAILED %d/%d checks\n", g_scene_test_failures, g_scene_test_checks);
    std::fflush(stderr);
    return 1;
}

// Official lookat (0x14018928a xmm7=0.5, 0x140189b7e–0x140189c24):
// lookat = [scene+0xf0] + 0.5*ortho*(1-inf) + (clamp(mx), clamp(1-my))*ortho*inf
inline Eigen::Vector2f ExpectedLookatWorld(const Eigen::Vector2f& cam_f0,
                                           const Eigen::Vector2f& mouse_ndc,
                                           const Eigen::Vector2f& ortho, float influence) {
    const float mx      = std::clamp(mouse_ndc.x(), 0.0f, 1.0f);
    const float my_term = std::clamp(1.0f - mouse_ndc.y(), 0.0f, 1.0f);
    const Eigen::Vector2f half         = ortho * 0.5f;
    const Eigen::Vector2f mouse_canvas { mx * ortho.x(), my_term * ortho.y() };
    return cam_f0 + half * (1.0f - influence) + mouse_canvas * influence;
}

// Vivid 2D GetPosition() is already official rest lookat (cam_f0 + 0.5*ortho,
// ctor 0x14018870c). Official [scene+0xf0] is the authored camera origin.
inline Eigen::Vector2f OfficialLookatFromViewCamera(const Eigen::Vector2f& view_xy,
                                                    const Eigen::Vector2f& mouse_ndc,
                                                    const Eigen::Vector2f& ortho,
                                                    float influence) {
    return ExpectedLookatWorld(view_xy - ortho * 0.5f, mouse_ndc, ortho, influence);
}

inline Eigen::Vector2f ExpectedPathBFromLookat(const Eigen::Vector2f& node_xy,
                                               const Eigen::Vector2f& lookat,
                                               const Eigen::Vector2f& depth, float amount) {
    return (node_xy - lookat).cwiseProduct(depth) * amount;
}

// cam_xy is Vivid view position (official rest lookat when +0xf0=0).
inline Eigen::Vector2f ExpectedPathBOffset(const Eigen::Vector2f& node_xy,
                                           const Eigen::Vector2f& cam_xy,
                                           const Eigen::Vector2f& mouse_ndc,
                                           const Eigen::Vector2f& ortho,
                                           const Eigen::Vector2f& depth, float amount,
                                           float influence) {
    return ExpectedPathBFromLookat(
        node_xy, OfficialLookatFromViewCamera(cam_xy, mouse_ndc, ortho, influence), depth,
        amount);
}

inline Eigen::Vector2f ExpectedParallaxPositionNdc(const Eigen::Vector2f& mouse_ndc, float influence,
                                                   bool enable) {
    if (! enable) return { 0.5f, 0.5f };
    const Eigen::Vector2f centered = mouse_ndc - Eigen::Vector2f { 0.5f, 0.5f };
    return Eigen::Vector2f { 0.5f, 0.5f } + Eigen::Scaling(1.0f, -1.0f) * centered * influence;
}
