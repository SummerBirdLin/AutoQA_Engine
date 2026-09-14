"""
多平台通用性与定位框缺失 Fallback 专项测试
验证：
1. 全屏模式截屏与坐标元数据映射 (fullscreen capture)
2. 选项定位框缺失时的几何版面推测 (geometric estimation)
3. 键盘按键模拟与静默桌面通知 Fallback 机制
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from core.capturer import ScreenCapturer
from core.executor import ActionExecutor
from core.ocr_engine import QAResult, TextBlock


class TestFullscreenAndFallback(unittest.TestCase):
    """测试全屏捕获与 Fallback 降级机制"""

    def setUp(self):
        self.config = {
            "capture": {
                "mode": "fullscreen",
                "monitor_index": 1,
                "auto_focus_window": False,
            },
            "action": {
                "pyautogui_failsafe": False,
                "fallback": {
                    "enable_geometric_click": True,
                    "enable_keyboard_shortcut": True,
                    "enable_user_notification": True,
                    "manual_grace_period_sec": 0.05,  # 缩短测试中的等待时间
                }
            }
        }
        self.capturer = ScreenCapturer(self.config)
        self.executor = ActionExecutor(self.config)

    def test_fullscreen_capture_structure(self):
        """测试全屏捕获数据格式与缩放映射"""
        img, meta = self.capturer.capture_fullscreen(monitor_idx=1)
        self.assertIsInstance(img, np.ndarray)
        self.assertGreater(img.shape[0], 0)
        self.assertGreater(img.shape[1], 0)
        self.assertTrue(meta.get("is_fullscreen", False))
        self.assertIn("logical_width", meta)
        self.assertIn("scale_x", meta)

    def test_geometric_fallback_with_partial_options(self):
        """测试已知 A 和 B 时线性推导 C 和 D 的坐标"""
        options_coords = {
            "A": (100.0, 200.0),
            "B": (100.0, 260.0),
        }
        metadata = {"physical_width": 800.0, "physical_height": 600.0}

        # 推导 C: 260 + (260 - 200) = 320.0
        coord_c = ActionExecutor._estimate_geometric_coord("C", options_coords, metadata)
        self.assertIsNotNone(coord_c)
        self.assertAlmostEqual(coord_c[0], 100.0, places=1)
        self.assertAlmostEqual(coord_c[1], 320.0, places=1)

        # 推导 D: 260 + 2 * 60 = 380.0
        coord_d = ActionExecutor._estimate_geometric_coord("D", options_coords, metadata)
        self.assertIsNotNone(coord_d)
        self.assertAlmostEqual(coord_d[0], 100.0, places=1)
        self.assertAlmostEqual(coord_d[1], 380.0, places=1)

    def test_geometric_fallback_with_stem_only(self):
        """测试仅有题干文字块时自上而下推测选项坐标"""
        stem_blocks = [
            TextBlock(box=[[10, 20], [200, 20], [200, 45], [10, 45]], text="下列关于计算机网络的说法正确的是？", confidence=0.9),
            TextBlock(box=[[10, 50], [150, 50], [150, 75], [10, 75]], text="第二行题干描述内容", confidence=0.9),
        ]
        qa_result = QAResult(
            question="下列关于计算机网络的说法正确的是？ 第二行题干描述内容",
            options=["A. 选项一", "B. 选项二"],
            options_coords={},
            text_blocks=stem_blocks,
        )
        metadata = {"physical_width": 1000.0, "physical_height": 800.0}

        # 选项 A 推算 (stem max y 约为 62.5，推算 A 位于约 117.5)
        coord_a = ActionExecutor._estimate_geometric_coord("A", {}, metadata, qa_result=qa_result)
        self.assertIsNotNone(coord_a)
        self.assertAlmostEqual(coord_a[0], 300.0, places=1)  # 1000 * 0.3
        self.assertGreater(coord_a[1], 62.5)

    def test_click_option_triggers_fallback_without_crashing(self):
        """测试当选项缺失时，click_option 不崩溃并成功执行 Fallback"""
        metadata = {
            "logical_left": 0.0,
            "logical_top": 0.0,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "physical_width": 800.0,
            "physical_height": 600.0,
        }
        # 目标选项 C 不在坐标表中
        options_coords = {"A": (100.0, 100.0), "B": (100.0, 150.0)}

        # 在 dry_run 模式下执行，应成功触发几何推测降级并返回 True
        res = self.executor.click_option(
            target_option="C",
            options_coords=options_coords,
            metadata=metadata,
            dry_run=True,
        )
        self.assertTrue(res)

    def test_silent_user_notification_fallback(self):
        """测试完全无法几何推测时，触发静默横幅通知与人工等待缓冲"""
        executor_no_geom = ActionExecutor({
            "action": {
                "fallback": {
                    "enable_geometric_click": False,
                    "enable_keyboard_shortcut": False,
                    "enable_user_notification": True,
                    "manual_grace_period_sec": 0.02,
                }
            }
        })
        metadata = {"scale_x": 1.0, "scale_y": 1.0}

        # 模拟点击未定位选项
        with patch.object(ActionExecutor, "send_desktop_notification") as mock_notify:
            res = executor_no_geom.click_option(
                target_option="X",
                options_coords={},
                metadata=metadata,
                dry_run=False,
            )
            self.assertTrue(res)
            mock_notify.assert_called_once()
            kwargs = mock_notify.call_args.kwargs
            self.assertIn("AutoQA", kwargs.get("title", ""))
            self.assertIn("X", kwargs.get("subtitle", ""))


if __name__ == "__main__":
    unittest.main()
