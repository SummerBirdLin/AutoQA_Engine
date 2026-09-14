"""
阶段一测试：Retina DPI 与屏幕坐标空间换算测试
"""

import unittest
from core.capturer import ScreenCapturer


class TestCoordinateTransform(unittest.TestCase):
    """测试 OCR 物理像素坐标到屏幕逻辑点击坐标的精确转换"""

    def test_standard_scale_1x(self):
        """测试标准 1.0x 显示器"""
        metadata = {
            "logical_left": 100.0,
            "logical_top": 200.0,
            "scale_x": 1.0,
            "scale_y": 1.0,
        }
        # 假设 OCR 检测到文本中心为 (50, 80)
        target_x, target_y = ScreenCapturer.calculate_logical_target(50.0, 80.0, metadata)
        self.assertAlmostEqual(target_x, 150.0, places=2)
        self.assertAlmostEqual(target_y, 280.0, places=2)

    def test_mac_retina_scale_2x(self):
        """测试 macOS 典型 Retina 2.0x 屏幕"""
        metadata = {
            "logical_left": 200.0,
            "logical_top": 150.0,
            "scale_x": 2.0,
            "scale_y": 2.0,
        }
        # 物理像素图像大小为 600x400，中心为 (300, 200)
        # 对应的逻辑中心偏移应为 (150, 100)
        target_x, target_y = ScreenCapturer.calculate_logical_target(300.0, 200.0, metadata)
        self.assertAlmostEqual(target_x, 350.0, places=2)
        self.assertAlmostEqual(target_y, 250.0, places=2)

    def test_fractional_scale_1_5x(self):
        """测试 1.5x 分数缩放屏幕"""
        metadata = {
            "logical_left": 300.0,
            "logical_top": 400.0,
            "scale_x": 1.5,
            "scale_y": 1.5,
        }
        # 物理坐标 150 -> 逻辑偏移 100
        target_x, target_y = ScreenCapturer.calculate_logical_target(150.0, 300.0, metadata)
        self.assertAlmostEqual(target_x, 400.0, places=2)
        self.assertAlmostEqual(target_y, 600.0, places=2)


class TestHotkeyNormalization(unittest.TestCase):
    """测试热键字符串标准化解析"""

    def test_hotkey_parsing(self):
        from main import normalize_hotkey
        from pynput.keyboard import HotKey

        cases = ["F8", "f8", "<f8>", "ctrl+q", "<ctrl>+q", "ctrl+alt+a", "CMD+SHIFT+Q"]
        for c in cases:
            norm = normalize_hotkey(c)
            # 验证 HotKey.parse 能顺利解析且不报错
            parsed = HotKey.parse(norm)
class TestWindowCaptureLogic(unittest.TestCase):
    """测试应用窗口截屏边界与坐标计算"""

    def test_window_bounds_target_calculation(self):
        # 模拟一个位于 (100, 50) 逻辑坐标、大小为 1200x800 的应用窗口
        # Retina 2.0 缩放下，物理图片大小为 2400x1600
        window_metadata = {
            "logical_left": 100.0,
            "logical_top": 50.0,
            "logical_width": 1200.0,
            "logical_height": 800.0,
            "physical_width": 2400.0,
            "physical_height": 1600.0,
            "scale_x": 2.0,
            "scale_y": 2.0,
            "window_id": 12345,
        }

        # 假设 OCR 在该应用窗口物理图片内识别出选项 B 中心在 (600, 400)
        target_x, target_y = ScreenCapturer.calculate_logical_target(600.0, 400.0, window_metadata)

        # 逻辑坐标应为: 100 + 600/2.0 = 400, 50 + 400/2.0 = 250
        self.assertAlmostEqual(target_x, 400.0, places=2)
        self.assertAlmostEqual(target_y, 250.0, places=2)


if __name__ == "__main__":
    unittest.main()

