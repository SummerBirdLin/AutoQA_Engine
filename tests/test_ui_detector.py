"""
测试阶段：OmniParser UI 控件检测器与双轨空间融合专项测试
"""

import os
import unittest
import numpy as np
import cv2

from core.ui_detector import OmniUIDetector, UIElement


class TestOmniUIDetector(unittest.TestCase):
    """OmniParser UI 控件检测器测试"""

    def test_ui_element_properties(self):
        """测试 UIElement 的单选圆圈与卡片判定逻辑"""
        # 单选圆圈 (小尺寸近似正方形/圆形)
        radio_elem = UIElement(
            class_name="icon",
            confidence=0.85,
            bbox=(40.0, 210.0, 80.0, 250.0),
            center_x=60.0,
            center_y=230.0,
            width=40.0,
            height=40.0,
        )
        self.assertTrue(radio_elem.is_radio_or_circle)
        self.assertFalse(radio_elem.is_card)

        # 选项大卡片 (长宽比>=2.0，宽>=180)
        card_elem = UIElement(
            class_name="icon",
            confidence=0.75,
            bbox=(20.0, 200.0, 620.0, 290.0),
            center_x=320.0,
            center_y=245.0,
            width=600.0,
            height=90.0,
        )
        self.assertFalse(card_elem.is_radio_or_circle)
        self.assertTrue(card_elem.is_card)

    def test_detector_disabled_returns_empty(self):
        """测试当配置禁用 OmniParser 时，直接返回空列表且不加载模型"""
        cfg = {"vision": {"enable_omniparser": False}}
        detector = OmniUIDetector(cfg)
        mock_img = np.zeros((300, 400, 3), dtype=np.uint8)
        elements = detector.detect_ui_elements(mock_img)
        self.assertEqual(elements, [])

    def test_spatial_fusion_snapping_to_radio(self):
        """测试双轨空间融合：OCR 选项文字自动吸附至左侧单选圆圈 (Radio Button)"""
        detector = OmniUIDetector({"vision": {"enable_omniparser": True}})
        
        # 模拟 OCR 检出的选项 A 坐标（落在文字上）
        ocr_coords = {"A": (180.0, 235.0)}

        # 模拟 OmniParser 检出的 Radio 按钮和卡片
        mock_elements = [
            UIElement(
                class_name="icon",
                confidence=0.88,
                bbox=(40.0, 215.0, 80.0, 255.0),
                center_x=60.0,
                center_y=235.0,
                width=40.0,
                height=40.0,
            ),
            UIElement(
                class_name="icon",
                confidence=0.70,
                bbox=(20.0, 200.0, 620.0, 290.0),
                center_x=320.0,
                center_y=245.0,
                width=600.0,
                height=90.0,
            ),
        ]

        mock_img = np.zeros((400, 600, 3), dtype=np.uint8)
        fused = detector.fuse_options_with_ocr(
            img_bgr=mock_img,
            ocr_options_coords=ocr_coords,
            detected_elements=mock_elements,
            stem_max_y=150.0
        )

        # 应该优先吸附到 Radio 按钮中心 (60.0, 235.0)，而不是停留在文字 (180.0, 235.0)
        self.assertIn("A", fused)
        self.assertEqual(fused["A"], (60.0, 235.0))

    def test_spatial_fusion_auto_completion(self):
        """测试双轨空间融合：当 OCR 漏检 C、D 选项时，利用 OmniParser 卡片自动补齐"""
        detector = OmniUIDetector({"vision": {"enable_omniparser": True}})

        # OCR 仅检出 A、B，缺失 C、D
        ocr_coords = {"A": (60.0, 235.0), "B": (60.0, 340.0)}

        # OmniParser 检出 4 个整齐排列的卡片与圆圈
        mock_elements = [
            UIElement(class_name="icon", confidence=0.8, bbox=(20, 200, 620, 280), center_x=320, center_y=240, width=600, height=80),
            UIElement(class_name="icon", confidence=0.8, bbox=(20, 300, 620, 380), center_x=320, center_y=340, width=600, height=80),
            UIElement(class_name="icon", confidence=0.8, bbox=(20, 400, 620, 480), center_x=320, center_y=440, width=600, height=80),
            UIElement(class_name="icon", confidence=0.8, bbox=(20, 500, 620, 580), center_x=320, center_y=540, width=600, height=80),
            # C 卡片内部的圆圈
            UIElement(class_name="icon", confidence=0.7, bbox=(40, 420, 80, 460), center_x=60, center_y=440, width=40, height=40),
        ]

        mock_img = np.zeros((700, 640, 3), dtype=np.uint8)
        fused = detector.fuse_options_with_ocr(
            img_bgr=mock_img,
            ocr_options_coords=ocr_coords,
            detected_elements=mock_elements,
            stem_max_y=150.0
        )

        # 检查 A, B 保留，C, D 成功被补齐
        self.assertIn("A", fused)
        self.assertIn("B", fused)
        self.assertIn("C", fused)
        self.assertIn("D", fused)
        # C 应命中内部圆圈 (60, 440)
        self.assertEqual(fused["C"], (60.0, 440.0))
        # D 应落在 D 卡片可点击范围
        self.assertEqual(fused["D"][1], 540.0)

    def test_real_model_inference(self):
        """测试真实 OmniParser 模型权重加载与单帧推理 (若本地已下载)"""
        weight_path = "models/omniparser/model.pt"
        if not os.path.exists(weight_path):
            self.skipTest("本地未检测到 model.pt 权重，跳过真实推理测试")

        detector = OmniUIDetector({"vision": {"enable_omniparser": True, "model_path": weight_path, "device": "auto"}})
        synthetic_img = np.full((400, 600, 3), 255, dtype=np.uint8)
        # 绘制一个单选框测试
        cv2.circle(synthetic_img, (60, 200), 15, (0, 0, 0), 2)
        cv2.rectangle(synthetic_img, (30, 160), (570, 240), (200, 200, 200), 2)

        elements = detector.detect_ui_elements(synthetic_img)
        self.assertIsInstance(elements, list)
        self.assertTrue(detector._initialized)


if __name__ == "__main__":
    unittest.main()
