"""
阶段三测试：全流程端到端离线闭环模拟验证 (Capture -> OCR -> LLM -> Action)
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2

from core.capturer import ScreenCapturer
from core.ocr_engine import OCREngine
from core.llm_client import LLMReasoner
from core.executor import ActionExecutor
from main import AutoQAAgent


def create_mock_question_image(width=600, height=400) -> np.ndarray:
    """生成包含题目与选项的合成测试图片 (物理像素)"""
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    
    # 绘制题干
    cv2.putText(img, "Which function gets list length in Python?", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 绘制选项 A / B / C / D
    cv2.putText(img, "A. size()", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "B. length()", (40, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "C. len()", (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "D. count()", (40, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return img


class TestWorkflowEndToEnd(unittest.TestCase):
    """端到端闭环模拟测试"""

    def setUp(self):
        self.mock_img = create_mock_question_image()
        self.agent = AutoQAAgent(dry_run=True)

    def test_end_to_end_mock_pipeline(self):
        """测试截屏 -> OCR -> 模型推理 -> 坐标换算全流程"""
        # 1. 模拟 Capturer 返回合成图像和 Retina 2.0x 元数据
        metadata = {
            "logical_left": 100.0,
            "logical_top": 150.0,
            "logical_width": 300.0,
            "logical_height": 200.0,
            "physical_width": 600.0,
            "physical_height": 400.0,
            "scale_x": 2.0,
            "scale_y": 2.0,
        }

        # 2. 执行真实 RapidOCR 识别合成图片
        qa_res = self.agent.ocr_engine.parse_qa(self.mock_img)
        self.assertIn("C", qa_res.options_coords)
        
        c_box_x, c_box_y = qa_res.options_coords["C"]
        # y 坐标在 250 附近
        self.assertTrue(220 < c_box_y < 280)

        # 3. 模拟 LLM 推理返回正确答案 "C"
        with patch.object(self.agent.llm_reasoner, "solve", return_value=["C"]):
            answers = self.agent.llm_reasoner.solve(qa_res.question, qa_res.options)
            self.assertEqual(answers, ["C"])

        # 4. 执行 ActionExecutor 坐标换算与模拟点击 (dry_run)
        ok = self.agent.executor.click_option(
            target_option="C",
            options_coords=qa_res.options_coords,
            metadata=metadata,
            dry_run=True
        )
        self.assertTrue(ok)

    def test_blind_click_protection(self):
        """测试防盲点保护：当模型返回不存在的选项时，必须安全拒绝点击"""
        options_coords = {"A": (100.0, 100.0), "B": (100.0, 150.0)}
        metadata = {"logical_left": 0.0, "logical_top": 0.0, "scale_x": 1.0, "scale_y": 1.0}

        # 模型返回 "D"，但只有 A 和 B
        ok = self.agent.executor.click_option(
            target_option="D",
            options_coords=options_coords,
            metadata=metadata,
            dry_run=True
        )
        self.assertFalse(ok)

    def test_auto_next_termination_on_last_question(self):
        """测试当未检测到下一题或下一题未更新时，自动安全停止"""
        from core.ocr_engine import QAResult

        # 构造无“下一题”按钮的题目结果 (最后一题)
        mock_qa = QAResult(
            question="这是最后一题吗？",
            options=["A. 是", "B. 否"],
            options_coords={"A": (100.0, 100.0), "B": (100.0, 150.0)},
            next_button_coord=None,  # 无下一题按钮
            submit_button_coord=(200.0, 300.0),
        )

        metadata = {"logical_left": 0.0, "logical_top": 0.0, "scale_x": 1.0, "scale_y": 1.0}

        with patch.object(self.agent.capturer, "capture", return_value=(self.mock_img, metadata)), \
             patch.object(self.agent.ocr_engine, "parse_qa", return_value=mock_qa), \
             patch.object(self.agent.llm_reasoner, "solve", return_value=["A"]):
            
            # 运行自动答题循环，应作答第 1 题后发现无下一题，正常退出
            success = self.agent.run_auto_qa_loop(once=False)
            self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()

