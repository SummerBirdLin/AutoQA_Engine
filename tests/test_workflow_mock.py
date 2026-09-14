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
        """测试防盲点保护：当选项不存在且所有降级机制关闭时，必须安全拒绝点击"""
        options_coords = {"A": (100.0, 100.0), "B": (100.0, 150.0)}
        metadata = {"logical_left": 0.0, "logical_top": 0.0, "scale_x": 1.0, "scale_y": 1.0}

        # 临时禁用 fallback，测试严格防盲点保护拦截
        orig_fb = self.agent.executor.config.get("action", {}).get("fallback", {})
        self.agent.executor.config["action"]["fallback"] = {
            "enable_geometric_click": False,
            "enable_keyboard_shortcut": False,
            "enable_user_notification": False,
        }
        try:
            ok = self.agent.executor.click_option(
                target_option="D",
                options_coords=options_coords,
                metadata=metadata,
                dry_run=True
            )
            self.assertFalse(ok)
        finally:
            self.agent.executor.config["action"]["fallback"] = orig_fb

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

    def test_cached_next_button_reuse(self):
        """测试下一题坐标缓存复用：第 1 题检测到按钮，第 2 题漏检时复用第 1 题坐标"""
        from core.ocr_engine import QAResult

        # 第 1 题：检出下一题按钮 (250.0, 350.0)
        qa1 = QAResult(
            question="这是第 1 题",
            options=["A. 选项 1", "B. 选项 2"],
            options_coords={"A": (100.0, 100.0), "B": (100.0, 150.0)},
            next_button_coord=(250.0, 350.0),
        )
        # 第 2 题：因 OCR 抖动漏检下一题 (next_button_coord=None)
        qa2 = QAResult(
            question="这是第 2 题",
            options=["A. 选项 1", "B. 选项 2"],
            options_coords={"A": (100.0, 100.0), "B": (100.0, 150.0)},
            next_button_coord=None,  # 漏检
        )

        metadata = {"logical_left": 0.0, "logical_top": 0.0, "scale_x": 1.0, "scale_y": 1.0}

        clicked_next_coords = []
        original_click_next = self.agent.executor.click_next_button

        def mock_click_next(next_button_coord, **kwargs):
            clicked_next_coords.append(next_button_coord)
            return original_click_next(next_button_coord, **kwargs)

        # 模拟执行第 1 题后跳转到第 2 题，再跳转判定停止
        with patch.object(self.agent.capturer, "capture", return_value=(self.mock_img, metadata)), \
             patch.object(self.agent.ocr_engine, "parse_qa", side_effect=[qa1, qa2, qa2]), \
             patch.object(self.agent.llm_reasoner, "solve", return_value=["A"]), \
             patch.object(self.agent.executor, "click_next_button", side_effect=mock_click_next), \
             patch.object(self.agent, "_wait_for_question_transition", side_effect=[True, False]):

            success = self.agent.run_auto_qa_loop(once=False)
            self.assertTrue(success)

            # 验证点击了两次下一题：第 1 次用原坐标，第 2 次成功复用缓存坐标
            self.assertEqual(len(clicked_next_coords), 2)
            self.assertEqual(clicked_next_coords[0], (250.0, 350.0))
            self.assertEqual(clicked_next_coords[1], (250.0, 350.0))

    def test_is_same_question_accuracy(self):
        """测试题干相似度对比与标点/OCR 抖动容忍度"""
        from main import AutoQAAgent

        # 标点、引号、空白差异应判定为同一题
        q_real1 = "5. 单选题 根据总体国家安全观，下列哪一项最能体现‘安全与发展并重’的原则？"
        q_real2 = "5. 单选题 根据总体国家安全观，下列哪一项最能体现'安全与发展并重'的原则？"
        q_real3 = "5. 单选题 根据总体国家安全观，下列哪一项最能体现“安全与发展并重”的原则？"
        self.assertTrue(AutoQAAgent.is_same_question(q_real1, q_real2))
        self.assertTrue(AutoQAAgent.is_same_question(q_real1, q_real3))

        # 明显不同题目应判定为不同题
        self.assertFalse(AutoQAAgent.is_same_question("这是第 1 题", "这是第 2 题"))
        self.assertFalse(AutoQAAgent.is_same_question("4. 单选题 维护国家安全的神圣义务", q_real1))

    def test_last_question_infinite_loop_prevention(self):
        """测试最后一题点击下一题后页面未变（或带微小 OCR 抖动）时，双重保护机制安全终止"""
        from core.ocr_engine import QAResult

        # 最后一题：带有微小标点差异的同题 OCR
        qa_frame1 = QAResult(
            question="5. 单选题 根据总体国家安全观，下列哪一项最能体现‘安全与发展并重’的原则？",
            options=["A. 选项A", "B. 选项B"],
            options_coords={"A": (100.0, 100.0), "B": (100.0, 150.0)},
            next_button_coord=(250.0, 350.0),
        )
        qa_frame2 = QAResult(
            question="5. 单选题 根据总体国家安全观，下列哪一项最能体现'安全与发展并重'的原则？",
            options=["A. 选项A", "B. 选项B"],
            options_coords={"A": (100.0, 100.0), "B": (100.0, 150.0)},
            next_button_coord=(250.0, 350.0),
        )

        metadata = {"logical_left": 0.0, "logical_top": 0.0, "scale_x": 1.0, "scale_y": 1.0}
        solved_count = 0

        def count_solve(*args, **kwargs):
            nonlocal solved_count
            solved_count += 1
            return ["B"]

        with patch.object(self.agent.capturer, "capture", return_value=(self.mock_img, metadata)), \
             patch.object(self.agent.ocr_engine, "parse_qa", side_effect=[qa_frame1, qa_frame2, qa_frame2]), \
             patch.object(self.agent.llm_reasoner, "solve", side_effect=count_solve), \
             patch.object(self.agent.executor, "click_next_button", return_value=True):

            # 设置极短翻页判定超时，防止单元测试耗时过长
            self.agent.config["auto_next"]["page_transition_timeout"] = 0.5
            success = self.agent.run_auto_qa_loop(once=False)
            self.assertTrue(success)
            # 必须仅作答 1 次，随后识别到页面未变退出，绝不陷入死循环
            self.assertEqual(solved_count, 1)


if __name__ == "__main__":
    unittest.main()

