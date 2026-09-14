"""
阶段二测试：OCR 结构化题目解析与 LLM 输出解析降级测试
"""

import unittest
import numpy as np
import cv2

from core.llm_client import LLMReasoner
from core.ocr_engine import OCREngine, TextBlock


class TestLLMResponseParser(unittest.TestCase):
    """测试 LLM 多重降级解析"""

    def test_standard_json(self):
        reply = '{"answer": "B"}'
        ans = LLMReasoner.parse_llm_response(reply, ["A", "B", "C", "D"])
        self.assertEqual(ans, ["B"])

    def test_json_in_markdown_block(self):
        reply = """```json
{
  "answer": "C"
}
```"""
        ans = LLMReasoner.parse_llm_response(reply, ["A", "B", "C", "D"])
        self.assertEqual(ans, ["C"])

    def test_multichoice_list_and_string(self):
        # 列表格式
        reply1 = '{"answer": ["A", "D"]}'
        self.assertEqual(LLMReasoner.parse_llm_response(reply1), ["A", "D"])

        # 字符串格式
        reply2 = '{"answer": "AC"}'
        self.assertEqual(LLMReasoner.parse_llm_response(reply2), ["A", "C"])

    def test_conversational_fallback_regex(self):
        # 模型返回解释性文字
        reply = "经过对题目的深度分析，本题考查的是 Python 内置函数，正确选项是 C，因为 len() 返回长度。"
        ans = LLMReasoner.parse_llm_response(reply, ["A", "B", "C", "D"])
        self.assertEqual(ans, ["C"])

    def test_numeric_options_fallback(self):
        # 数字标号选项
        reply = '{"answer": "3"}'
        ans = LLMReasoner.parse_llm_response(reply, ["1", "2", "3", "4"])
        self.assertEqual(ans, ["3"])


class TestOCRQAStructure(unittest.TestCase):
    """测试 OCR 结构化与选项提取逻辑"""

    def setUp(self):
        self.ocr_engine = OCREngine()

    def test_extract_qa_with_standard_options(self):
        # 模拟 5 个文本块：1 个题干，4 个标准选项
        blocks = [
            TextBlock(box=[[10, 10], [200, 10], [200, 30], [10, 30]], text="Python获取列表长度的内置函数是？", confidence=0.95),
            TextBlock(box=[[10, 50], [100, 50], [100, 70], [10, 70]], text="A. size()", confidence=0.92),
            TextBlock(box=[[10, 90], [100, 90], [100, 110], [10, 110]], text="B. length()", confidence=0.91),
            TextBlock(box=[[10, 130], [100, 130], [100, 150], [10, 150]], text="C. len()", confidence=0.94),
            TextBlock(box=[[10, 170], [100, 170], [100, 190], [10, 190]], text="D. count()", confidence=0.93),
        ]
        qa = self.ocr_engine._extract_qa_structure(blocks)

        self.assertIn("Python获取列表长度", qa.question)
        self.assertEqual(len(qa.options), 4)
        self.assertIn("A", qa.options_coords)
        self.assertIn("B", qa.options_coords)
        self.assertIn("C", qa.options_coords)
        self.assertIn("D", qa.options_coords)

        # 检查中心坐标换算 (C 选项: y 在 130~150，中心 y 应为 140)
        cx, cy = qa.options_coords["C"]
        self.assertAlmostEqual(cy, 140.0, places=1)

    def test_extract_qa_without_letters_fallback(self):
        """测试无字母选项时的纵向自上而下顺位排序降级策略"""
        blocks = [
            TextBlock(box=[[10, 10], [300, 10], [300, 30], [10, 30]], text="下列属于哺乳动物的是哪一项？", confidence=0.95),
            TextBlock(box=[[10, 60], [100, 60], [100, 80], [10, 80]], text="企鹅", confidence=0.90),
            TextBlock(box=[[10, 100], [100, 100], [100, 120], [10, 120]], text="蓝鲸", confidence=0.92),
            TextBlock(box=[[10, 140], [100, 140], [100, 160], [10, 160]], text="鳄鱼", confidence=0.89),
            TextBlock(box=[[10, 180], [100, 180], [100, 200], [10, 200]], text="金鱼", confidence=0.91),
        ]
        qa = self.ocr_engine._extract_qa_structure(blocks)

        self.assertIn("哺乳动物", qa.question)
        # 应降级自动映射 A/B/C/D 与 1/2/3/4
        self.assertIn("A", qa.options_coords)
        self.assertIn("B", qa.options_coords)
        self.assertIn("1", qa.options_coords)
        self.assertIn("2", qa.options_coords)

    def test_question_with_header_and_unlabelled_options(self):
        """测试包含 '1. 单选题' 标头且选项无字母 (类似网页单选按钮) 的真实场景"""
        blocks = [
            TextBlock(box=[[18, 29], [104, 29], [104, 51], [18, 51]], text="1.单选题", confidence=0.82),
            TextBlock(box=[[21, 87], [607, 89], [607, 114], [21, 112]], text="“中国梦”的实现离不开每个人的努力，下列哪一项最能", confidence=0.86),
            TextBlock(box=[[21, 128], [349, 128], [349, 152], [21, 152]], text="体现个体理想与中国梦的关系？", confidence=0.91),
            TextBlock(box=[[114, 222], [387, 222], [387, 246], [114, 246]], text="个人理想与中国梦毫无关联", confidence=0.90),
            TextBlock(box=[[113, 326], [523, 326], [523, 349], [113, 349]], text="个人理想的实现有助于推动中国梦的实现", confidence=0.90),
            TextBlock(box=[[114, 429], [455, 429], [455, 453], [114, 452]], text="中国梦实现后，个人理想自动实现", confidence=0.91),
            TextBlock(box=[[113, 532], [454, 532], [454, 556], [113, 556]], text="只有国家强大，个人理想才有意义", confidence=0.88),
        ]
        qa = self.ocr_engine._extract_qa_structure(blocks)

        # 验证 '1.单选题' 没有被错当成选项
        self.assertNotIn("1.单选题", [opt.replace("A. ", "").replace("1. ", "") for opt in qa.options])
        self.assertIn("体现个体理想与中国梦的关系？", qa.question)
        
        # 验证成功识别出 4 个选项
        self.assertEqual(len(qa.options), 4)
        self.assertIn("A", qa.options_coords)
        self.assertIn("B", qa.options_coords)
        self.assertIn("C", qa.options_coords)
        self.assertIn("D", qa.options_coords)
        # 且支持数字别名
        self.assertIn("1", qa.options_coords)
        self.assertIn("2", qa.options_coords)
        self.assertIn("3", qa.options_coords)
        self.assertIn("4", qa.options_coords)

    def test_next_and_submit_button_detection(self):
        """测试页面底部包含'下一题'与'交卷'按钮时的识别与选项排除"""
        blocks = [
            TextBlock(box=[[10, 10], [200, 10], [200, 30], [10, 30]], text="请问地球到太阳的平均距离大约是？", confidence=0.95),
            TextBlock(box=[[10, 50], [100, 50], [100, 70], [10, 70]], text="A. 1.5亿公里", confidence=0.92),
            TextBlock(box=[[10, 90], [100, 90], [100, 110], [10, 110]], text="B. 38万公里", confidence=0.91),
            TextBlock(box=[[10, 130], [100, 130], [100, 150], [10, 150]], text="C. 1光年", confidence=0.94),
            TextBlock(box=[[10, 170], [100, 170], [100, 190], [10, 190]], text="D. 4000万公里", confidence=0.93),
            # 底部按钮
            TextBlock(box=[[200, 250], [280, 250], [280, 280], [200, 280]], text="下一题", confidence=0.96),
            TextBlock(box=[[320, 250], [400, 250], [400, 280], [320, 280]], text="交卷", confidence=0.96),
        ]
        qa = self.ocr_engine._extract_qa_structure(blocks)

        # 选项中只有4个，不能把“下一题”当成选项
        self.assertEqual(len(qa.options), 4)
        self.assertNotIn("下一题", [opt for opt in qa.options])

        # 验证按钮坐标提取成功
        self.assertIsNotNone(qa.next_button_coord)
        self.assertAlmostEqual(qa.next_button_coord[0], 240.0, places=1)
        self.assertAlmostEqual(qa.next_button_coord[1], 265.0, places=1)

        self.assertIsNotNone(qa.submit_button_coord)
        self.assertAlmostEqual(qa.submit_button_coord[0], 360.0, places=1)
        self.assertAlmostEqual(qa.submit_button_coord[1], 265.0, places=1)

    def test_judgment_question_detection(self):
        """测试判断题专精解析与别名映射"""
        blocks = [
            TextBlock(box=[[17, 30], [106, 30], [106, 50], [17, 50]], text="2.判断题", confidence=0.8),
            TextBlock(box=[[19, 81], [603, 81], [603, 104], [19, 104]], text="实现“中国梦”需要每个人都树立远大理想，并为之努力", confidence=0.85),
            TextBlock(box=[[16, 119], [78, 119], [78, 148], [16, 148]], text="奋斗。", confidence=0.72),
            TextBlock(box=[[107, 213], [134, 213], [134, 240], [107, 240]], text="对", confidence=0.42),
            TextBlock(box=[[108, 316], [134, 316], [134, 343], [108, 343]], text="错", confidence=0.50),
        ]
        qa = self.ocr_engine._extract_qa_structure(blocks)

        self.assertIn("2.判断题", qa.question)
        self.assertIn("奋斗。", qa.question)
        self.assertEqual(len(qa.options), 2)
        self.assertIn("A. 对", qa.options)
        self.assertIn("B. 错", qa.options)

        # 验证全别名映射
        for k in ["A", "1", "对", "正确", "TRUE"]:
            self.assertIn(k, qa.options_coords)
            self.assertAlmostEqual(qa.options_coords[k][0], 120.5, places=1)
            self.assertAlmostEqual(qa.options_coords[k][1], 226.5, places=1)

        for k in ["B", "2", "错", "错误", "FALSE"]:
            self.assertIn(k, qa.options_coords)
            self.assertAlmostEqual(qa.options_coords[k][0], 121.0, places=1)
            self.assertAlmostEqual(qa.options_coords[k][1], 329.5, places=1)

    def test_window_chrome_filtering(self):
        """测试过滤窗口顶部状态栏（电量、时间、课程标题、顶部提交作业）"""
        blocks = [
            TextBlock(box=[[10, 5], [40, 5], [40, 18], [10, 18]], text="32", confidence=0.9),
            TextBlock(box=[[50, 5], [90, 5], [90, 18], [50, 18]], text="16:51", confidence=0.9),
            TextBlock(box=[[150, 5], [300, 5], [300, 18], [150, 18]], text="中国特色社会主义的信念", confidence=0.9),
            TextBlock(box=[[450, 5], [520, 5], [520, 18], [450, 18]], text="提交作业", confidence=0.9),
            TextBlock(box=[[17, 30], [106, 30], [106, 50], [17, 50]], text="2.判断题", confidence=0.8),
            TextBlock(box=[[19, 81], [603, 81], [603, 104], [19, 104]], text="某测试题干内容", confidence=0.85),
            TextBlock(box=[[107, 213], [134, 213], [134, 240], [107, 240]], text="对", confidence=0.42),
            TextBlock(box=[[108, 316], [134, 316], [134, 343], [108, 343]], text="错", confidence=0.50),
        ]
        qa = self.ocr_engine._extract_qa_structure(blocks)

        self.assertNotIn("32", qa.question)
        self.assertNotIn("16:51", qa.question)
        self.assertNotIn("中国特色社会主义的信念", qa.question)
        self.assertNotIn("提交作业", qa.question)
        self.assertTrue(qa.question.startswith("2.判断题"))

    def test_llm_judgment_response_parsing(self):
        """测试 LLM 回复不同判断词形式的自动映射"""
        valid_keys = ["A", "B", "1", "2", "对", "错", "正确", "错误"]

        # 形式 1: JSON answer: 对
        self.assertEqual(LLMReasoner.parse_llm_response('{"answer": "对"}', valid_keys), ["对"])
        # 形式 2: JSON answer: A
        self.assertEqual(LLMReasoner.parse_llm_response('{"answer": "A"}', valid_keys), ["A"])
        # 形式 3: JSON answer: 1
        self.assertEqual(LLMReasoner.parse_llm_response('{"answer": 1}', valid_keys), ["1"])
        # 形式 4: 自然语言解释
        reply = "经分析，该说法是正确的，本题答案是对。"
        self.assertEqual(LLMReasoner.parse_llm_response(reply, valid_keys), ["对"])


if __name__ == "__main__":
    unittest.main()


