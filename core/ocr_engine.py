"""
识别与坐标定位模块 (OCR & Localization Module)
封装 RapidOCR，解析题目文本、提取选项 A/B/C/D 及其四角包围盒与中心坐标，
支持无选项字母时的自上而下顺位排序降级策略。
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import cv2

from utils.logger import logger, TimerContext


@dataclass
class TextBlock:
    """OCR 识别出的单个文本块"""
    box: List[List[float]]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    text: str
    confidence: float
    center_x: float = 0.0
    center_y: float = 0.0

    def __post_init__(self):
        if self.box and len(self.box) == 4:
            self.center_x = sum(pt[0] for pt in self.box) / 4.0
            self.center_y = sum(pt[1] for pt in self.box) / 4.0


@dataclass
class QAResult:
    """结构化题目解析结果"""
    question: str
    options: List[str]
    # 选项标号到中心坐标 (物理像素，相对于 ROI 图像) 的映射，如 {"A": (120.5, 340.0), ...}
    options_coords: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    text_blocks: List[TextBlock] = field(default_factory=list)
    # 导航与交卷按钮坐标 (物理像素)
    next_button_coord: Optional[Tuple[float, float]] = None
    submit_button_coord: Optional[Tuple[float, float]] = None


class OCREngine:
    """RapidOCR 封装与结构化题目解析引擎"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        ocr_cfg = self.config.get("ocr", {})
        self.min_confidence = float(ocr_cfg.get("min_confidence", 0.35))
        self.save_debug_image = bool(ocr_cfg.get("save_debug_image", False))
        self.debug_image_path = str(ocr_cfg.get("debug_image_path", "logs/debug_ocr.png"))

        with TimerContext("RapidOCR Model Initialization"):
            from rapidocr_onnxruntime import RapidOCR
            self.engine = RapidOCR()
            if hasattr(self.engine, "min_height"):
                self.engine.min_height = 10
        logger.info("🔍 RapidOCR 引擎初始化完成 (CPU 本地推理)")

        # 视觉神经网络辅助检测器 (Microsoft OmniParser)
        try:
            from core.ui_detector import OmniUIDetector
            self.ui_detector = OmniUIDetector(self.config)
        except Exception as e:
            logger.warning(f"⚠️ 未加载 OmniUIDetector: {e}")
            self.ui_detector = None

    def recognize(self, img_bgr: np.ndarray) -> List[TextBlock]:
        """执行 OCR 识别并返回过滤后的 TextBlock 列表"""
        with TimerContext("RapidOCR Inference"):
            # 显式传递 text_score=0.35 与 min_height=10，防止 RapidOCR 内部默认 0.5 过滤掉 '对'/'错' 等小字或单字
            score_thresh = min(self.min_confidence, 0.35)
            raw_res, _ = self.engine(img_bgr, text_score=score_thresh, min_height=10)

        if not raw_res:
            return []

        blocks: List[TextBlock] = []
        for item in raw_res:
            box, text, score = item
            score = float(score)
            text = text.strip()
            if score >= self.min_confidence and text:
                blocks.append(TextBlock(box=box, text=text, confidence=score))

        # 按纵向坐标 (Y) 排序，便于自上而下阅读与解析
        blocks.sort(key=lambda b: (b.center_y, b.center_x))
        return blocks

    def parse_qa(self, img_bgr: np.ndarray) -> QAResult:
        """
        全流程分析题目图像：
        1. 运行 OCR 识别所有文本行及坐标
        2. 识别题干与各选项标识 (A/B/C/D 或 1/2/3/4 或 对/错)
        3. 提取中心物理像素坐标
        4. 若选项未带字母，启动保底顺位映射与卡片检测
        """
        blocks = self.recognize(img_bgr)
        if not blocks:
            logger.warning("OCR 未识别出任何有效文本内容！")
            return QAResult(question="", options=[], options_coords={}, text_blocks=[])

        qa_result = self._extract_qa_structure(blocks, img_bgr)

        # 5. 双轨空间融合：若启用了 OmniParser 视觉目标检测，融合视觉控件与 OCR 文本坐标
        if getattr(self, "ui_detector", None) and self.ui_detector.enabled and img_bgr is not None:
            try:
                h, w = img_bgr.shape[:2]
                stem_max_y = 0.0
                if qa_result.text_blocks:
                    stem_max_y = max(b.center_y for b in qa_result.text_blocks[:min(3, len(qa_result.text_blocks))])

                ui_elements = self.ui_detector.detect_ui_elements(img_bgr)
                qa_result.options_coords = self.ui_detector.fuse_options_with_ocr(
                    img_bgr=img_bgr,
                    ocr_options_coords=qa_result.options_coords,
                    detected_elements=ui_elements,
                    stem_max_y=stem_max_y
                )

                # 视觉保底：若 OCR 未识别到“下一题”文字，检测屏幕右下角区域的可交互按钮
                if qa_result.next_button_coord is None and ui_elements:
                    bottom_right_btns = [
                        e for e in ui_elements
                        if e.center_x > w * 0.60 and e.center_y > h * 0.80 and e.width < w * 0.50
                    ]
                    if bottom_right_btns:
                        best_btn = max(bottom_right_btns, key=lambda e: (e.center_y, e.center_x))
                        qa_result.next_button_coord = (best_btn.center_x, best_btn.center_y)
                        logger.info(
                            f"✨ [OmniParser 视觉补位] OCR 未识别到下一题文字，视觉网络在右下角精准锁定按钮坐标: "
                            f"({best_btn.center_x:.1f}, {best_btn.center_y:.1f})"
                        )
            except Exception as e:
                logger.error(f"❌ OmniParser 空间融合异常: {e}")

        if self.save_debug_image:
            self.draw_debug_image(img_bgr, qa_result, self.debug_image_path)

        return qa_result

    def _extract_qa_structure(self, blocks: List[TextBlock], img_bgr: Optional[np.ndarray] = None) -> QAResult:
        """
        从 OCR 文本块中提取题干和选项映射：
        1. 识别并排除底部控制按钮 (下一题 / 交卷)。
        2. 识别题目起始位置，过滤顶部窗口标题栏、电量、时间等顶栏杂项。
        3. 优先适配 '判断题'：准确定位 对/错 选项坐标及全别名映射 (A/B/1/2/对/错)。
        4. 带字母选项 (A/B/C/D) 提取。
        5. 无字母选项自上而下顺位提取。
        """
        if not blocks:
            return QAResult(question="", options=[], options_coords={}, text_blocks=[])

        # 匹配常见题目标头（例如 "1.单选题"、"【多选题】"、"[单选]"、"2.判断题"、"1、单选(2分)"）
        header_pattern = re.compile(
            r"^(?:第\s*\d+\s*[题步]|【?[单多不]?选[选题]?】?|[（(]?[单多不]?选[选题]?[）)]?|判断题?|填空题?|简答题?|\d+[\.、\s]+[单多不]?选[选题]?|\d+[\.、\s]+判断题?|.*?(?:单选题|多选题|判断题|\d+分))",
            re.IGNORECASE
        )

        # 匹配底部控制按钮 (下一题 / 提交交卷 / 上一题)
        next_btn_pattern = re.compile(
            r"^(?:下一题|下\s*一\s*题|进入下一题|下一道|下一题\s*[>»》›〉]|提交并下一题|下题|Next)$",
            re.IGNORECASE
        )
        submit_btn_pattern = re.compile(
            r"^(?:提交|交卷|完成答题|完成|结束考试|Submit|Finish)$",
            re.IGNORECASE
        )
        prev_btn_pattern = re.compile(
            r"^(?:上一题|上\s*一\s*题|上一道|Prev|Previous)$",
            re.IGNORECASE
        )
        disclaimer_pattern = re.compile(
            r"^(?:温馨提示|提示|注意|说明|注[:：]|答题说明|本题得分)",
            re.IGNORECASE
        )

        next_button_coord: Optional[Tuple[float, float]] = None
        submit_button_coord: Optional[Tuple[float, float]] = None
        control_btn_indices = set()
        footer_indices = set()

        for i, b in enumerate(blocks):
            clean_t = b.text.strip()
            # 剥离各种首尾修饰符与标点 (< > « » ‹ › 《 》 〈 〉 【 】 [ ] ( ))
            core_t = re.sub(r"^[<«‹《〈【\[\(\s]+|[>»›》〉】\]\)\s]+$", "", clean_t)

            if next_btn_pattern.match(clean_t) or next_btn_pattern.match(core_t) or ("下一题" in core_t and len(core_t) <= 6):
                next_button_coord = (b.center_x, b.center_y)
                control_btn_indices.add(i)
                logger.info(f"🔘 检测到【下一题】按钮: '{clean_t}' 中心坐标: ({b.center_x:.1f}, {b.center_y:.1f})")
            elif submit_btn_pattern.match(clean_t) or submit_btn_pattern.match(core_t) or (any(w in core_t for w in ["提交", "交卷"]) and len(core_t) <= 6):
                # 仅当中下部区域时认定为交卷，防止误判顶部导航的‘提交作业’
                if b.center_y > 150:
                    submit_button_coord = (b.center_x, b.center_y)
                    control_btn_indices.add(i)
                    logger.info(f"🔘 检测到【交卷/提交】按钮: '{clean_t}' 中心坐标: ({b.center_x:.1f}, {b.center_y:.1f})")
            elif prev_btn_pattern.match(clean_t) or prev_btn_pattern.match(core_t) or ("上一题" in core_t and len(core_t) <= 6) or core_t in ["题卡", "答题卡"]:
                control_btn_indices.add(i)
            elif disclaimer_pattern.search(clean_t):
                footer_indices.add(i)

        # 将提示语后续连续多行一并归入 footer_indices
        for i in range(len(blocks)):
            if i in footer_indices:
                base_y = blocks[i].center_y
                for j in range(i + 1, len(blocks)):
                    if j not in control_btn_indices and blocks[j].center_y - base_y < 60.0:
                        footer_indices.add(j)

        # --- 步骤 1：顶部状态栏/窗口标题栏去噪 ---
        first_q_idx = -1
        for i, b in enumerate(blocks):
            if i in control_btn_indices:
                continue
            if header_pattern.match(b.text):
                first_q_idx = i
                break

        chrome_indices = set()
        if first_q_idx > 0:
            first_q_y = blocks[first_q_idx].center_y
            for i in range(first_q_idx):
                if i not in control_btn_indices and blocks[i].center_y < first_q_y - 12:
                    chrome_indices.add(i)
                    logger.debug(f"🧹 过滤顶部窗口杂项文本: '{blocks[i].text}' (Y={blocks[i].center_y:.1f})")

        ignore_indices = control_btn_indices | chrome_indices | footer_indices
        valid_blocks = [b for idx, b in enumerate(blocks) if idx not in ignore_indices]

        # --- 步骤 2：判断题专用精准结构解析 ---
        # 优先排查是否包含单选/多选/选择题标头 (防止题干出现"可以判断..."时误判为判断题)
        has_choice_header = any(
            bool(re.search(r"(?:单选|多选|不定项|选择题)", b.text))
            for b in valid_blocks[:4]
        )
        has_judgment_header = any(
            bool(re.search(r"^(?:\d+[\.、:：\s]*)?判断题?", b.text.strip()))
            for b in valid_blocks[:4]
        )

        is_judgment = False
        if not has_choice_header:
            if has_judgment_header:
                is_judgment = True
            else:
                # 若无明确标头，仅当页面存在独立的“对/错”判断词且缺乏长选项时，才认定为判断题
                standalone_judgment_tokens = [
                    b for b in valid_blocks
                    if b.text.strip() in ["对", "错", "正确", "错误", "√", "×"]
                ]
                if len(standalone_judgment_tokens) >= 1:
                    long_blocks = [
                        b for b in valid_blocks
                        if len(b.text.strip()) > 8 and not header_pattern.match(b.text)
                    ]
                    if len(long_blocks) <= 2:
                        is_judgment = True

        if is_judgment:
            return self._extract_judgment_qa(
                blocks=blocks,
                img_bgr=img_bgr,
                control_btn_indices=control_btn_indices,
                chrome_indices=chrome_indices,
                next_button_coord=next_button_coord,
                submit_button_coord=submit_button_coord,
            )

        # --- 步骤 3：常规带字母选项 A/B/C/D 匹配 ---
        letter_pattern = re.compile(r"^\s*([A-Da-d])[\.、:：\s\-)]\s*(.*)$")
        single_letter_pattern = re.compile(r"^\s*([A-Da-d])\s*$")

        letter_matches = []
        for i, b in enumerate(blocks):
            if i in ignore_indices or header_pattern.match(b.text):
                continue
            m = letter_pattern.match(b.text) or single_letter_pattern.match(b.text)
            if m:
                letter = m.group(1).upper()
                content = m.group(2).strip() if m.lastindex and m.lastindex >= 2 else ""
                letter_matches.append((i, letter, content))

        unique_letters = set(item[1] for item in letter_matches)
        # 必须至少识别出两个不同的字母 (例如 A 和 B)，避免题干中偶然出现的单个大写字母被误当成选项
        if len(unique_letters) >= 2:
            first_opt_idx = letter_matches[0][0]
            question_lines = [b.text for idx, b in enumerate(blocks[:first_opt_idx]) if idx not in ignore_indices]
            question_text = " ".join(question_lines).strip()

            options_dict: Dict[str, str] = {}
            options_coords: Dict[str, Tuple[float, float]] = {}
            for idx, letter, content in letter_matches:
                b = blocks[idx]
                options_coords[letter] = (b.center_x, b.center_y)
                # 兼容数字别名
                num_alias = str(ord(letter) - ord('A') + 1)
                options_coords[num_alias] = (b.center_x, b.center_y)
                options_dict[letter] = f"{letter}. {content}".strip() if content else f"{letter}."

            logger.info(f"📝 题干提取 (带字母选项): {question_text[:60]}...")
            logger.info(f"🎯 选项提取 ({len(options_dict)} 项): {list(options_dict.keys())}")
            return QAResult(
                question=question_text,
                options=list(options_dict.values()),
                options_coords=options_coords,
                text_blocks=blocks,
                next_button_coord=next_button_coord,
                submit_button_coord=submit_button_coord,
            )

        # --- 步骤 4：无字母选项智能识别与顺位分界 ---
        logger.info("ℹ️  未识别出标准 A/B/C/D 字母前缀，启动无字母/图标选项智能定位...")
        split_idx = -1

        # 策略 4.1: 优先寻找问号或设问标志
        for i, b in enumerate(blocks):
            if i in ignore_indices:
                continue
            if "？" in b.text or "?" in b.text:
                split_idx = i + 1
                break

        # 策略 4.2: 寻找冒号或典型设问关键词
        if split_idx == -1:
            for i, b in enumerate(blocks):
                if i in ignore_indices:
                    continue
                t = b.text.strip()
                if t.endswith(("：", ":")) or any(k in t for k in ["下列哪", "哪一项", "哪项", "正确的是", "错误的是", "指的是", "是指"]):
                    split_idx = i + 1
                    break

        # 策略 4.3: 寻找句号 (。) 且下一行间距明显扩大的分界点
        if split_idx == -1:
            for i in range(len(blocks) - 1):
                if i in ignore_indices:
                    continue
                t = blocks[i].text.strip()
                if t.endswith(("。", ".")):
                    gap = blocks[i+1].center_y - blocks[i].center_y
                    if gap > 45.0:
                        split_idx = i + 1
                        break

        # 策略 4.4: 若仍未定位，寻找垂直间距突变点
        if split_idx == -1 or split_idx >= len(blocks):
            max_gap = 0
            best_i = 1
            start_search = 1 if len(blocks) > 2 and header_pattern.match(blocks[0].text) else 0
            for i in range(start_search, len(blocks) - 1):
                if i in ignore_indices:
                    continue
                gap = blocks[i+1].center_y - blocks[i].center_y
                if gap > max_gap:
                    max_gap = gap
                    best_i = i + 1
            split_idx = best_i

        q_blocks = [b for idx, b in enumerate(blocks[:split_idx]) if idx not in ignore_indices]
        opt_raw_blocks = [b for idx, b in enumerate(blocks[split_idx:]) if (split_idx + idx) not in ignore_indices]

        # 题干文本
        question_text = " ".join(b.text for b in q_blocks).strip()

        # 对候选选项文本块做多行折行合并
        grouped_options = self._group_option_blocks(opt_raw_blocks)

        labels = ["A", "B", "C", "D", "E", "F"]
        options_dict = {}
        options_coords = {}
        for idx, (opt_text, cx, cy) in enumerate(grouped_options[:6]):
            lbl = labels[idx]
            num_lbl = str(idx + 1)
            options_coords[lbl] = (cx, cy)
            options_coords[num_lbl] = (cx, cy)
            # 剔除可能混入选项首部的单字母与标点 (如 'B 道德的...' -> '道德的...')
            clean_opt = re.sub(rf"^[{lbl}{lbl.lower()}][\.、:：\s\-)]\s*", "", opt_text).strip()
            options_dict[lbl] = f"{lbl}. {clean_opt}" if clean_opt else f"{lbl}. {opt_text}"

        options_list = list(options_dict.values())
        logger.info(f"📝 题干提取: {question_text[:60]}..." if len(question_text) > 60 else f"📝 题干提取: {question_text}")
        logger.info(f"🎯 选项顺位映射 ({len(options_dict)} 项): {list(options_dict.keys())}")

        return QAResult(
            question=question_text,
            options=options_list,
            options_coords=options_coords,
            text_blocks=blocks,
            next_button_coord=next_button_coord,
            submit_button_coord=submit_button_coord,
        )

    def _extract_judgment_qa(
        self,
        blocks: List[TextBlock],
        img_bgr: Optional[np.ndarray],
        control_btn_indices: set,
        chrome_indices: set,
        next_button_coord: Optional[Tuple[float, float]],
        submit_button_coord: Optional[Tuple[float, float]],
    ) -> QAResult:
        """
        判断题专用精准解析器：
        1. 识别或推导 '对/正确' 与 '错/错误' 的中心点击坐标
        2. 若 OCR 文字缺失，结合底层卡片轮廓或几何推导补齐
        3. 自动生成 A/B/1/2/对/错/正确/错误 全别名映射
        """
        coord_true: Optional[Tuple[float, float]] = None
        coord_false: Optional[Tuple[float, float]] = None
        first_opt_y = 99999.0

        # 1. 扫描文字块中显式出现的 对/错 或 A/B 单字母
        for idx, b in enumerate(blocks):
            if idx in control_btn_indices or idx in chrome_indices:
                continue
            t = b.text.strip()
            if "判断题" in t:
                continue

            # 对 / 正确 / A
            if t in ["对", "正确", "√"] or re.match(r"^[A1][\.、:：\s\-)]?\s*(?:对|正确)?$", t, re.IGNORECASE):
                if coord_true is None:
                    coord_true = (b.center_x, b.center_y)
                    first_opt_y = min(first_opt_y, b.center_y)
            # 错 / 错误 / B
            elif t in ["错", "错误", "×"] or re.match(r"^[B2][\.、:：\s\-)]?\s*(?:错|错误)?$", t, re.IGNORECASE):
                if coord_false is None:
                    coord_false = (b.center_x, b.center_y)
                    first_opt_y = min(first_opt_y, b.center_y)

        # 2. 若坐标缺失，尝试卡片轮廓检测保底
        if img_bgr is not None and (coord_true is None or coord_false is None):
            cards = self._detect_option_cards(img_bgr, min_y=120.0)
            if len(cards) >= 2:
                if coord_true is None:
                    coord_true = (cards[0][4], cards[0][5])
                    first_opt_y = min(first_opt_y, cards[0][5])
                if coord_false is None:
                    coord_false = (cards[1][4], cards[1][5])
                    first_opt_y = min(first_opt_y, cards[1][5])

        # 3. 若仍只有单个坐标，按典型行间距 (约 102px) 几何推断另一选项
        if coord_true is not None and coord_false is None:
            coord_false = (coord_true[0], coord_true[1] + 102.0)
        elif coord_false is not None and coord_true is None:
            coord_true = (coord_false[0], max(10.0, coord_false[1] - 102.0))
            first_opt_y = min(first_opt_y, coord_true[1])

        # 保底默认坐标 (若彻底未检出，以图像中下部居中估算)
        if coord_true is None or coord_false is None:
            h, w = (img_bgr.shape[:2]) if img_bgr is not None else (400, 600)
            coord_true = coord_true or (w * 0.5, h * 0.55)
            coord_false = coord_false or (w * 0.5, h * 0.75)
            first_opt_y = min(first_opt_y, coord_true[1])

        # 4. 提取纯净题干：位于 first_opt_y 上方的有效文字行
        stem_blocks = [
            b for idx, b in enumerate(blocks)
            if idx not in control_btn_indices
            and idx not in chrome_indices
            and b.center_y < first_opt_y - 15.0
        ]
        question_text = " ".join(b.text for b in stem_blocks).strip()

        # 5. 构建全方位别名映射字典 (彻底解决 LLM 回复格式不一造成的防盲点拦截)
        options_coords: Dict[str, Tuple[float, float]] = {}
        for k in ["A", "1", "对", "正确", "TRUE", "T"]:
            options_coords[k] = coord_true
        for k in ["B", "2", "错", "错误", "FALSE", "F"]:
            options_coords[k] = coord_false

        options = ["A. 对", "B. 错"]
        logger.info(f"📝 题干提取 (判断题): {question_text}")
        logger.info(f"🎯 选项提取 (判断题): A(对)={coord_true}, B(错)={coord_false}")

        return QAResult(
            question=question_text,
            options=options,
            options_coords=options_coords,
            text_blocks=blocks,
            next_button_coord=next_button_coord,
            submit_button_coord=submit_button_coord,
        )

    @staticmethod
    def _detect_option_cards(img_bgr: np.ndarray, min_y: float = 100.0) -> List[Tuple[int, int, int, int, float, float]]:
        """检测题干下方的矩形卡片容器 (例如无字单选框/背景卡片)"""
        if img_bgr is None or img_bgr.size == 0:
            return []
        try:
            bg_sample = np.median(img_bgr[:min(30, img_bgr.shape[0]), :min(30, img_bgr.shape[1])], axis=(0, 1))
            diff = np.max(np.abs(img_bgr.astype(np.int32) - bg_sample), axis=-1).astype(np.uint8)
            _, mask = cv2.threshold(diff, 3, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 10))
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cards = []
            min_width = img_bgr.shape[1] * 0.35
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if y >= min_y and w >= min_width and 20 <= h <= 180:
                    cx = float(x + w / 2.0)
                    cy = float(y + h / 2.0)
                    cards.append((x, y, w, h, cx, cy))
            cards.sort(key=lambda c: c[1])
            return cards
        except Exception as e:
            logger.debug(f"卡片轮廓检测跳过: {e}")
            return []

    @staticmethod
    def _group_option_blocks(opt_blocks: List[TextBlock]) -> List[Tuple[str, float, float]]:
        """
        对无字母候选选项块进行智能折行合并：
        如果后续文本块垂直距离显著小于选项间距，则视为前一选项的多行折行内容。
        """
        if not opt_blocks:
            return []

        gaps = [opt_blocks[i+1].center_y - opt_blocks[i].center_y for i in range(len(opt_blocks)-1)]
        wrap_threshold = 42.0
        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            wrap_threshold = min(45.0, avg_gap * 0.5)

        grouped = []
        cur_text = opt_blocks[0].text
        cur_cx = opt_blocks[0].center_x
        cur_cy = opt_blocks[0].center_y

        for i in range(1, len(opt_blocks)):
            b = opt_blocks[i]
            gap = b.center_y - opt_blocks[i-1].center_y
            if gap < wrap_threshold:
                cur_text += " " + b.text
            else:
                grouped.append((cur_text.strip(), cur_cx, cur_cy))
                cur_text = b.text
                cur_cx = b.center_x
                cur_cy = b.center_y

        grouped.append((cur_text.strip(), cur_cx, cur_cy))
        return grouped

    @staticmethod
    def draw_debug_image(img_bgr: np.ndarray, qa_result: QAResult, output_path: str = "logs/debug_ocr.png") -> None:
        """在图像上绘制检测到的文字框、中心点和选项标号，用于排查问题"""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        vis = img_bgr.copy()

        for b in qa_result.text_blocks:
            pts = np.array(b.box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        for opt, (cx, cy) in qa_result.options_coords.items():
            icx, icy = int(cx), int(cy)
            cv2.circle(vis, (icx, icy), 6, (0, 0, 255), -1)
            cv2.putText(vis, opt, (icx + 10, icy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imwrite(output_path, vis)
        logger.debug(f"已保存 OCR 调试可视化图片至: {output_path}")

