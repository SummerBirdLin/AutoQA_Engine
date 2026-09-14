"""
UI 控件目标检测模块 (UI Object Detection Module)
封装微软 OmniParser-v2.0 (微调 YOLO11m) 视觉神经网络，
用于像素级精准定位 UI 交互控件（单选圆圈、复选框、选项卡片与功能按钮），
与 RapidOCR 形成“文本语义 + 视觉控件锚框”双轨空间融合机制。
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# 避免 Ultralytics 在只读系统目录创建配置
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

from utils.logger import logger, TimerContext


@dataclass
class UIElement:
    """UI 交互元素边界框与属性"""
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    center_x: float
    center_y: float
    width: float
    height: float

    @property
    def is_radio_or_circle(self) -> bool:
        """判定是否为单选圆圈或小型图标按钮 (近正方形或圆形，宽<=85)"""
        aspect = self.width / max(1.0, self.height)
        return 0.6 <= aspect <= 1.6 and self.width <= 85.0 and self.height <= 85.0

    @property
    def is_card(self) -> bool:
        """判定是否为横向铺开的选项大卡片 (宽>=180，长宽比>=2.0)"""
        aspect = self.width / max(1.0, self.height)
        return self.width >= 180.0 and aspect >= 2.0


class OmniUIDetector:
    """微软 OmniParser UI 控件检测器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        vision_cfg = self.config.get("vision", {})

        self.enabled = bool(vision_cfg.get("enable_omniparser", True))
        self.model_path = vision_cfg.get("model_path", "models/omniparser/model.pt")
        self.conf_threshold = float(vision_cfg.get("conf_threshold", 0.20))
        self.imgsz = int(vision_cfg.get("imgsz", 640))
        self.device_cfg = vision_cfg.get("device", "auto")

        self.model = None
        self._device = "cpu"
        self._initialized = False

    def _init_model(self) -> bool:
        """按需惰性加载 YOLO 模型"""
        if self._initialized:
            return self.model is not None

        self._initialized = True
        if not self.enabled:
            logger.info("ℹ️ OmniParser 视觉目标检测未启用 (vision.enable_omniparser: false)")
            return False

        if not os.path.exists(self.model_path):
            logger.warning(f"⚠️ 未检测到 OmniParser 模型权重: '{self.model_path}'，尝试自动下载...")
            ok = self.ensure_model_weights(self.model_path)
            if not ok:
                logger.error("❌ 无法获取 OmniParser 权重，自动降级为纯 OCR 与传统几何模式。")
                return False

        try:
            import torch
            from ultralytics import YOLO

            # 设备选择：
            # 在 macOS 上，CPU 多核推理 (~200ms) 稳定且无需跨设备 NMS 同步；
            # 如果用户明确配置 device="mps"，则尊重用户设置
            if self.device_cfg == "auto":
                if torch.cuda.is_available():
                    self._device = "cuda"
                else:
                    self._device = "cpu"
            else:
                self._device = self.device_cfg

            with TimerContext("OmniParser Model Loading"):
                self.model = YOLO(self.model_path)

            logger.info(
                f"🧠 OmniParser UI 控件检测器就绪 | Model: {os.path.basename(self.model_path)} | "
                f"Device: {self._device} | ImgSz: {self.imgsz} | Conf: {self.conf_threshold}"
            )
            return True
        except Exception as e:
            logger.error(f"❌ 初始化 OmniParser 失败: {e}，降级运行。")
            self.model = None
            return False

    def detect_ui_elements(self, img_bgr: np.ndarray) -> List[UIElement]:
        """
        在图像上执行交互控件目标检测
        返回过滤后的 UI 交互元素列表 (按 Y 轴自上而下排序)
        """
        if not self._init_model() or self.model is None or img_bgr is None:
            return []

        h, w = img_bgr.shape[:2]
        elements: List[UIElement] = []

        try:
            with TimerContext("OmniParser UI Inference"):
                results = self.model.predict(
                    img_bgr,
                    device=self._device,
                    imgsz=self.imgsz,
                    conf=self.conf_threshold,
                    verbose=False,
                )

            for r in results:
                for box in r.boxes:
                    conf = float(box.conf.item())
                    cls_id = int(box.cls.item())
                    cls_name = self.model.names.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    box_w = x2 - x1
                    box_h = y2 - y1

                    # 过滤掉几乎占据全图的底板容器 (宽>95% 且 高>95%)
                    if box_w > w * 0.95 and box_h > h * 0.95:
                        continue

                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    elements.append(
                        UIElement(
                            class_name=cls_name,
                            confidence=conf,
                            bbox=(x1, y1, x2, y2),
                            center_x=cx,
                            center_y=cy,
                            width=box_w,
                            height=box_h,
                        )
                    )

            # 默认按纵向垂直位置自上而下排序
            elements.sort(key=lambda elem: elem.center_y)
            return elements
        except Exception as e:
            logger.error(f"❌ OmniParser 检测异常: {e}")
            return []

    def fuse_options_with_ocr(
        self,
        img_bgr: np.ndarray,
        ocr_options_coords: Dict[str, Tuple[float, float]],
        detected_elements: Optional[List[UIElement]] = None,
        stem_max_y: float = 0.0
    ) -> Dict[str, Tuple[float, float]]:
        """
        双轨空间融合：将 RapidOCR 提取的选项与 OmniParser 检测出的物理控件进行吸附对齐
        1. 优先吸附到对应选项行上的圆形 Radio/单选图标中心；
        2. 若无圆形图标，吸附到对应选项卡片的正中心；
        3. 若 OCR 漏检选项，根据整齐排布的选项卡片自动补齐！
        """
        elements = detected_elements if detected_elements is not None else self.detect_ui_elements(img_bgr)
        if not elements:
            return ocr_options_coords

        fused_coords = dict(ocr_options_coords)
        
        # 筛选题干下方的可点击选项候选 (Radio 或 卡片)
        candidate_radios = [e for e in elements if e.is_radio_or_circle and e.center_y > stem_max_y]
        candidate_cards = [e for e in elements if e.is_card and e.center_y > stem_max_y]

        # 1. 对已有的 OCR 选项坐标进行“物理吸附精调”
        for opt_key, (ox, oy) in list(fused_coords.items()):
            # 优先寻找同行 (垂直距离 <= 35px) 的 Radio 单选圆圈
            matching_radios = [
                r for r in candidate_radios
                if abs(r.center_y - oy) <= 35.0
            ]
            if matching_radios:
                # 取离选项文字最近或最靠左的 Radio
                best_radio = min(matching_radios, key=lambda r: r.center_x)
                fused_coords[opt_key] = (best_radio.center_x, best_radio.center_y)
                continue

            # 其次寻找同行的可点击卡片
            matching_cards = [
                c for c in candidate_cards
                if abs(c.center_y - oy) <= 45.0
            ]
            if matching_cards:
                best_card = matching_cards[0]
                # 点击卡片左侧靠拢区域或卡片中心
                click_x = best_card.bbox[0] + min(60.0, best_card.width * 0.15)
                fused_coords[opt_key] = (click_x, best_card.center_y)

        # 2. 如果缺少选项（例如仅检出 1~2 个，但 OmniParser 检出了整齐的 4 个卡片/Radio），自动补齐！
        standard_keys = ["A", "B", "C", "D"]
        if candidate_cards and len(candidate_cards) in [3, 4, 5]:
            # 按 Y 轴升序排列
            candidate_cards.sort(key=lambda c: c.center_y)
            for idx, card in enumerate(candidate_cards[:4]):
                opt_key = standard_keys[idx]
                if opt_key not in fused_coords:
                    # 寻找该卡片内部是否有对应的单选圆圈
                    inner_radios = [
                        r for r in candidate_radios
                        if card.bbox[1] <= r.center_y <= card.bbox[3]
                    ]
                    if inner_radios:
                        best_r = min(inner_radios, key=lambda r: r.center_x)
                        fused_coords[opt_key] = (best_r.center_x, best_r.center_y)
                    else:
                        click_x = card.bbox[0] + min(60.0, card.width * 0.15)
                        fused_coords[opt_key] = (click_x, card.center_y)
                    logger.info(f"✨ [OmniParser 空间融合补齐] 检出选项 '{opt_key}' 控件坐标: {fused_coords[opt_key]}")

        return fused_coords

    @staticmethod
    def ensure_model_weights(target_path: str = "models/omniparser/model.pt") -> bool:
        """检查并确保模型权重存在，不存在时自动下载"""
        if os.path.exists(target_path) and os.path.getsize(target_path) > 30 * 1024 * 1024:
            return True

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        logger.info("⏳ 正在从镜像站下载微软 OmniParser-v2.0 控件检测模型 (~38.7MB)...")
        
        # 尝试通过 urllib / curl 下载
        url = "https://hf-mirror.com/microsoft/OmniParser-v2.0/resolve/main/icon_detect/model.pt"
        try:
            import urllib.request
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
            )
            with urllib.request.urlopen(req, timeout=30.0) as resp, open(target_path, "wb") as f:
                f.write(resp.read())
            logger.info(f"🎉 OmniParser 权重下载完成: {target_path}")
            return True
        except Exception as e:
            logger.error(f"❌ 自动下载 OmniParser 权重失败: {e}")
            return False
