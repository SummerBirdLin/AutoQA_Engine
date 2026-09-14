"""
屏幕与应用窗口捕获模块 (Capture Module)
支持两种截屏策略：
1. 【应用窗口一键截屏 (Window Mode)】：利用 macOS Quartz 独立截取指定应用或当前活跃应用窗口，
   避免全屏与桌面其他干扰，自动获取窗口全局逻辑边界与 Retina 缩放比。
2. 【感兴趣区域截屏 (ROI Mode)】：基于 mss 微秒级截取屏幕指定矩形区域。
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2

from utils.logger import logger, TimerContext
from utils.macos_helpers import get_macos_scale_factor, is_macos


def cgimage_to_bgr(cg_image) -> np.ndarray:
    """将 macOS CGImage 转换为 OpenCV BGR numpy 数组"""
    import Quartz
    width = Quartz.CGImageGetWidth(cg_image)
    height = Quartz.CGImageGetHeight(cg_image)
    bytes_per_row = Quartz.CGImageGetBytesPerRow(cg_image)
    data_provider = Quartz.CGImageGetDataProvider(cg_image)
    data = Quartz.CGDataProviderCopyData(data_provider)
    
    # 4 通道 BGRA/RGBA 转 numpy
    arr = np.frombuffer(data, dtype=np.uint8).reshape((height, bytes_per_row // 4, 4))
    arr = arr[:, :width, :]  # 裁剪右侧字节对齐留白
    return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)


def list_visible_windows() -> List[Dict[str, Any]]:
    """列出 macOS 上当前屏幕所有可见的应用主窗口"""
    if not is_macos():
        return []
    import Quartz
    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    window_list = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)
    results = []
    for w in window_list:
        layer = w.get("kCGWindowLayer", 0)
        owner = w.get("kCGWindowOwnerName", "")
        name = w.get("kCGWindowName", "")
        wid = w.get("kCGWindowNumber", 0)
        bounds = w.get("kCGWindowBounds", {})
        width = bounds.get("Width", 0)
        height = bounds.get("Height", 0)
        # 仅保留普通应用窗口 (layer 0) 且有合理尺寸的窗口
        if layer == 0 and width >= 150 and height >= 150:
            results.append({
                "id": wid,
                "owner": owner,
                "title": name,
                "bounds": bounds,
                "area": width * height,
            })
    return results


def focus_application(app_name: str) -> bool:
    """激活并聚焦指定应用 (支持 macOS 浏览器、考试客户端等)"""
    if not is_macos() or not app_name or app_name.lower() == "active":
        return False
    try:
        from AppKit import NSWorkspace
        workspace = NSWorkspace.sharedWorkspace()
        for app in workspace.runningApplications():
            if app.localizedName() and app_name.lower() in app.localizedName().lower():
                app.activateWithOptions_(1 << 1)  # NSApplicationActivateIgnoringOtherApps
                logger.info(f"✨ 已自动激活并置顶应用窗口: '{app.localizedName()}'")
                return True
    except Exception:
        pass
    try:
        import subprocess
        cmd = f'tell application "{app_name}" to activate'
        res = subprocess.run(["osascript", "-e", cmd], capture_output=True, text=True, timeout=1.0)
        return res.returncode == 0
    except Exception:
        return False


def get_frontmost_app_window() -> Optional[Dict[str, Any]]:
    """获取当前处于最前端活跃应用的主窗口"""
    if not is_macos():
        return None
    try:
        from AppKit import NSWorkspace
        front_app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if not front_app:
            return None
        app_name = front_app.localizedName()
        windows = list_visible_windows()
        # 找到属于该 app 的最大窗口 (主内容窗口)
        matching = [w for w in windows if w["owner"] == app_name]
        if matching:
            matching.sort(key=lambda x: x["area"], reverse=True)
            return matching[0]
    except Exception as e:
        logger.debug(f"获取最前端窗口异常: {e}")
    return None


def find_window_by_app_name(app_name: str) -> Optional[Dict[str, Any]]:
    """根据应用名称 (模糊匹配) 查找主窗口"""
    windows = list_visible_windows()
    target = app_name.lower().strip()
    matching = [
        w for w in windows
        if target in w["owner"].lower() or (w["title"] and target in w["title"].lower())
    ]
    if matching:
        matching.sort(key=lambda x: x["area"], reverse=True)
        return matching[0]
    return None


class ScreenCapturer:
    """屏幕与应用窗口捕获换算器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        capture_cfg = self.config.get("capture", {})
        
        # 捕获模式: "window" (推荐窗口模式), "fullscreen" (全屏浏览器/全屏平台), "roi" (指定局部矩形)
        self.mode = capture_cfg.get("mode", "window")
        self.target_app = capture_cfg.get("target_app", "active")
        self.auto_focus_window = bool(capture_cfg.get("auto_focus_window", True))
        
        self.roi = capture_cfg.get("roi", {"left": 0, "top": 0, "width": 800, "height": 600})
        self.monitor_index = capture_cfg.get("monitor_index", 1)
        
        # 缩放比例
        configured_scale = float(capture_cfg.get("scale_factor", 0.0))
        if configured_scale > 0:
            self.scale_factor = configured_scale
        else:
            self.scale_factor = get_macos_scale_factor()
        
        self._sct = None
        logger.info(
            f"🖥️  屏幕捕获初始化完成 | 模式: {self.mode} | "
            f"目标应用: {self.target_app} | 缩放比: {self.scale_factor:.2f}"
        )

    def _get_sct(self):
        """延迟导入与创建 mss 实例"""
        if self._sct is None:
            import mss
            self._sct = mss.mss()
        return self._sct

    def capture(
        self,
        custom_roi: Optional[Dict[str, int]] = None,
        target_app: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        统一捕获入口：根据配置策略自动执行全屏截屏、应用窗口截屏或 ROI 截屏
        """
        mode = self.mode.lower() if self.mode else "window"
        app = target_app or self.target_app

        if self.auto_focus_window and app and app.lower() != "active":
            focus_application(app)

        # 模式 1: 原生全屏截屏 (针对全屏浏览器 Web 平台 / 全屏客户端)
        if mode in ["fullscreen", "screen", "all"]:
            logger.info(f"🖥️  [全屏模式截屏] 捕获显示器 {self.monitor_index} 完整全屏画面...")
            return self.capture_fullscreen(self.monitor_index)

        # 模式 2: 应用窗口独立截屏 (支持普通窗口与全屏窗口)
        if mode == "window" and is_macos():
            target_win = None
            if app and app.lower() != "active":
                target_win = find_window_by_app_name(app)
                if not target_win:
                    logger.warning(f"⚠️  未找到名为 '{app}' 的窗口，尝试获取最前端活跃窗口...")
            
            if not target_win:
                target_win = get_frontmost_app_window()

            if target_win:
                logger.info(
                    f"🪟 [应用窗口截屏] 应用: {target_win['owner']} | "
                    f"标题: {target_win['title'][:35]} | 尺寸: {int(target_win['bounds']['Width'])}x{int(target_win['bounds']['Height'])}"
                )
                try:
                    return self.capture_window(target_win["id"], target_win["bounds"])
                except Exception as e:
                    logger.error(f"❌ 独立窗口截屏失败 ({e})，正在自动无缝降级为全屏截屏...")
                    return self.capture_fullscreen(self.monitor_index)
            else:
                logger.info("ℹ️ 未枚举到独立窗口句柄 (可能处于独立全屏 Space 或 Web 全屏)，自动启动全屏捕获...")
                return self.capture_fullscreen(self.monitor_index)

        # 模式 3: ROI 局部截屏
        return self.capture_roi(custom_roi)

    def capture_fullscreen(self, monitor_idx: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        原生全屏截屏 (适配全屏浏览器 Web 答题平台、独立 Space 全屏、客户端 F11 全屏)
        """
        sct = self._get_sct()
        idx = monitor_idx or self.monitor_index
        mon_idx = min(max(1, idx), len(sct.monitors) - 1)
        mon = sct.monitors[mon_idx]

        with TimerContext("Screen Capture (Fullscreen)"):
            sct_img = sct.grab(mon)
            raw = np.array(sct_img)
            img_bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

        actual_h, actual_w = img_bgr.shape[:2]
        logical_w = float(mon["width"])
        logical_h = float(mon["height"])
        logical_left = float(mon["left"])
        logical_top = float(mon["top"])

        scale_x = actual_w / logical_w if logical_w > 0 else self.scale_factor
        scale_y = actual_h / logical_h if logical_h > 0 else self.scale_factor
        self.scale_factor = scale_x

        metadata = {
            "logical_left": logical_left,
            "logical_top": logical_top,
            "logical_width": logical_w,
            "logical_height": logical_h,
            "physical_width": actual_w,
            "physical_height": actual_h,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "is_fullscreen": True,
        }
        return img_bgr, metadata

    def capture_window(self, window_id: int, bounds: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        独立截取单个应用窗口 (完全剥离桌面其他窗口与背景干扰)
        """
        import Quartz

        with TimerContext("Window Capture (Quartz)"):
            cg_img = Quartz.CGWindowListCreateImage(
                Quartz.CGRectNull,
                Quartz.kCGWindowListOptionIncludingWindow,
                window_id,
                Quartz.kCGWindowImageBoundsIgnoreFraming
            )
            if not cg_img:
                raise RuntimeError(f"无法获取窗口 ID {window_id} 的图像数据")
            img_bgr = cgimage_to_bgr(cg_img)

        actual_h, actual_w = img_bgr.shape[:2]
        logical_left = float(bounds.get("X", 0))
        logical_top = float(bounds.get("Y", 0))
        logical_w = float(bounds.get("Width", actual_w))
        logical_h = float(bounds.get("Height", actual_h))

        scale_x = actual_w / logical_w if logical_w > 0 else self.scale_factor
        scale_y = actual_h / logical_h if logical_h > 0 else self.scale_factor
        self.scale_factor = scale_x

        metadata = {
            "logical_left": logical_left,
            "logical_top": logical_top,
            "logical_width": logical_w,
            "logical_height": logical_h,
            "physical_width": actual_w,
            "physical_height": actual_h,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "window_id": window_id,
        }
        return img_bgr, metadata

    def capture_roi(self, custom_roi: Optional[Dict[str, int]] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        截取指定 ROI 区域图像 (基于 mss)
        """
        roi = custom_roi or self.roi
        left = int(roi["left"])
        top = int(roi["top"])
        width = int(roi["width"])
        height = int(roi["height"])

        with TimerContext("Screen Capture (mss ROI)"):
            sct = self._get_sct()
            monitor_roi = {
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "mon": self.monitor_index,
            }
            sct_img = sct.grab(monitor_roi)
            raw = np.array(sct_img)
            img_bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

        actual_h, actual_w = img_bgr.shape[:2]
        scale_x = actual_w / width if width > 0 else self.scale_factor
        scale_y = actual_h / height if height > 0 else self.scale_factor
        self.scale_factor = scale_x

        metadata = {
            "logical_left": left,
            "logical_top": top,
            "logical_width": width,
            "logical_height": height,
            "physical_width": actual_w,
            "physical_height": actual_h,
            "scale_x": scale_x,
            "scale_y": scale_y,
        }
        return img_bgr, metadata

    @staticmethod
    def calculate_logical_target(
        box_center_x: float,
        box_center_y: float,
        metadata: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        根据 OCR 在图像内部检测到的中心坐标 (物理像素)，
        换算出屏幕上的绝对逻辑坐标 (供 PyAutoGUI 点击)

        公式:
            Target_X = logical_left + (box_center_x / scale_x)
            Target_Y = logical_top + (box_center_y / scale_y)
        """
        scale_x = metadata.get("scale_x", 1.0)
        scale_y = metadata.get("scale_y", 1.0)
        logical_left = metadata.get("logical_left", 0.0)
        logical_top = metadata.get("logical_top", 0.0)

        target_x = logical_left + (box_center_x / scale_x)
        target_y = logical_top + (box_center_y / scale_y)
        return target_x, target_y

    def close(self):
        """释放 mss 资源"""
        if self._sct is not None:
            self._sct.close()
            self._sct = None
