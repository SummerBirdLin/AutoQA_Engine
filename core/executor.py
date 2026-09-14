"""
自动化执行模块 (Action Module)
封装 PyAutoGUI 拟人化鼠标平滑移动、微小抖动点击，
内置 FAILSAFE 机制与防盲点保护。
"""

import random
import time
import threading
from typing import Dict, Tuple, Optional, Any
import pyautogui

from utils.logger import logger, TimerContext
from core.capturer import ScreenCapturer


class ActionExecutor:
    """鼠标动作与点击执行器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        act_cfg = self.config.get("action", {})

        self.jitter_pixels = int(act_cfg.get("random_jitter_pixels", 3))
        self.move_dur_min = float(act_cfg.get("move_duration_min", 0.05))
        self.move_dur_max = float(act_cfg.get("move_duration_max", 0.12))
        self.pyautogui_failsafe = bool(act_cfg.get("pyautogui_failsafe", True))
        self.click_delay = float(act_cfg.get("click_after_delay_sec", 0.02))

        # 开启 PyAutoGUI 安全制动：鼠标甩至屏幕四角立即终止
        pyautogui.FAILSAFE = self.pyautogui_failsafe
        logger.info(f"🖱️  动作执行器初始化完成 | FAILSAFE: {self.pyautogui_failsafe} | Jitter: ±{self.jitter_pixels}px")

    def click_coordinate(
        self,
        raw_box_x: float,
        raw_box_y: float,
        metadata: Dict[str, float],
        label: str = "目标点",
        dry_run: bool = False,
        stop_event: Optional[threading.Event] = None,
    ) -> bool:
        """
        换算物理像素坐标并模拟拟人化点击
        """
        if stop_event and stop_event.is_set():
            logger.warning("🛑 收到急停信号，取消点击执行！")
            return False

        # 1. 换算逻辑坐标
        base_x, base_y = ScreenCapturer.calculate_logical_target(raw_box_x, raw_box_y, metadata)

        # 2. 拟人化微随机抖动
        if self.jitter_pixels > 0:
            jitter_x = random.uniform(-self.jitter_pixels, self.jitter_pixels)
            jitter_y = random.uniform(-self.jitter_pixels, self.jitter_pixels)
        else:
            jitter_x, jitter_y = 0.0, 0.0

        target_x = round(base_x + jitter_x, 1)
        target_y = round(base_y + jitter_y, 1)

        move_duration = random.uniform(self.move_dur_min, self.move_dur_max)

        logger.info(
            f"🎯 [{label}] 像素中心: ({raw_box_x:.1f}, {raw_box_y:.1f}) -> "
            f"逻辑点击坐标: ({target_x}, {target_y}) (平滑移动耗时: {move_duration:.3f}s)"
        )

        if dry_run:
            logger.info(f"🧪 [Dry-Run 模式] 仅模拟坐标计算，不执行实际屏幕点击 [{label}]。")
            return True

        # 3. 产生平滑移动与点击
        with TimerContext(f"Click [{label}]"):
            try:
                if stop_event and stop_event.is_set():
                    logger.warning("🛑 点击执行前收到急停信号！")
                    return False

                pyautogui.moveTo(target_x, target_y, duration=move_duration)
                if self.click_delay > 0:
                    time.sleep(self.click_delay)
                pyautogui.click()
                logger.info(f"✅ 成功点击 [{label}] 坐标 ({target_x}, {target_y})")
                return True
            except pyautogui.FailSafeException:
                logger.critical("🛑 触发 PyAutoGUI FAILSAFE 紧急制动 (鼠标已移至屏幕边缘)！")
                return False
            except Exception as e:
                logger.error(f"❌ 鼠标点击执行异常: {e}")
                return False

    def click_option(
        self,
        target_option: str,
        options_coords: Dict[str, Tuple[float, float]],
        metadata: Dict[str, float],
        dry_run: bool = False,
        stop_event: Optional[threading.Event] = None,
    ) -> bool:
        """根据识别出的目标选项，换算坐标并模拟拟人化鼠标点击"""
        if target_option not in options_coords:
            logger.error(f"❌ 【防盲点保护触发】目标选项 '{target_option}' 不在识别坐标表中: {list(options_coords.keys())}")
            logger.error("🛑 放弃本次点击，防止误操作！")
            self._notify_user_failure(f"未定位到选项 {target_option} 坐标，已取消点击")
            return False

        raw_box_x, raw_box_y = options_coords[target_option]
        return self.click_coordinate(
            raw_box_x, raw_box_y, metadata,
            label=f"选项 {target_option}",
            dry_run=dry_run,
            stop_event=stop_event
        )

    def click_next_button(
        self,
        next_button_coord: Tuple[float, float],
        metadata: Dict[str, float],
        dry_run: bool = False,
        stop_event: Optional[threading.Event] = None,
    ) -> bool:
        """点击“下一题”导航按钮"""
        raw_x, raw_y = next_button_coord
        return self.click_coordinate(
            raw_x, raw_y, metadata,
            label="下一题按钮",
            dry_run=dry_run,
            stop_event=stop_event
        )

    @staticmethod
    def _notify_user_failure(message: str):
        """macOS 终端通知或蜂鸣警告"""
        print("\a", end="", flush=True)  # 终端蜂鸣
        # 尝试使用 osascript 发送 macOS 桌面横幅通知
        try:
            import subprocess
            cmd = f'display notification "{message}" with title "AutoQA 告警" subtitle "点击已终止"'
            subprocess.run(["osascript", "-e", cmd], check=False, stderr=subprocess.DEVNULL)
        except Exception:
            pass

