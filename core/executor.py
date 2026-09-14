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

        # 多级 Fallback 降级容错配置
        fb_cfg = act_cfg.get("fallback", {})
        self._enable_geometric_click = bool(fb_cfg.get("enable_geometric_click", True))
        self._enable_keyboard_shortcut = bool(fb_cfg.get("enable_keyboard_shortcut", True))
        self._enable_user_notification = bool(fb_cfg.get("enable_user_notification", True))
        self._manual_grace_period_sec = float(fb_cfg.get("manual_grace_period_sec", 3.0))

        # 开启 PyAutoGUI 安全制动：鼠标甩至屏幕四角立即终止
        pyautogui.FAILSAFE = self.pyautogui_failsafe
        logger.info(
            f"🖱️  动作执行器初始化完成 | FAILSAFE: {self.pyautogui_failsafe} | Jitter: ±{self.jitter_pixels}px | "
            f"Fallback: (几何推测={self.enable_geometric_click}, 键盘按键={self.enable_keyboard_shortcut}, 静默通知={self.enable_user_notification})"
        )

    @property
    def enable_geometric_click(self) -> bool:
        fb_cfg = self.config.get("action", {}).get("fallback", {})
        if "enable_geometric_click" in fb_cfg:
            return bool(fb_cfg["enable_geometric_click"])
        return self._enable_geometric_click

    @enable_geometric_click.setter
    def enable_geometric_click(self, val: bool):
        self._enable_geometric_click = val

    @property
    def enable_keyboard_shortcut(self) -> bool:
        fb_cfg = self.config.get("action", {}).get("fallback", {})
        if "enable_keyboard_shortcut" in fb_cfg:
            return bool(fb_cfg["enable_keyboard_shortcut"])
        return self._enable_keyboard_shortcut

    @enable_keyboard_shortcut.setter
    def enable_keyboard_shortcut(self, val: bool):
        self._enable_keyboard_shortcut = val

    @property
    def enable_user_notification(self) -> bool:
        fb_cfg = self.config.get("action", {}).get("fallback", {})
        if "enable_user_notification" in fb_cfg:
            return bool(fb_cfg["enable_user_notification"])
        return self._enable_user_notification

    @enable_user_notification.setter
    def enable_user_notification(self, val: bool):
        self._enable_user_notification = val

    @property
    def manual_grace_period_sec(self) -> float:
        fb_cfg = self.config.get("action", {}).get("fallback", {})
        if "manual_grace_period_sec" in fb_cfg:
            return float(fb_cfg["manual_grace_period_sec"])
        return self._manual_grace_period_sec

    @manual_grace_period_sec.setter
    def manual_grace_period_sec(self, val: float):
        self._manual_grace_period_sec = val

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
        qa_result: Optional[Any] = None,
        dry_run: bool = False,
        stop_event: Optional[threading.Event] = None,
    ) -> bool:
        """
        根据识别出的目标选项，换算坐标并模拟拟人化鼠标点击；
        若缺少定位框，自动触发多级智能 Fallback (几何推算 -> 键盘快捷键 -> 静默横幅通知)。
        """
        if target_option not in options_coords:
            logger.warning(
                f"⚠️  目标选项 '{target_option}' 未在识别坐标表中检出: {list(options_coords.keys())}，启动 Fallback 容错机制..."
            )
            return self._handle_option_fallback(
                target_option=target_option,
                options_coords=options_coords,
                metadata=metadata,
                qa_result=qa_result,
                dry_run=dry_run,
                stop_event=stop_event,
            )

        raw_box_x, raw_box_y = options_coords[target_option]
        return self.click_coordinate(
            raw_box_x, raw_box_y, metadata,
            label=f"选项 {target_option}",
            dry_run=dry_run,
            stop_event=stop_event
        )

    def _handle_option_fallback(
        self,
        target_option: str,
        options_coords: Dict[str, Tuple[float, float]],
        metadata: Dict[str, float],
        qa_result: Optional[Any] = None,
        dry_run: bool = False,
        stop_event: Optional[threading.Event] = None,
    ) -> bool:
        """
        多级智能降级体系 (当目标选项未能获取精确定位框时触发)：
        1. 几何版面推测点击 (结合已有选项间距或题干估算)
        2. 键盘按键模拟 (发送按键 A/B/C/D)
        3. macOS 桌面横幅静默通知 + 人工预留缓冲等待
        """
        # 策略 1: 几何版面推断点击
        geometric_coord = self._estimate_geometric_coord(target_option, options_coords, metadata, qa_result)
        if self.enable_geometric_click and geometric_coord is not None:
            gx, gy = geometric_coord
            logger.warning(
                f"📐 [Fallback 策略 1: 几何推算点击] 根据题干与版面推断选项 '{target_option}' 坐标: ({gx:.1f}, {gy:.1f})"
            )
            click_ok = self.click_coordinate(
                gx, gy, metadata,
                label=f"选项 {target_option} (几何推测)",
                dry_run=dry_run,
                stop_event=stop_event,
            )
            # 配合策略 2 键盘按键作为双重保险
            if self.enable_keyboard_shortcut and not dry_run:
                self._simulate_key_press(target_option)
            return click_ok

        # 策略 2: 键盘按键模拟
        if self.enable_keyboard_shortcut:
            logger.warning(f"⌨️  [Fallback 策略 2: 键盘按键模拟] 向当前应用发送按键 '{target_option}' 完成勾选")
            if not dry_run:
                self._simulate_key_press(target_option)
            return True

        # 策略 3: macOS 静默横幅通知 + 人工缓冲等待
        if self.enable_user_notification:
            return self._notify_and_wait_user(target_option, dry_run=dry_run, stop_event=stop_event)

        logger.error(f"❌ 所有 Fallback 策略均未启用，放弃本次点击: '{target_option}'")
        return False

    @staticmethod
    def _estimate_geometric_coord(
        target_option: str,
        options_coords: Dict[str, Tuple[float, float]],
        metadata: Dict[str, float],
        qa_result: Optional[Any] = None,
    ) -> Optional[Tuple[float, float]]:
        """根据已有上下文与版面规律估算目标选项的物理像素坐标"""
        # 1. 如果 options_coords 中有其他选项的坐标，利用等距线性推算
        known_opts = []
        for opt_char in ["A", "B", "C", "D", "E", "F"]:
            if opt_char in options_coords:
                known_opts.append((ord(opt_char) - ord("A"), options_coords[opt_char]))

        target_idx = -1
        t_up = target_option.upper()
        if len(t_up) == 1 and "A" <= t_up <= "F":
            target_idx = ord(t_up) - ord("A")
        elif t_up in ["1", "2", "3", "4", "5"]:
            target_idx = int(t_up) - 1
        elif t_up in ["对", "正确"]:
            target_idx = 0
        elif t_up in ["错", "错误"]:
            target_idx = 1

        if target_idx < 0:
            return None

        if len(known_opts) >= 2:
            known_opts.sort(key=lambda item: item[0])
            idx1, (x1, y1) = known_opts[0]
            idx2, (x2, y2) = known_opts[-1]
            if idx2 > idx1:
                step_y = (y2 - y1) / (idx2 - idx1)
                est_x = x1
                est_y = y1 + (target_idx - idx1) * step_y
                return est_x, est_y
        elif len(known_opts) == 1:
            idx1, (x1, y1) = known_opts[0]
            step_y = 55.0  # 典型行间距
            est_x = x1
            est_y = y1 + (target_idx - idx1) * step_y
            return est_x, est_y

        # 2. 若没有任何已知选项坐标，根据题干底部推算
        if qa_result and getattr(qa_result, "text_blocks", None):
            stem_blocks = qa_result.text_blocks
            if stem_blocks:
                stem_max_y = max(b.center_y for b in stem_blocks[:min(4, len(stem_blocks))])
                card_x = metadata.get("physical_width", 800) * 0.3
                card_y = stem_max_y + 55.0 + target_idx * 55.0
                return card_x, card_y

        return None

    @staticmethod
    def _simulate_key_press(target_option: str):
        """向当前聚焦窗口发送按键事件"""
        try:
            key = target_option.lower().strip()
            if key in ["a", "b", "c", "d", "e", "f", "1", "2", "3", "4"]:
                pyautogui.press(key)
                logger.info(f"✅ 已模拟按键: '{key}'")
            elif key in ["对", "正确"]:
                pyautogui.press("a")
            elif key in ["错", "错误"]:
                pyautogui.press("b")
        except Exception as e:
            logger.debug(f"模拟按键失败: {e}")

    def _notify_and_wait_user(
        self,
        target_option: str,
        dry_run: bool = False,
        stop_event: Optional[threading.Event] = None
    ) -> bool:
        """macOS 桌面横幅静默提示 + 终端高亮 + 人工预留等待时长"""
        logger.warning("=" * 60)
        logger.warning(f"⚠️  【定位框缺失】无法获取选项 '{target_option}' 的精准坐标！")
        logger.warning(f"💡 【AI 决策推荐答案】: 👉👉👉  【 {target_option} 】  👈👈👈")
        logger.warning(f"⏳ 正在预留 {self.manual_grace_period_sec:.1f} 秒供您在屏幕上直接手动勾选...")
        logger.warning("=" * 60)

        # 发送 macOS 原生静默横幅桌面通知
        self.send_desktop_notification(
            title="AutoQA 智能答题",
            subtitle=f"💡 推荐答案: 【 {target_option} 】",
            message=f"未获取精确定位框，请在 {int(self.manual_grace_period_sec)} 秒内手动勾选"
        )

        if dry_run:
            logger.info("🧪 [Dry-Run 模式] 跳过人工预留等待。")
            return True

        # 优雅缓冲等待（支持急停打断）
        start_t = time.perf_counter()
        while time.perf_counter() - start_t < self.manual_grace_period_sec:
            if stop_event and stop_event.is_set():
                logger.warning("🛑 人工预留等待期间收到急停信号！")
                return False
            time.sleep(0.1)

        logger.info(f"⏰ 人工预留等待期结束，继续自动化流程...")
        return True

    @staticmethod
    def send_desktop_notification(title: str, subtitle: str, message: str):
        """发送 macOS 桌面横幅通知 (非阻塞、不夺取焦点)"""
        try:
            import subprocess
            clean_sub = subtitle.replace('"', '\\"')
            clean_msg = message.replace('"', '\\"')
            clean_title = title.replace('"', '\\"')
            cmd = f'display notification "{clean_msg}" with title "{clean_title}" subtitle "{clean_sub}" sound name "Tink"'
            subprocess.run(["osascript", "-e", cmd], capture_output=True, text=True, timeout=1.5)
        except Exception as e:
            logger.debug(f"发送系统通知跳过: {e}")

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

    def _notify_user_failure(self, message: str):
        """macOS 终端通知或蜂鸣警告"""
        print("\a", end="", flush=True)
        self.send_desktop_notification("AutoQA 告警", "点击已终止", message)

