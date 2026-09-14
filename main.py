"""
Screen-QA-Agent 主入口程序 (Main Workflow & Scheduler)
整合屏幕捕获、RapidOCR 结构化解析、LLM 智能推理与 PyAutoGUI 拟人化点击。
支持全局热键监听 (F8 触发 / Ctrl+Q 急停) 与可选图像变化轮询触发。
"""

import argparse
import hashlib
import os
import signal
import sys
import threading
import time
from typing import Dict, Any, Optional

import yaml
from pynput import keyboard

from utils.logger import logger, TimerContext
from utils.macos_helpers import check_macos_permissions, is_macos
from utils.calibrate import calibrate_screen
from core.capturer import ScreenCapturer
from core.ocr_engine import OCREngine
from core.llm_client import LLMReasoner
from core.executor import ActionExecutor


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """加载配置并补全默认字段"""
    if not os.path.exists(config_path):
        logger.warning(f"⚠️  未找到配置文件 {config_path}，使用系统默认配置。")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_hotkey(key_str: str) -> str:
    """
    规范化热键字符串为 pynput GlobalHotKeys 格式
    例如:
        'F8' -> '<f8>'
        'ctrl+q' -> '<ctrl>+q'
        'Ctrl+Alt+A' -> '<ctrl>+<alt>+a'
    """
    if not key_str:
        return ""
    import re
    parts = [p.strip() for p in key_str.split("+")]
    normalized = []
    special_names = {
        "ctrl", "alt", "cmd", "shift", "space", "enter", "return",
        "esc", "escape", "tab", "backspace", "delete", "up", "down",
        "left", "right", "home", "end", "page_up", "page_down"
    }
    for p in parts:
        lower = p.lower()
        if lower.startswith("<") and lower.endswith(">"):
            normalized.append(lower)
        elif lower in special_names or re.match(r"^f\d{1,2}$", lower):
            normalized.append(f"<{lower}>")
        else:
            normalized.append(lower)
    return "+".join(normalized)


class AutoQAAgent:
    """全自动答题 Agent 调度中心"""

    def __init__(self, config_path: str = "config/settings.yaml", dry_run: bool = False):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.dry_run = dry_run

        self.capturer = ScreenCapturer(self.config)
        self.ocr_engine = OCREngine(self.config)
        self.llm_reasoner = LLMReasoner(self.config)
        self.executor = ActionExecutor(self.config)

        self.stop_event = threading.Event()
        self.is_busy = False
        self._last_img_hash: Optional[str] = None
        self._last_trigger_time: float = 0.0

    def run_auto_qa_loop(self, once: bool = False) -> bool:
        """
        全自动答题流程：
        支持单题模式或连续点击“下一题”的多题自动答题循环。
        """
        if self.is_busy:
            logger.warning("⏳ 答题任务已在运行中，请勿重复触发。")
            return False

        self.is_busy = True
        self.stop_event.clear()

        auto_next_cfg = self.config.get("auto_next", {})
        auto_next_enabled = bool(auto_next_cfg.get("enabled", True)) and not once
        click_to_next_delay = float(auto_next_cfg.get("click_option_to_next_delay", 0.5))
        transition_timeout = float(auto_next_cfg.get("page_transition_timeout", 2.5))
        max_questions = int(auto_next_cfg.get("max_questions", 100))

        q_count = 0
        overall_success = True
        cached_next_coord: Optional[Tuple[float, float]] = None

        try:
            while q_count < max_questions:
                if self.stop_event.is_set():
                    logger.warning("🛑 任务被急停打断。")
                    break

                q_count += 1
                logger.info("=" * 60)
                logger.info(f"🚀 [正在处理第 {q_count} 题] 开始识别推理...")
                total_start = time.perf_counter()

                # 1. 屏幕或应用窗口抓取
                img_bgr, metadata = self.capturer.capture()
                if img_bgr is None or img_bgr.size == 0:
                    logger.error("❌ 截屏失败，未获取到有效图像数据！")
                    overall_success = False
                    break

                if self.stop_event.is_set():
                    break

                # 2. OCR 识别与题目结构化
                qa_res = self.ocr_engine.parse_qa(img_bgr)
                if not qa_res.question:
                    logger.warning("⚠️  未能从当前画面解析出有效题干内容，停止答题。")
                    overall_success = False
                    break

                if not qa_res.options_coords:
                    logger.warning("⚠️  未提取到选项精确定位框，进入智能 Fallback 答题流程...")

                if self.stop_event.is_set():
                    break

                # 3. LLM 智能决策
                valid_keys = list(qa_res.options_coords.keys()) if qa_res.options_coords else ["A", "B", "C", "D"]
                answers = self.llm_reasoner.solve(qa_res.question, qa_res.options, valid_keys)
                if not answers:
                    logger.error("❌ 模型未返回有效答案，终止答题。")
                    overall_success = False
                    break

                if self.stop_event.is_set():
                    break

                # 4. 执行点击动作 (精准定位点击，或触发多级 Fallback 降级)
                for ans in answers:
                    ok = self.executor.click_option(
                        target_option=ans,
                        options_coords=qa_res.options_coords,
                        metadata=metadata,
                        qa_result=qa_res,
                        dry_run=self.dry_run,
                        stop_event=self.stop_event,
                    )
                    if not ok:
                        overall_success = False
                    time.sleep(0.05)

                total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0
                logger.info(f"🏁 第 {q_count} 题作答完成 | 单题总耗时: {total_elapsed_ms:.2f} ms")

                # 如果为单题模式或未开启连题自动跳转，则本题完成即退出
                if once or not auto_next_enabled:
                    break

                # 5. “下一题”按钮检测与坐标缓存复用
                if qa_res.next_button_coord:
                    cached_next_coord = qa_res.next_button_coord

                target_next_coord = qa_res.next_button_coord or cached_next_coord
                if not target_next_coord:
                    logger.info("🏁 【已到达最后一题】当前画面未检测到“下一题”按钮且无历史缓存，自动安全停止答题闭环！")
                    if qa_res.submit_button_coord:
                        logger.info(f"💡 检测到【交卷/提交】按钮，坐标: {qa_res.submit_button_coord} (请人工确认交卷)")
                    break

                if not qa_res.next_button_coord and cached_next_coord:
                    logger.info(f"💡 [下一题坐标缓存复用] 当前题未检出按钮，直接复用已锁定的坐标: {cached_next_coord}")

                # 停顿片刻，模拟拟人化答题节奏并确保网页选项勾选状态已更新
                if click_to_next_delay > 0:
                    time.sleep(click_to_next_delay)

                if self.stop_event.is_set():
                    break

                # 点击“下一题”
                logger.info(f"👉 正在点击【下一题】按钮 (坐标: {target_next_coord})...")
                self.executor.click_next_button(
                    next_button_coord=target_next_coord,
                    metadata=metadata,
                    dry_run=self.dry_run,
                    stop_event=self.stop_event,
                )

                # 6. 等待并判断是否成功跳转至下一题
                jump_ok = self._wait_for_question_transition(qa_res.question, timeout=transition_timeout)
                if not jump_ok:
                    logger.info("🏁 【页面未更新】点击“下一题”后页面内容未发生变化，判定已是最后一题，自动停止！")
                    break
                else:
                    logger.info("🎉 检测到新题目已加载，继续作答下一题...")

            logger.info("=" * 60)
            logger.info(f"✨ [答题流程结束] 累计完成 {q_count} 道题目作答。")
            return overall_success

        except Exception as e:
            logger.error(f"❌ 运行答题闭环时捕获异常: {e}", exc_info=True)
            return False
        finally:
            self.is_busy = False
            self._last_trigger_time = time.time()

    def run_single_shot(self) -> bool:
        """单题执行模式"""
        return self.run_auto_qa_loop(once=True)

    def _wait_for_question_transition(self, last_question: str, timeout: float = 2.5) -> bool:
        """等待下一题加载并比对题干语义"""
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            if self.stop_event.is_set():
                return False
            time.sleep(0.4)
            try:
                img_bgr, _ = self.capturer.capture()
                new_qa = self.ocr_engine.parse_qa(img_bgr)
                # 题干已提取且与上一题不同
                if new_qa.question and new_qa.question != last_question:
                    return True
            except Exception:
                pass
        return False

    def start_hotkey_listener(self):
        """注册全局热键监听"""
        triggers_cfg = self.config.get("triggers", {})
        raw_trigger = triggers_cfg.get("trigger_hotkey", "<f8>")
        raw_kill = triggers_cfg.get("kill_switch_hotkey", "<ctrl>+q")

        trigger_key_str = normalize_hotkey(raw_trigger)
        kill_key_str = normalize_hotkey(raw_kill)

        logger.info(f"⌨️  正在注册全局热键: [触发: {raw_trigger} ({trigger_key_str})] | [急停退出: {raw_kill} ({kill_key_str})]")

        def on_trigger():
            logger.info(f"🔔 监听到触发热键 ({raw_trigger})！")
            threading.Thread(target=self.run_auto_qa_loop, daemon=True).start()

        def on_kill():
            logger.critical(f"🛑 监听到急停热键 ({raw_kill})！立即停止所有任务并退出。")
            self.stop_event.set()
            os._exit(0)

        hotkey_map = {
            trigger_key_str: on_trigger,
            kill_key_str: on_kill,
        }

        # 使用 GlobalHotKeys 监听组合键
        try:
            with keyboard.GlobalHotKeys(hotkey_map) as h:
                logger.info("🟢 全局热键监听服务已就绪，保持后台运行中... (按 Ctrl+C 退出)")
                h.join()
        except ValueError as e:
            logger.error(f"❌ 热键语法解析错误: {e} (触发键: {raw_trigger}, 急停键: {raw_kill})")
        except Exception as e:
            logger.error(f"❌ 注册全局热键失败: {e}")
            if is_macos():
                logger.warning("👉 若提示权限问题，请在【系统设置】->【隐私与安全性】->【辅助功能】中检查是否已为当前终端授予权限。")

    def start_polling_loop(self):
        """轮询感知模式 (基于图像差分 hash 感知屏幕新题目)"""
        triggers_cfg = self.config.get("triggers", {})
        interval = float(triggers_cfg.get("polling_interval_seconds", 1.0))
        logger.info(f"🔄 启动轮询自动答题模式 | 扫描间隔: {interval}s")

        while not self.stop_event.is_set():
            time.sleep(interval)
            # 点击后 1.5s 冷却，避免翻页过渡期重复识别
            if time.time() - self._last_trigger_time < 1.5:
                continue

            try:
                img_bgr, _ = self.capturer.capture()
                # 计算快速哈希
                h = hashlib.md5(img_bgr.tobytes()).hexdigest()
                if self._last_img_hash is not None and h != self._last_img_hash:
                    logger.info("👀 检测到题目内容更新，自动触发答题！")
                    self.run_single_shot()
                self._last_img_hash = h
            except Exception as e:
                logger.warning(f"轮询扫描异常: {e}")


def print_window_list():
    """打印当前可见应用窗口列表"""
    from core.capturer import list_visible_windows
    windows = list_visible_windows()
    if not windows:
        print("未检测到可见的应用主窗口 (可能缺少 macOS 屏幕录制权限)")
        return
    print("\n" + "=" * 80)
    print(f"{'窗口ID':<10} | {'应用名称 (Owner)':<22} | {'尺寸 (W x H)':<16} | {'窗口标题 (Title)'}")
    print("-" * 80)
    for w in windows:
        bounds = w["bounds"]
        size_str = f"{int(bounds.get('Width', 0))} x {int(bounds.get('Height', 0))}"
        print(f"{w['id']:<10} | {w['owner']:<22} | {size_str:<16} | {w['title'][:32]}")
    print("=" * 80)
    print("💡 提示：您可在 config/settings.yaml 的 target_app 中填入上述应用名称，或运行 main.py --window \"应用名\"\n")


def main():
    parser = argparse.ArgumentParser(description="Screen-QA-Agent 屏幕与应用窗口自动答题与点击引擎")
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="配置文件路径")
    parser.add_argument("--calibrate", action="store_true", help="启动屏幕 ROI 区域标定工具")
    parser.add_argument("--list-windows", action="store_true", help="列出当前所有可见的应用窗口及 ID")
    parser.add_argument("--check-api", action="store_true", help="检测大模型 API 连通性与网络往返延迟")
    parser.add_argument("--window", type=str, default=None, help="临时指定目标应用名称进行一键截屏答题 (如 'Google Chrome')")
    parser.add_argument("--dry-run", action="store_true", help="演练模式：仅识别推理与计算坐标，不产生实际鼠标点击")
    parser.add_argument("--once", action="store_true", help="单次执行答题测试后立即退出")
    parser.add_argument("--polling", action="store_true", help="开启轮询感知自动答题模式")

    args = parser.parse_args()

    # 1. 检查 API 连通性
    if args.check_api:
        cfg = load_config(args.config)
        reasoner = LLMReasoner(cfg)
        ok, msg = reasoner.check_connection()
        if ok:
            logger.info(f"✅ [API 检测成功] {msg}")
        else:
            logger.error(f"❌ [API 检测失败] {msg}")
        return

    # 2. 列出窗口
    if args.list_windows:
        print_window_list()
        return

    # 2. 检查标定模式
    if args.calibrate:
        calibrate_screen(args.config)
        return

    # 3. macOS 权限自检
    if is_macos():
        check_macos_permissions()

    # 4. 初始化 Agent
    agent = AutoQAAgent(config_path=args.config, dry_run=args.dry_run)
    if args.window:
        agent.capturer.mode = "window"
        agent.capturer.target_app = args.window
        logger.info(f"🎯 命令行覆盖目标应用为: '{args.window}'")

    # 5. 处理单次执行模式
    if args.once:
        logger.info("▶️ 启动单次答题测试流程...")
        agent.run_single_shot()
        return

    # 6. 轮询模式 vs 热键常驻模式
    if args.polling:
        threading.Thread(target=agent.start_polling_loop, daemon=True).start()

    # 启动主线程全局热键监听
    agent.start_hotkey_listener()


if __name__ == "__main__":
    main()

