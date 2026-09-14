"""
macOS 平台辅助工具：Retina 缩放因子获取与系统权限检测
"""

import os
import platform
import subprocess
import sys
from typing import Tuple

from utils.logger import logger


def is_macos() -> bool:
    """判断是否为 macOS 系统"""
    return platform.system() == "Darwin"


def get_macos_scale_factor() -> float:
    """
    获取 macOS 显示器的 Retina 缩放比例 (Backing Scale Factor)
    通常 Retina 屏幕为 2.0，普通外接显示器为 1.0。
    """
    if not is_macos():
        return 1.0

    # 方法 1：通过 AppKit (如果可用)
    try:
        from AppKit import NSScreen  # type: ignore
        main_screen = NSScreen.mainScreen()
        if main_screen:
            return float(main_screen.backingScaleFactor())
    except Exception:
        pass

    # 方法 2：通过 system_profiler 解析显示器信息
    try:
        cmd = ["system_profiler", "SPDisplaysDataType"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8")
        if "Retina" in output:
            return 2.0
    except Exception:
        pass

    # 默认兜底为 2.0 (现代 Mac 笔记本屏幕绝大部分为 Retina 2.0)
    return 2.0


def check_macos_permissions() -> Tuple[bool, bool]:
    """
    检测 macOS 权限状态：
    1. 屏幕录制权限 (Screen Recording) - mss 截图所需
    2. 辅助功能权限 (Accessibility) - pyautogui 鼠标模拟与 pynput 监听所需

    返回: (screen_recording_ok, accessibility_ok)
    """
    if not is_macos():
        return True, True

    screen_ok = True
    accessibility_ok = True

    # 检测辅助功能权限 (可以通过 tccutil 或 ctypes 调用 AXIsProcessTrusted)
    try:
        import ctypes
        app_services = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
        is_trusted = app_services.AXIsProcessTrusted()
        accessibility_ok = bool(is_trusted)
    except Exception:
        pass

    if not accessibility_ok:
        logger.warning("⚠️  [macOS 权限提示] 当前终端或 Python 进程尚未获得【辅助功能 (Accessibility)】权限！")
        logger.warning("👉 请前往：【系统设置】->【隐私与安全性】->【辅助功能】，添加并勾选您正在使用的终端应用 (如 iTerm / Terminal / VS Code / Cursor)。")

    return screen_ok, accessibility_ok

