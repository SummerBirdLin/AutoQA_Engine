"""
交互式屏幕感兴趣区域 (ROI) 标定工具
基于 Tkinter 实现半透明全屏框选，并将逻辑坐标自动写入 config/settings.yaml
"""

import os
import sys
import tkinter as tk
from typing import Optional, Dict, Any
import yaml

from utils.logger import logger


def load_settings(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_roi_to_settings(roi: Dict[str, int], config_path: str = "config/settings.yaml") -> None:
    """将 ROI 坐标写回 settings.yaml"""
    settings = load_settings(config_path)
    if "capture" not in settings:
        settings["capture"] = {}
    settings["capture"]["roi"] = roi
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(settings, f, allow_unicode=True, sort_keys=False)
    logger.info(f"✅ ROI 区域已成功保存至 {config_path}: {roi}")


class ScreenCalibrator:
    """屏幕框选标定器"""

    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = config_path
        self.root: Optional[tk.Tk] = None
        self.canvas: Optional[tk.Canvas] = None
        self.start_x: int = 0
        self.start_y: int = 0
        self.cur_x: int = 0
        self.cur_y: int = 0
        self.rect_id: Optional[int] = None
        self.text_id: Optional[int] = None
        self.tip_id: Optional[int] = None
        self.selected_roi: Optional[Dict[str, int]] = None

    def run(self) -> Optional[Dict[str, int]]:
        """启动半透明框选界面"""
        self.root = tk.Tk()
        self.root.title("AutoQA Screen ROI Calibrator")

        # 设置全屏与半透明遮罩
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-alpha", 0.35)
        self.root.config(cursor="cross")

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        self.canvas = tk.Canvas(self.root, width=screen_w, height=screen_h, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 提示文案
        tip_text = "【屏幕区域标定模式】\n按住鼠标左键拖拽框选题干与选项区域\n确认选区：按 [Enter] 或 [空格] 保存并退出\n放弃操作：按 [Esc] 取消"
        self.tip_id = self.canvas.create_text(
            screen_w // 2, 80,
            text=tip_text,
            fill="#00FFCC",
            font=("Helvetica", 18, "bold"),
            justify=tk.CENTER
        )

        # 绑定事件
        self.canvas.bind("<Button-1>", self._on_button_press)
        self.canvas.bind("<B1-Motion>", self._on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self._on_button_release)
        self.root.bind("<Return>", self._on_confirm)
        self.root.bind("<space>", self._on_confirm)
        self.root.bind("<Escape>", self._on_cancel)

        logger.info("📐 启动全屏标定界面，请拖拽鼠标框选答题区域...")
        self.root.mainloop()
        return self.selected_roi

    def _on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        if self.text_id:
            self.canvas.delete(self.text_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="#FF3366", width=2, dash=(4, 4)
        )
        self.text_id = self.canvas.create_text(
            self.start_x, max(20, self.start_y - 20),
            text="0 x 0", fill="#FFFF00", font=("Helvetica", 14, "bold"), anchor="nw"
        )

    def _on_move_press(self, event):
        self.cur_x = event.x
        self.cur_y = event.y
        if self.rect_id and self.canvas:
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, self.cur_x, self.cur_y)
            w = abs(self.cur_x - self.start_x)
            h = abs(self.cur_y - self.start_y)
            lx = min(self.start_x, self.cur_x)
            ly = min(self.start_y, self.cur_y)
            info = f"Left: {lx}, Top: {ly} | Size: {w} x {h}"
            self.canvas.coords(self.text_id, lx, max(20, ly - 22))
            self.canvas.itemconfig(self.text_id, text=info)

    def _on_button_release(self, event):
        self.cur_x = event.x
        self.cur_y = event.y
        left = min(self.start_x, self.cur_x)
        top = min(self.start_y, self.cur_y)
        width = abs(self.cur_x - self.start_x)
        height = abs(self.cur_y - self.start_y)

        if width > 10 and height > 10:
            self.selected_roi = {
                "left": int(left),
                "top": int(top),
                "width": int(width),
                "height": int(height),
            }
            logger.info(f"📍 当前已选定 ROI: {self.selected_roi}")
            if self.canvas and self.text_id:
                info = f"选定区域: {self.selected_roi} (按 Enter 保存，Esc 取消)"
                self.canvas.itemconfig(self.text_id, text=info, fill="#00FF66")

    def _on_confirm(self, event):
        if self.selected_roi:
            save_roi_to_settings(self.selected_roi, self.config_path)
        else:
            logger.warning("未检测到有效选区，未做任何修改。")
        if self.root:
            self.root.destroy()

    def _on_cancel(self, event):
        logger.info("已取消标定。")
        self.selected_roi = None
        if self.root:
            self.root.destroy()


def calibrate_screen(config_path: str = "config/settings.yaml") -> Optional[Dict[str, int]]:
    """独立调用标定工具函数"""
    calibrator = ScreenCalibrator(config_path=config_path)
    return calibrator.run()


if __name__ == "__main__":
    calibrate_screen()

