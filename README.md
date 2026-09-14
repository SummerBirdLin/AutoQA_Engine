# AutoQA_Engine (Screen-QA-Agent)

基于 Python 的本地轻量级屏幕局部自动答题与点击 Agent，专为 **macOS 系统**（适配 Retina 高清屏与多屏环境）设计。

端到端低延迟（$\le 2.5$ 秒），采用毫秒级抓屏、本地轻量 OCR 结构化提取、OpenAI 规范高性价比大模型推理、以及拟人化鼠标点击闭环。

---

## 目录结构

```text
AutoQA_Engine/
├── config/
│   └── settings.yaml          # 系统全局配置（ROI区域、Retina缩放、API Key、热键等）
├── core/
│   ├── capturer.py            # 阶段一：mss 抓屏与 macOS Retina 坐标换算
│   ├── ocr_engine.py          # 阶段二：RapidOCR 封装、BBox解析、选项与坐标映射
│   ├── llm_client.py          # 阶段二：OpenAI 规范接口、严格 JSON 与正则降级容错
│   └── executor.py            # 阶段三：PyAutoGUI 拟人化微抖动点击与急停制动
├── utils/
│   ├── calibrate.py           # 阶段一：Tkinter 半透明交互式拖拽框选标定工具
│   ├── logger.py              # 彩色终端日志与分步毫秒级耗时追踪
│   └── macos_helpers.py       # macOS Retina 比例识别与系统权限自检
├── tests/
│   ├── test_coordinate.py     # 坐标空间换算单测
│   ├── test_ocr_and_llm.py    # OCR 结构化与大模型输出容错单测
│   └── test_workflow_mock.py  # 全链路端到端闭环模拟测试
├── main.py                    # 主程序入口（全局热键、轮询调度与工作流编排）
├── requirements.txt           # 项目依赖
└── 需求文档.md                 # 原始需求说明书
```

---

## 快速开始

### 1. 虚拟环境准备与依赖安装

```bash
# 进入项目目录
cd AutoQA_Engine

# 激活虚拟环境 (已配置完成)
source .venv/bin/activate

# 若在全新机器配置，可执行:
pip install -r requirements.txt
```

### 2. macOS 权限配置说明 (极其重要)

macOS 出于系统安全考虑，需要对运行终端授予权限：
1. **屏幕录制权限 (Screen Recording)**：用于 `mss` 抓取屏幕区域。
   - 路径：`系统设置 -> 隐私与安全性 -> 屏幕录制`，勾选正在使用的终端（如 Terminal / iTerm / VS Code / Cursor）。
2. **辅助功能权限 (Accessibility)**：用于 `pyautogui` 模拟鼠标点击和 `pynput` 监听全局热键。
   - 路径：`系统设置 -> 隐私与安全性 -> 辅助功能`，勾选正在使用的终端。

---

## 使用指南

### 1. 屏幕感兴趣区域 (ROI) 交互式框选

运行标定工具，屏幕将显示半透明遮罩，鼠标左键按住拖拽框选题干和选项所在区域：

```bash
python main.py --calibrate
```

- **[Enter] 或 [空格]**：保存选区至 `config/settings.yaml` 并退出。
- **[Esc]**：放弃操作并退出。

### 2. 配置大模型 API

编辑 `config/settings.yaml`，配置 OpenAI API 规范的大模型（推荐 DeepSeek-V3 / Qwen-Plus 等）：

```yaml
llm:
  base_url: "https://api.deepseek.com/v1"
  api_key: "${DEEPSEEK_API_KEY}" # 或在此直接填入 sk-...
  model: "deepseek-chat"
  timeout_seconds: 2.5
```

也可以通过环境变量导入：
```bash
export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
```

### 3. 截屏策略与模式选择

系统提供两种截屏策略（在 `config/settings.yaml` 中配置）：

1. **应用窗口一键独立截屏 (`mode: "window"`, 默认推荐)**：
   - 基于 macOS 原生 Quartz 接口，**独立抓取指定的应用窗口**（如 Chrome、Safari 或考试客户端），完全屏蔽桌面背景、任务栏及其他通知弹窗的干扰。
   - `target_app: "active"`：按下热键时，**自动抓取当前最前端活跃应用的主窗口**。
   - 也可以指定特定应用（如 `target_app: "Google Chrome"`）。
   - 换算逻辑完全由应用窗口边界动态换算为全局点击点，窗口随意拖动或改变大小均不受影响。

2. **局部感兴趣区域截屏 (`mode: "roi"`)**：
   - 截取由 `python main.py --calibrate` 框选的固定矩形区域。

### 4. 运行模式与常用命令

#### 查看当前可见应用窗口列表
```bash
python main.py --list-windows
```
可查看当前所有运行的应用名称（Owner）、窗口 ID、标题与尺寸。

#### 模式 A：全局热键监听模式 (默认，支持连续答题)
```bash
python main.py
```
- 按 **`<f8>` (或 Fn+F8)**：
  - 自动识别答题并点击目标选项；
  - 自动定位并点击“下一题”按钮；
  - 智能检测页面跳转：若成功跳转则自动继续答下一题；
  - **最后一题安全停止**：若无“下一题”按钮或点击后题目未更新，判定已是最后一题，自动停止！
- 按 **`Ctrl + Q`**：紧急制动（Kill Switch），立即打断任务并安全退出。
- 鼠标快速甩到屏幕任意四角亦可触发 PyAutoGUI 内置 FAILSAFE 紧急制动。

#### 模式 B：针对指定应用窗口临时答题
```bash
# 针对 Google Chrome 窗口
python main.py --window "Google Chrome"

# 演练模式 (Dry-Run，仅计算坐标不产生实际物理点击)
python main.py --window "Google Chrome" --dry-run
```

#### 模式 C：单次测试模式 (测试单次流程)
```bash
python main.py --once --dry-run
```

#### 模式 D：轮询感知模式 (自动感知题目变化并作答)
```bash
python main.py --polling
```

---

## 运行自动化测试

本项目包含涵盖三个阶段所有关键路径的单元测试与端到端模拟测试：

```bash
python -m unittest discover -s tests
```

测试覆盖内容：
- 标准 1.0x、Retina 2.0x 及任意比例屏幕的坐标数学换算精确性；
- OCR 题干抽取、A/B/C/D 标识识别及坐标提取；
- 无字母选项按纵向坐标自上而下顺位排序（1/2/3/4）降级策略；
- LLM 返回标准 JSON、Markdown 块、解释性文本的正则保底抽取；
- 防盲点误触保护与端到端闭环调度。

