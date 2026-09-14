"""
大语言模型推理模块 (LLM Reasoning Module)
封装 OpenAI 规范 API 请求，约束 JSON 格式输出，
包含多重降级解析 (JSON -> Markdown 剥离 -> 正则提取) 与超时控制。
"""

import json
import os
import re
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from openai import OpenAI

from utils.logger import logger, TimerContext


SYSTEM_PROMPT = """你是一个高精度的自动答题助理。
用户将以 JSON 格式提供题目内容和选项。
请你仔细阅读题目，分析选项，直接给出正确答案。

【极其重要】：
1. 必须且只能输出合法的 JSON 格式，严禁输出任何分析、解释或多余文本。
2. 单选题输出格式：{"answer": "C"}
3. 多选题输出格式：{"answer": "AC"} 或 {"answer": ["A", "C"]}
4. 判断题输出格式：{"answer": "A"} (若为"对/正确") 或 {"answer": "B"} (若为"错/错误")
5. 答案字母/标识必须来自给定的选项列表 (如 A/B/C/D 或 对/错 或 1/2)。
"""


class LLMReasoner:
    """大模型推理与决策客户端"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        llm_cfg = self.config.get("llm", {})

        raw_base_url = llm_cfg.get("base_url", "https://api.deepseek.com/v1").strip()
        # 自动纠正用户在 base_url 尾部误填 /chat/completions 的常见错误
        clean_url = raw_base_url.rstrip("/")
        if clean_url.endswith("/chat/completions"):
            clean_url = clean_url[:-len("/chat/completions")].rstrip("/")
        self.base_url = clean_url
        
        # 支持环境变量插值，如 ${DEEPSEEK_API_KEY}
        raw_api_key = str(llm_cfg.get("api_key", "")).strip()
        if raw_api_key.startswith("${") and raw_api_key.endswith("}"):
            env_var = raw_api_key[2:-1]
            self.api_key = os.environ.get(env_var, "")
        else:
            self.api_key = raw_api_key or os.environ.get("OPENAI_API_KEY", "") or os.environ.get("DEEPSEEK_API_KEY", "")

        self.model = llm_cfg.get("model", "deepseek-chat")
        # 默认超时 8.0s，防止公网 API 偶发抖动造成异常中断
        self.timeout_seconds = float(llm_cfg.get("timeout_seconds", 8.0))
        self.temperature = float(llm_cfg.get("temperature", 0.1))

        self.client: Optional[OpenAI] = None
        if self.api_key:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout_seconds,
            )
            logger.info(f"🤖 LLM 客户端就绪 | Model: {self.model} | BaseURL: {self.base_url}")
        else:
            logger.warning("⚠️  未检测到有效的 LLM API_KEY！若需在线推理，请在 config/settings.yaml 或环境变量中配置。")

    def check_connection(self) -> Tuple[bool, str]:
        """连通性与延迟自检"""
        if not self.client:
            return False, "未配置 API_KEY"
        try:
            start = time.perf_counter()
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "ping"}
                ],
                max_tokens=5,
                timeout=5.0
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            reply = resp.choices[0].message.content or ""
            return True, f"连通成功！模型响应: '{reply.strip()}' | 往返耗时: {elapsed_ms:.1f}ms"
        except Exception as e:
            return False, f"连接失败: {e}"

    def solve(self, question: str, options: List[str], valid_keys: Optional[List[str]] = None) -> Optional[List[str]]:
        """
        根据题目和选项进行推理，返回答案标号列表 (如 ["B"] 或 ["A", "C"])
        
        参数:
            question: 题干文本
            options: 选项列表，如 ["A. 苹果", "B. 香蕉"]
            valid_keys: 合法选项标号，如 ["A", "B", "C", "D"]
        """
        if not self.client:
            logger.error("❌ 无法发起推理：未配置 API_KEY！")
            return None

        payload = {
            "question": question,
            "options": options,
        }
        user_content = json.dumps(payload, ensure_ascii=False)

        raw_reply = ""
        with TimerContext("LLM Reasoning API"):
            max_attempts = 2
            for attempt in range(1, max_attempts + 1):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                        ],
                        temperature=self.temperature,
                        response_format={"type": "json_object"} if "deepseek" in self.model or "gpt" in self.model else None,
                    )
                    raw_reply = response.choices[0].message.content or ""
                    if raw_reply.strip():
                        break
                except Exception as e:
                    if attempt < max_attempts:
                        logger.warning(f"⚠️  LLM 请求抖动异常 ({e})，正在自动重试第 {attempt} 次...")
                        time.sleep(0.5)
                        continue
                    else:
                        logger.error(f"❌ LLM 请求失败或超时: {e}")
                        return None

        # 解析模型返回
        answers = self.parse_llm_response(raw_reply, valid_keys)
        logger.info(f"💡 LLM 决策输出答案: {answers} (原始响应: {raw_reply.strip()})")
        return answers

    @staticmethod
    def parse_llm_response(reply: str, valid_keys: Optional[List[str]] = None) -> Optional[List[str]]:
        """
        健壮的降级解析逻辑：
        1. 尝试直接反序列化 JSON
        2. 剥离 Markdown 代码块 ```json ... ``` 后反序列化
        3. 正则匹配提取合法选项字母或判断词 (A-D, 1-4, 对/错)
        """
        if not reply:
            return None

        text = reply.strip()

        # 策略 1 & 2：JSON 解析
        parsed_json = None
        try:
            parsed_json = json.loads(text)
        except Exception:
            clean_text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            clean_text = re.sub(r"\s*```$", "", clean_text).strip()
            try:
                parsed_json = json.loads(clean_text)
            except Exception:
                pass

        if parsed_json and isinstance(parsed_json, dict):
            ans_val = parsed_json.get("answer") or parsed_json.get("answers") or parsed_json.get("result")
            if ans_val is not None:
                if isinstance(ans_val, list):
                    raw_list = [str(x).strip().upper() for x in ans_val if str(x).strip()]
                    return LLMReasoner._filter_valid_keys(raw_list, valid_keys)
                elif isinstance(ans_val, (str, int, bool)):
                    ans_str = str(ans_val).strip()
                    # 匹配英文字母、数字及中文字符 (对/错/正确/错误)
                    extracted = re.findall(r"([A-Za-z0-9]|对|错|正确|错误)", ans_str)
                    if extracted:
                        raw_list = [x.upper() for x in extracted]
                        return LLMReasoner._filter_valid_keys(raw_list, valid_keys)

        # 策略 3：正则匹配保底降级 (从模型冗余解释中定位明确答案)
        logger.warning(f"⚠️  LLM 未按纯 JSON 返回，启动正则保底提取: {text}")
        # 3.1 优先提取显式结论句型 (如 "正确选项是 C"、"答案为 A"、"选择 B")
        pattern_explicit = re.compile(
            r"(?:正确(?:答案|选项)?|选项|答案|选择|应该选)[是为:：\s]*([A-Da-d]|[1-4]|对|错|正确|错误)",
            re.IGNORECASE
        )
        explicit_match = pattern_explicit.search(text)
        if explicit_match:
            cand = explicit_match.group(1).upper()
            return LLMReasoner._filter_valid_keys([cand], valid_keys)

        # 3.2 匹配独立单词边界的大写字母 A-D 或数字 1-4
        found = re.findall(r"\b([A-Da-d]|[1-4])\b", text)
        if not found:
            # 3.3 匹配孤立判断词
            found = re.findall(r"(?:^|[^\w])([A-Da-d]|[1-4]|对|错)(?:[^\w]|$)", text)

        if found:
            res = [x.upper() for x in found]
            return LLMReasoner._filter_valid_keys(res, valid_keys)

        logger.error(f"❌ 无法从模型回复中提取任何有效答案标号: {text}")
        return None

    @staticmethod
    def _filter_valid_keys(tokens: List[str], valid_keys: Optional[List[str]]) -> List[str]:
        """严格按有效选项标号表过滤与映射别名"""
        if not tokens:
            return []
        if not valid_keys:
            # 未提供有效选项列表时保留全部提取结果 (多选兼容)
            return tokens

        valid_upper = [k.upper() for k in valid_keys]
        filtered = []

        # 语义映射字典 (例如输入 '对' 对应 'A' 或 '1' 等)
        semantic_map = {
            "对": ["对", "正确", "A", "1", "TRUE", "T"],
            "正确": ["对", "正确", "A", "1", "TRUE", "T"],
            "TRUE": ["对", "正确", "A", "1", "TRUE", "T"],
            "T": ["对", "正确", "A", "1", "TRUE", "T"],
            "错": ["错", "错误", "B", "2", "FALSE", "F"],
            "错误": ["错", "错误", "B", "2", "FALSE", "F"],
            "FALSE": ["错", "错误", "B", "2", "FALSE", "F"],
            "F": ["错", "错误", "B", "2", "FALSE", "F"],
            "1": ["1", "A", "对", "正确"],
            "2": ["2", "B", "错", "错误"],
            "A": ["A", "1", "对", "正确"],
            "B": ["B", "2", "错", "错误"],
        }

        for t in tokens:
            t_up = t.upper()
            if t_up in valid_upper:
                if t_up not in filtered:
                    filtered.append(t_up)
            else:
                # 检查语义近义映射
                candidates = semantic_map.get(t_up, [])
                for c in candidates:
                    if c in valid_upper:
                        if c not in filtered:
                            filtered.append(c)
                        break

        return filtered if filtered else [tokens[0]]

