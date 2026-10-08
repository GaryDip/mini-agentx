import json
import os
from pathlib import Path
import time
import tomllib
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


class LLMError(ValueError):
    pass


class DeepSeekClient:
    def __init__(self, config_path="configs/llm.toml"):
        with Path(config_path).open("rb") as file:
            self.config = tomllib.load(file)["llm"]
        if self.config["base_url"] != "https://api.deepseek.com":
            raise ValueError("本教程仅支持官方 DeepSeek endpoint")
        for name in ("max_tokens", "max_attempts", "timeout_seconds"):
            value = self.config[name]
            if type(value) is not int or value < 1:
                raise ValueError(f"llm.{name} 必须是正整数")
        if self.config["max_attempts"] > 2:
            raise ValueError("本阶段调用预算最多两次")
        load_dotenv(Path.cwd() / ".env", override=False)
        self._key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not self._key:
            raise ValueError("缺少 DEEPSEEK_API_KEY，请配置项目根目录 .env")
        self.model = os.environ.get("DEEPSEEK_MODEL", self.config["model"])

    def complete(self, messages):
        payload = {
            "model": self.model, "messages": messages, "stream": False,
            "response_format": {"type": "json_object"},
            "max_tokens": self.config["max_tokens"], "temperature": 0,
            "thinking": {"type": "disabled"},
        }
        request = Request(self.config["base_url"] + "/chat/completions",
                          data=json.dumps(payload).encode(),
                          headers={"Authorization": "Bearer " + self._key,
                                   "Content-Type": "application/json"})
        started = time.monotonic()
        try:
            with urlopen(request, timeout=self.config["timeout_seconds"]) as response:
                body = json.loads(response.read())
        except HTTPError as error:
            # Never echo vendor error body, request headers, or key.
            raise LLMError(f"DeepSeek HTTP {error.code}；检查模型、余额或认证配置") from None
        except (URLError, TimeoutError, OSError):
            raise LLMError("DeepSeek 网络连接或超时失败") from None
        try:
            choice = body["choices"][0]
            if choice["finish_reason"] != "stop":
                raise LLMError("模型输出未正常结束，不能作为有效结论")
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise LLMError("模型返回空内容")
            output = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            raise LLMError("DeepSeek 返回内容无法解析为 JSON") from None
        return output, {"model": body.get("model", self.model),
                        "usage": body.get("usage", {}),
                        "latency_seconds": time.monotonic() - started}
