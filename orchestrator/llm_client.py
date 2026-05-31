import os
import json
import re
import httpx
import openai
from openai import AsyncOpenAI


class ToolCallStreamFilter:
    def __init__(self, on_token_callback):
        self.on_token = on_token_callback
        self.state = "NORMAL"  # "NORMAL", "CHECKING_START", "BUFFERING_BLOCK", "CHECKING_END"
        self.buffer = ""
        self.block_buffer = ""

    def process(self, text: str, is_reasoning: bool):
        if not self.on_token:
            return

        for char in text:
            if self.state == "NORMAL":
                if char == "<":
                    self.state = "CHECKING_START"
                    self.buffer = "<"
                else:
                    self.on_token(char, is_reasoning)

            elif self.state == "CHECKING_START":
                self.buffer += char
                target = "<tool_call>"
                if target.startswith(self.buffer):
                    if self.buffer == target:
                        self.state = "BUFFERING_BLOCK"
                        self.block_buffer = "<tool_call>"
                        self.buffer = ""
                else:
                    for c in self.buffer:
                        self.on_token(c, is_reasoning)
                    self.state = "NORMAL"
                    self.buffer = ""

            elif self.state == "BUFFERING_BLOCK":
                self.block_buffer += char
                if char == "<":
                    self.state = "CHECKING_END"
                    self.buffer = "<"

            elif self.state == "CHECKING_END":
                self.buffer += char
                target = "</tool_call>"
                if target.startswith(self.buffer):
                    if self.buffer == target:
                        full_block = self.block_buffer + self.buffer
                        inner_content = self.block_buffer[11:].strip()
                        is_tool_call = False
                        try:
                            parsed = json.loads(inner_content)
                            if isinstance(parsed, dict) and "name" in parsed:
                                is_tool_call = True
                        except Exception:
                            pass
                        
                        if not is_tool_call:
                            # Flush the entire block because it is literal explanation/text
                            for c in full_block:
                                self.on_token(c, is_reasoning)
                        
                        self.buffer = ""
                        self.block_buffer = ""
                        self.state = "NORMAL"
                else:
                    self.block_buffer += self.buffer
                    self.buffer = ""
                    self.state = "BUFFERING_BLOCK"

    def flush(self, is_reasoning: bool = False):
        if self.state == "CHECKING_START" and self.buffer:
            for c in self.buffer:
                self.on_token(c, is_reasoning)
        elif self.state in ["BUFFERING_BLOCK", "CHECKING_END"]:
            for c in self.block_buffer + self.buffer:
                self.on_token(c, is_reasoning)
        self.buffer = ""
        self.block_buffer = ""
        self.state = "NORMAL"


PRO_LLM_URL = os.environ.get("PRO_LLM_URL", "https://llm.amuhak.com/v1")
FLASH_LLM_URL = os.environ.get("FLASH_LLM_URL", "https://llm.prnt.ink/v1")
PRO_MODEL = os.environ.get("PRO_MODEL", "unsloth/Qwen3.6-27B")
FLASH_MODEL = os.environ.get("FLASH_MODEL", "unsloth/Qwen3.6-27B")
FLASH_TIMEOUT = int(os.environ.get("FLASH_TIMEOUT_SECONDS", "180"))


class LLMResponse:
    def __init__(
        self,
        content: str,
        timed_out: bool = False,
        partial: str = "",
        usage: dict | None = None,
        tool_calls: list[dict] = None,
    ):
        self.content = content
        self.timed_out = timed_out
        self.partial = partial
        self.usage = usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.tool_calls = tool_calls or []


class LLMClient:
    def __init__(self, base_url: str, model: str, timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(None, read=float(timeout), connect=10.0),
                limits=httpx.Limits(max_connections=10),
            ),
        )

    async def invoke(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        tool_choice: str = None,
        temperature: float = None,
        on_token=None,
    ) -> LLMResponse:
        return await self._request(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            on_token=on_token,
        )

    async def invoke_json(
        self,
        messages: list[dict],
        temperature: float = None,
        on_token=None,
    ) -> LLMResponse:
        return await self._request(
            messages=messages,
            temperature=temperature,
            on_token=on_token,
            response_format={"type": "json_object"},
        )

    async def _request(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        tool_choice: str = None,
        temperature: float = None,
        on_token=None,
        response_format: dict = None,
    ) -> LLMResponse:
        # Build standard chat completion parameters with minimal explicit overrides
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 16384,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if temperature is not None:
            kwargs["temperature"] = temperature
        if response_format is not None:
            kwargs["response_format"] = response_format
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        debug_mode = os.environ.get("DEBUG_LLM", "0") == "1"
        if debug_mode:
            print(f"[LLM Client] URL: {self.base_url}/chat/completions")
            print(f"[LLM Client] Model: {self.model}")
            print(f"[LLM Client] Messages count: {len(messages)}")

        for attempt in range(3):
            try:
                content = ""
                reasoning = ""
                tool_calls = []
                usage = None

                stream_filter = ToolCallStreamFilter(on_token) if on_token else None
                on_token_fn = stream_filter.process if stream_filter else None

                stream = await self.client.chat.completions.create(**kwargs)
                async for chunk in stream:
                    if not chunk.choices:
                        if hasattr(chunk, "usage") and chunk.usage:
                            usage = chunk.usage.model_dump()
                        continue

                    delta = chunk.choices[0].delta
                    new_content = delta.content or ""
                    
                    # Capture reasoning content (handles OpenAI o1/o3 and Qwen models)
                    new_reasoning = ""
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        new_reasoning = delta.reasoning_content
                    elif delta.model_extra and "reasoning_content" in delta.model_extra:
                        new_reasoning = delta.model_extra["reasoning_content"] or ""

                    content += new_content
                    reasoning += new_reasoning

                    if on_token_fn:
                        if new_reasoning:
                            on_token_fn(new_reasoning, True)
                        if new_content:
                            on_token_fn(new_content, False)

                    # Stream & accumulate tool calls
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            while len(tool_calls) <= idx:
                                tool_calls.append({
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                })
                            curr = tool_calls[idx]
                            if tc.id:
                                curr["id"] = tc.id
                            if tc.type:
                                curr["type"] = tc.type
                            if tc.function:
                                if tc.function.name:
                                    curr["function"]["name"] = tc.function.name
                                if tc.function.arguments:
                                    curr["function"]["arguments"] += tc.function.arguments

                if stream_filter:
                    stream_filter.flush(is_reasoning=False)

                if len(content.strip()) < 10 and reasoning.strip():
                    content = reasoning

                def clean_tool_calls(text: str, actual_tool_calls: list[dict]) -> str:
                    if not actual_tool_calls:
                        return text
                    pattern = r"<tool_call>(.*?)</tool_call>"
                    def replace_fn(match):
                        inner_text = match.group(1).strip()
                        try:
                            parsed = json.loads(inner_text)
                            parsed_name = parsed.get("name")
                            parsed_args = parsed.get("arguments", {})
                            if isinstance(parsed_args, str):
                                try:
                                    parsed_args = json.loads(parsed_args)
                                except Exception:
                                    pass
                            for tc in actual_tool_calls:
                                func_data = tc.get("function", {})
                                if parsed_name == func_data.get("name"):
                                    actual_args = func_data.get("arguments", {})
                                    if isinstance(actual_args, str):
                                        try:
                                            actual_args = json.loads(actual_args)
                                        except Exception:
                                            pass
                                    if parsed_args == actual_args:
                                        return ""
                        except Exception:
                            pass
                        return match.group(0)
                    return re.sub(pattern, replace_fn, text, flags=re.DOTALL)

                content = clean_tool_calls(content, tool_calls)
                reasoning = clean_tool_calls(reasoning, tool_calls)

                return LLMResponse(
                    content=content,
                    timed_out=False,
                    usage=usage,
                    tool_calls=tool_calls,
                )

            except (openai.APITimeoutError, httpx.TimeoutException):
                if attempt == 2:
                    return LLMResponse(
                        content="",
                        timed_out=True,
                        partial="Timeout on all attempts",
                    )
                continue
            except Exception as e:
                return LLMResponse(
                    content=f"[WORKER ERROR: {str(e)}]",
                    timed_out=False,
                    partial=f"Error: {str(e)}",
                )

        return LLMResponse(
            content="",
            timed_out=True,
            partial="All attempts failed",
        )

    async def close(self):
        pass


pro_client = LLMClient(PRO_LLM_URL, PRO_MODEL, timeout=FLASH_TIMEOUT)
flash_client = LLMClient(PRO_LLM_URL, PRO_MODEL, timeout=FLASH_TIMEOUT)