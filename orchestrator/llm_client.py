import os
import json
import httpx

PRO_LLM_URL = os.environ.get("PRO_LLM_URL", "https://llm.amuhak.com/v1")
FLASH_LLM_URL = os.environ.get("FLASH_LLM_URL", "https://llm.prnt.ink/v1")
PRO_MODEL = os.environ.get("PRO_MODEL", "unsloth/Qwen3.6-27B")
FLASH_MODEL = os.environ.get("FLASH_MODEL", "unsloth/Qwen3.6-27B")
FLASH_TIMEOUT = int(os.environ.get("FLASH_TIMEOUT_SECONDS", "180"))


class LLMResponse:
    def __init__(self, content: str, timed_out: bool = False, partial: str = ""):
        self.content = content
        self.timed_out = timed_out
        self.partial = partial


class LLMClient:
    def __init__(self, base_url: str, model: str, timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            limits=httpx.Limits(max_connections=10),
        )

    async def invoke(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        on_token=None,
        top_p: float = 1.0,
        top_k: int = 50,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 16384,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
        }

        for attempt in range(3):
            result = await self._request(payload, on_token)
            if not result.timed_out:
                return result

        return LLMResponse(
            content="",
            timed_out=True,
            partial="All 3 attempts timed out",
        )

    async def _request(self, payload: dict, on_token=None) -> LLMResponse:
        payload["stream"] = True
        try:
            async with self.client.stream(
                "POST", "/chat/completions", json=payload
            ) as resp:
                if resp.status_code == 200:
                    content = ""
                    reasoning = ""
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data["choices"][0].get("delta", {})
                                new_content = delta.get("content", "") or ""
                                new_reasoning = delta.get("reasoning_content", "") or ""
                                content += new_content
                                reasoning += new_reasoning
                                if on_token:
                                    if new_reasoning:
                                        on_token(new_reasoning, True)
                                    if new_content:
                                        on_token(new_content, False)
                            except (json.JSONDecodeError, KeyError, IndexError):
                                pass
                    if len(content.strip()) < 10 and reasoning.strip():
                        content = reasoning
                    return LLMResponse(content=content, timed_out=False)
                elif resp.status_code in (524, 504):
                    return LLMResponse(
                        content="",
                        timed_out=True,
                        partial=f"Gateway timeout ({resp.status_code})",
                    )
                else:
                    err_text = await resp.aread()
                    return LLMResponse(
                        content="",
                        timed_out=False,
                        partial=f"HTTP {resp.status_code}: {err_text.decode('utf-8', errors='ignore')[:200]}",
                    )
        except httpx.TimeoutException:
            return LLMResponse(content="", timed_out=True, partial="")
        except Exception as e:
            return LLMResponse(content="", timed_out=False, partial=f"Error: {str(e)}")

    async def invoke_json(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        on_token=None,
        top_p: float = 1.0,
        top_k: int = 50,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 16384,
            "response_format": {"type": "json_object"},
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
        }
        for attempt in range(3):
            result = await self._request_json(payload, on_token)
            if not result.timed_out:
                return result
        return LLMResponse(
            content="", timed_out=True, partial="All 3 attempts timed out"
        )

    async def _request_json(self, payload: dict, on_token=None) -> LLMResponse:
        payload["stream"] = True
        try:
            async with self.client.stream(
                "POST", "/chat/completions", json=payload
            ) as resp:
                if resp.status_code == 200:
                    content = ""
                    reasoning = ""
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data["choices"][0].get("delta", {})
                                new_content = delta.get("content", "") or ""
                                new_reasoning = delta.get("reasoning_content", "") or ""
                                content += new_content
                                reasoning += new_reasoning
                                if on_token:
                                    if new_reasoning:
                                        on_token(new_reasoning, True)
                                    if new_content:
                                        on_token(new_content, False)
                            except (json.JSONDecodeError, KeyError, IndexError):
                                pass
                    if len(content.strip()) < 10 and reasoning.strip():
                        content = reasoning
                    return LLMResponse(content=content, timed_out=False)
                elif resp.status_code in (524, 504):
                    return LLMResponse(
                        content="",
                        timed_out=True,
                        partial=f"Gateway timeout ({resp.status_code})",
                    )
                else:
                    err_text = await resp.aread()
                    return LLMResponse(
                        content="",
                        timed_out=False,
                        partial=f"HTTP {resp.status_code}: {err_text.decode('utf-8', errors='ignore')[:200]}",
                    )
        except httpx.TimeoutException:
            return LLMResponse(content="", timed_out=True, partial="")
        except Exception as e:
            return LLMResponse(content="", timed_out=False, partial=f"Error: {str(e)}")

    async def close(self):
        await self.client.aclose()


pro_client = LLMClient(PRO_LLM_URL, PRO_MODEL, timeout=FLASH_TIMEOUT)
flash_client = LLMClient(FLASH_LLM_URL, FLASH_MODEL, timeout=FLASH_TIMEOUT)