from typing import TypedDict, Annotated, Literal
import operator

FlashPrompt = dict  # {"text": str, "type": "prove" | "refute"}
FlashOutput = (
    dict  # {"worker_id": int, "prompt_type": str, "response": str, "timed_out": bool}
)
ExecutionLog = dict  # {"worker_id": int, "code": str, "stdout": str, "stderr": str, "exit_code": int, "timed_out": bool}
UsageStats = dict  # {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}


def reduce_usage(left: UsageStats | None, right: UsageStats | None) -> UsageStats:
    if not left:
        return right or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not right:
        return left
    return {
        "prompt_tokens": left.get("prompt_tokens", 0) + right.get("prompt_tokens", 0),
        "completion_tokens": left.get("completion_tokens", 0)
        + right.get("completion_tokens", 0),
        "total_tokens": left.get("total_tokens", 0) + right.get("total_tokens", 0),
    }


def reduce_pending_pdfs(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    if right is None:
        return left or []
    # If the update is explicitly an empty list, it means we are clearing the queue
    if isinstance(right, list) and len(right) == 0:
        return []
    return (left or []) + right


class DeepThinkState(TypedDict):
    user_prompt: str
    chat_history: list[dict]  # [{"role": str, "content": str}]
    current_plan: str
    flash_prompts: list[FlashPrompt]

    # operator.add required — Send() workers append results here
    flash_outputs: Annotated[list[FlashOutput], operator.add]
    execution_logs: Annotated[list[ExecutionLog], operator.add]
    evaluation_history: Annotated[list[str], operator.add]

    # Global memory across loops - avoid repeating failed searches/URLs
    failed_urls: Annotated[list[str], operator.add]
    failed_queries: Annotated[list[str], operator.add]
    pending_pdfs: Annotated[list[dict], reduce_pending_pdfs]

    status: Literal["SOLVED", "RETRY", "PIVOT", "RUNNING"]
    loop_count: int
    final_answer: str
    usage: Annotated[UsageStats, reduce_usage]
