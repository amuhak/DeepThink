from typing import TypedDict, Annotated, Literal
import operator

FlashPrompt = dict  # {"text": str, "type": "prove" | "refute"}
FlashOutput = (
    dict  # {"worker_id": int, "prompt_type": str, "response": str, "timed_out": bool}
)
ExecutionLog = dict  # {"worker_id": int, "code": str, "stdout": str, "stderr": str, "exit_code": int, "timed_out": bool}


class DeepThinkState(TypedDict):
    user_prompt: str
    current_plan: str
    flash_prompts: list[FlashPrompt]

    # operator.add required — Send() workers append results here
    flash_outputs: Annotated[list[FlashOutput], operator.add]
    execution_logs: Annotated[list[ExecutionLog], operator.add]
    evaluation_history: Annotated[list[str], operator.add]

    status: Literal["SOLVED", "RETRY", "PIVOT", "RUNNING"]
    loop_count: int
