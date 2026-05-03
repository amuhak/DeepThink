from langgraph.config import get_stream_writer
from state import DeepThinkState
from llm_client import pro_client

SYNTHESIZER_SYSTEM = """You are the Lead Advisor. Your task is to provide the absolute best, most professional, and most authoritative final technical report based on the provided research.

CRITICAL RULES:
1. NO META-COMMENTARY: Do NOT output your thinking process, "Mental Checks", "Self-Corrections", or drafting steps.
2. START IMMEDIATELY: Begin your response directly with the technical report.
3. AUTHORITATIVE TONE: Write with confidence and ruthless clarity. 
4. ELITE FORMATTING: Use Markdown headers, bold text, and bullet points to make the information beautiful and easy to scan.
5. MATHEMATICS: Use LaTeX (e.g., $x^2$ or $$E=mc^2$$) for all formulas.
6. CODE: Include any relevant algorithms or code snippets discovered.
7. SOURCE ATTRIBUTION: Cite sources for key facts discovered by the workers.
8. NO RAW DATA: Never include worker IDs, raw JSON, or internal status flags.
9. CONCLUSION: End with a "Final Verdict" or "Conclusion" that directly answers the original prompt.

You are the final layer of intelligence the user sees. Make it elite."""


async def advisor_synthesizer(state: DeepThinkState) -> dict:
    writer = get_stream_writer()
    writer({"event": "synthesizing"})
    print("[Synthesizer] Starting...", flush=True)

    # IGNORE reasoning tokens for the final synthesis to keep the UI clean
    def on_token(chunk, is_reasoning=False):
        if not is_reasoning:
            writer({"event": "token", "source": "Synthesizer", "text": chunk})

    history_str = ""
    if state.get("chat_history"):
        history = state["chat_history"]
        if history and history[-1]["content"] == state["user_prompt"]:
            history = history[:-1]
        if history:
            history_str = (
                "CONVERSATION HISTORY:\n"
                + "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history])
                + "\n\n"
            )

    worker_results = []
    if state.get("flash_outputs"):
        for i, out in enumerate(state["flash_outputs"]):
            worker_id = out.get("worker_id", i)
            resp = out.get("response", "No output")
            worker_results.append(f"### Worker {worker_id} Findings:\n{resp}")

    critique_str = ""
    if state.get("evaluation_history"):
        critique_str = f"Evaluator's Final Critique: {state['evaluation_history'][-1]}"

    messages = [
        {"role": "system", "content": SYNTHESIZER_SYSTEM},
        {
            "role": "user",
            "content": f"{history_str}Original Problem: {state['user_prompt']}\n\n{critique_str}\n\nRESEARCH FINDINGS:\n" + "\n\n---\n\n".join(worker_results),
        },
    ]

    # Use 'General Thinking' settings for high-quality prose
    resp = await pro_client.invoke(
        messages,
        temperature=1.0,
        on_token=on_token,
        top_p=0.95,
        top_k=20,
        presence_penalty=1.5,
    )

    return {"final_answer": resp.content, "status": "SOLVED"}
