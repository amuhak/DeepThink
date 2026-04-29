from langgraph.config import get_stream_writer
from state import DeepThinkState
from llm_client import pro_client

SYNTHESIZER_SYSTEM = """You are the Lead Advisor. Your task is to provide the absolute best, most professional, and most authoritative final response based on the research provided.

CRITICAL RULES:
1. NO META-COMMENTARY: Do NOT output your thinking process, "Mental Checks", "Self-Corrections", or drafting steps.
2. START IMMEDIATELY: Begin your response directly with the answer or the relevant summary.
3. AUTHORITATIVE TONE: Write with confidence and ruthless clarity. 
4. ELITE FORMATTING: Use Markdown headers, bold text, and bullet points to make the information beautiful and easy to scan.
5. NO RAW DATA: Never include worker IDs, raw JSON, or internal status flags.
6. ADDRESS THE USER: Provide a "Final Verdict" or "Conclusion" that directly answers the original prompt.

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

    messages = [
        {"role": "system", "content": SYNTHESIZER_SYSTEM},
        {
            "role": "user",
            "content": f"{history_str}Original Problem: {state['user_prompt']}\n\nResearch Verdict: {state['evaluation_history'][-1]}\n\nRaw Answer: {state.get('final_answer', 'No answer provided.')}",
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
