"""Augmentation + generation."""
SYSTEM_PROMPT = """You are an assistant for customer support executives.
Answer ONLY from the numbered context passages. Cite passages like [1], [2].
If the context does not contain the answer, say you could not find it in the knowledge base \
and suggest escalating. Be concise and give step-by-step instructions when relevant.
Never invent policies, prices, or timelines."""

NO_ANSWER = "I couldn't find this in the knowledge base. Consider escalating to a senior agent or rephrasing the question."


def build_context(passages: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"[{i}] (source: {label})\n{text}" for i, (label, text) in enumerate(passages, 1))


def build_messages(question: str, context: str, history: list[tuple[str, str]] | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for q, a in (history or [])[-3:]:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
    return msgs


class OllamaLLM:
    def __init__(self, model: str = "llama3.2", timeout: int = 180):
        import ollama

        self.model = model
        self.client = ollama.Client(timeout=timeout)

    def chat(self, messages: list[dict]) -> str:
        r = self.client.chat(model=self.model, messages=messages, options={"temperature": 0.1})
        return r["message"]["content"].strip()

    def rewrite(self, question: str, history: list[tuple[str, str]]) -> str:
        """Turn a follow-up ('what about refunds for it?') into a standalone search query."""
        convo = "\n".join(f"Q: {q}\nA: {a[:300]}" for q, a in history[-3:])
        out = self.chat([
            {"role": "system", "content": "Rewrite the last question as a standalone search query. Output only the query."},
            {"role": "user", "content": f"{convo}\n\nLast question: {question}"},
        ])
        return out.splitlines()[0].strip() or question
