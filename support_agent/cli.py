"""CLI: python -m support_agent.cli index data/kb  |  python -m support_agent.cli chat"""
import argparse

from .pipeline import SupportAgent

DEFAULT_EMBED = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    ap = argparse.ArgumentParser(prog="support_agent")
    ap.add_argument("command", choices=["index", "chat"])
    ap.add_argument("docs", nargs="?", default="data/kb", help="documents dir/file (index only)")
    ap.add_argument("--index-dir", default="index")
    ap.add_argument("--embedder", default=DEFAULT_EMBED, help="model name, or 'hashing' for offline mode")
    ap.add_argument("--model", default="llama3.2", help="Ollama model")
    ap.add_argument("--no-llm", action="store_true", help="show retrieved passages only")
    ap.add_argument("--rerank", nargs="?", const="cross-encoder/ms-marco-MiniLM-L-6-v2", default=None)
    ap.add_argument("--top-k", type=int, default=4)
    a = ap.parse_args()

    llm = None
    if a.command == "chat" and not a.no_llm:
        from .generate import OllamaLLM
        llm = OllamaLLM(a.model)
    agent = SupportAgent(a.index_dir, a.embedder, llm, a.rerank)

    if a.command == "index":
        print(f"Indexed {agent.build_index(a.docs)} chunks into {a.index_dir}/")
        return

    agent.load_index()
    print("Support assistant ready. Type 'quit' to exit, 'reset' to clear conversation.")
    while True:
        q = input("\nQuestion> ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        if q.lower() == "reset":
            agent.history.clear()
            continue
        if not q:
            continue
        ans = agent.ask(q, a.top_k)
        print(f"\n{ans.text}\n")
        for s in ans.sources:
            print(f"  [{s['n']}] {s['source']}  (score {s['score']})")


if __name__ == "__main__":
    main()
