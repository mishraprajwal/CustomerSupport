import numpy as np

from support_agent.embeddings import HashingEmbedder
from support_agent.ingest import chunk_text
from support_agent.pipeline import SupportAgent
from support_agent.retrieval import BM25, HybridRetriever, tokenize

KB = "data/kb"


def make_agent(tmp_path, llm=None):
    a = SupportAgent(str(tmp_path), "hashing", llm)
    a.build_index(KB)
    return a


def test_chunking_keeps_heading_and_limits_size():
    text = "# Title\n\n" + " ".join(f"Sentence number {i}." for i in range(100))
    chunks = chunk_text(text, "x.md", max_chars=200, overlap=40)
    assert len(chunks) > 1
    assert all(c.section == "Title" and len(c.text) <= 200 + len("Title\n") for c in chunks)


def test_tokenizer_keeps_identifiers():
    assert "err-4012" in tokenize("Why ERR-4012?")


def test_bm25_exact_id_match_beats_others():
    docs = [tokenize(t) for t in ["login problem sso", "ERR-4012 declined card", "shipping times"]]
    assert int(np.argmax(BM25(docs).scores(tokenize("ERR-4012")))) == 1


def test_hybrid_finds_error_code_and_policy(tmp_path):
    a = make_agent(tmp_path)
    assert "ERR-4012" in a.chunks[a.retriever.search("ERR-4012", 1)[0].idx].text
    top = a.chunks[a.retriever.search("refund window for unused product purchase", 3)[0].idx]
    assert "Refund" in top.section or "refund" in top.text.lower()


def test_index_roundtrip(tmp_path):
    make_agent(tmp_path)
    b = SupportAgent(str(tmp_path), "hashing")
    b.load_index()
    assert b.ask("What is the express shipping cost?").sources


def test_out_of_scope_refuses_without_llm_call(tmp_path):
    class Boom:
        def chat(self, m):
            raise AssertionError("LLM must not be called")

    a = make_agent(tmp_path, Boom())
    assert "couldn't find" in a.ask("zzzz qqqq xylophone").text


def test_augmented_prompt_contains_context(tmp_path):
    seen = {}

    class Fake:
        def chat(self, m):
            seen["m"] = m
            return "ok [1]"

    a = make_agent(tmp_path, Fake())
    ans = a.ask("How long do refunds take?")
    assert ans.text == "ok [1]" and "[1]" in seen["m"][-1]["content"] and "refund" in seen["m"][-1]["content"].lower()
