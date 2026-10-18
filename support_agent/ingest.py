"""Ingestion: load documents and split them into overlapping, heading-aware chunks."""
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".txt", ".md"}


@dataclass
class Chunk:
    id: int
    text: str
    source: str
    section: str = ""

    def label(self) -> str:
        return f"{self.source} > {self.section}" if self.section else self.source


def load_files(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED)


def _sections(text: str):
    """Yield (heading, body) pairs; markdown headings define sections."""
    heading, buf = "", []
    for line in text.splitlines():
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            if "".join(buf).strip():
                yield heading, "\n".join(buf)
            heading, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        yield heading, "\n".join(buf)


def _split(text: str, max_chars: int, overlap: int) -> list[str]:
    """Pack paragraphs/sentences into chunks of <= max_chars with character overlap."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
        else:
            units.extend(s for s in re.split(r"(?<=[.!?])\s+", para) if s)
    chunks, cur = [], ""
    for u in units:
        while len(u) > max_chars:  # hard-split pathological long sentences
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(u[:max_chars])
            u = u[max_chars - overlap:]
        if cur and len(cur) + len(u) + 1 > max_chars:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cur = (tail[tail.find(" ") + 1:] + " " + u).strip() if tail else u
        else:
            cur = f"{cur} {u}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def chunk_text(text: str, source: str, start_id: int = 0, max_chars: int = 800, overlap: int = 120) -> list[Chunk]:
    out: list[Chunk] = []
    for heading, body in _sections(text):
        for piece in _split(body, max_chars, overlap):
            # Prefixing the heading gives both BM25 and embeddings section context.
            full = f"{heading}\n{piece}" if heading else piece
            out.append(Chunk(start_id + len(out), full, source, heading))
    return out


def ingest(path: str, max_chars: int = 800, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for f in load_files(path):
        chunks += chunk_text(f.read_text(encoding="utf-8"), f.name, len(chunks), max_chars, overlap)
    return chunks
