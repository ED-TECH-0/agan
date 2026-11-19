"""Heading-aware text chunking for retrieval."""
import re
from dataclasses import dataclass

from agan.schemas import Chunk


@dataclass
class ChunkResult:
    id: str
    text: str
    heading_path: str
    tokens: int


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return max(1, len(text) // 4)


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Split text by markdown/ATX headings, returning (heading_path, section_text) pairs."""
    # Match ATX headings (# ## ### etc.) and Setext headings (--- ===)
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    matches = list(heading_pattern.finditer(text))
    if not matches:
        return [("", text.strip())]

    sections = []
    heading_stack: list[tuple[int, str]] = []  # (level, text)

    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading_text = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()

        # Update heading stack
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, heading_text))

        heading_path = " > ".join(h[1] for h in heading_stack)
        sections.append((heading_path, section_text))

    # Handle content before first heading
    if matches[0].start() > 0:
        before = text[:matches[0].start()].strip()
        if before:
            sections.insert(0, ("", before))

    return sections


def _split_paragraphs(text: str, max_tokens: int = 500, overlap: int = 50) -> list[str]:
    """Split text into paragraphs, then combine into chunks respecting token limits."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text] if text.strip() else []

    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)
        if current_tokens + para_tokens > max_tokens and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            # Start new chunk with overlap from end of previous
            overlap_text = "\n\n".join(current_chunk)
            overlap_tokens = _estimate_tokens(overlap_text)
            if overlap_tokens > overlap:
                # Take last paragraphs that fit in overlap
                overlap_paras = []
                overlap_count = 0
                for p in reversed(current_chunk):
                    p_tokens = _estimate_tokens(p)
                    if overlap_count + p_tokens > overlap:
                        break
                    overlap_paras.insert(0, p)
                    overlap_count += p_tokens
                current_chunk = overlap_paras
                current_tokens = overlap_count
            else:
                current_chunk = []
                current_tokens = 0

        current_chunk.append(para)
        current_tokens += para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def chunk_text(
    text: str,
    url: str,
    fetched_at: float,
    trust: float = 0.5,
    version_tag: str | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Chunk text into heading-aware pieces with overlap."""
    sections = _split_by_headings(text)
    all_chunks: list[ChunkResult] = []
    chunk_counter = 0

    for heading_path, section_text in sections:
        if not section_text.strip():
            continue
        sub_chunks = _split_paragraphs(section_text, max_tokens=chunk_size, overlap=chunk_overlap)
        for sub in sub_chunks:
            chunk_counter += 1
            chunk_id = f"{url}#{chunk_counter}"
            all_chunks.append(ChunkResult(
                id=chunk_id,
                text=sub,
                heading_path=heading_path,
                tokens=_estimate_tokens(sub),
            ))

    # Convert to Chunk schema objects
    result = []
    for cr in all_chunks:
        result.append(Chunk(
            id=cr.id,
            url=url,
            title="",  # Will be filled by caller if available
            text=cr.text,
            fetched_at=fetched_at,
            trust=trust,
            version_tag=version_tag,
            tainted=True,
        ))
    return result


def chunk_code_blocks(text: str, url: str, fetched_at: float, trust: float = 0.5) -> list[Chunk]:
    """Extract code blocks as separate chunks, preserving them intact."""
    code_pattern = re.compile(r'```(\w*)\n(.*?)\n```', re.DOTALL)
    chunks = []
    for i, match in enumerate(code_pattern.finditer(text)):
        lang = match.group(1) or "text"
        code = match.group(2)
        chunk_id = f"{url}#code{i}"
        chunks.append(Chunk(
            id=chunk_id,
            url=url,
            title=f"Code block ({lang})",
            text=f"```{lang}\n{code}\n```",
            fetched_at=fetched_at,
            trust=trust,
            version_tag=None,
            tainted=True,
        ))
    return chunks


__all__ = ["ChunkResult", "_estimate_tokens", "chunk_code_blocks", "chunk_text"]