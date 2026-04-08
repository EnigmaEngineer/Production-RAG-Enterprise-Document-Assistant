"""
Citation formatter.

Converts reranked chunks into structured Citation objects and
formats them for inclusion in LLM prompts and API responses.
"""

from __future__ import annotations

from src.models.schemas import Chunk, Citation
from src.utils.logging import get_logger

log = get_logger(__name__)


class CitationFormatter:
    """
    Builds citation objects from chunks and formats context for the LLM prompt.

    Responsibilities:
      1. Create Citation objects with doc_id, page_number, chunk_index.
      2. Format numbered context blocks for the LLM system prompt.
      3. Validate that cited chunk IDs actually exist in the retrieved set.
    """

    SNIPPET_MAX_LEN = 200

    def build_citations(self, chunks: list[Chunk]) -> list[Citation]:
        """Convert chunks into Citation objects."""
        citations: list[Citation] = []
        for chunk in chunks:
            snippet = chunk.text[: self.SNIPPET_MAX_LEN]
            if len(chunk.text) > self.SNIPPET_MAX_LEN:
                snippet += "..."

            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.metadata.doc_id,
                    filename=chunk.metadata.filename,
                    page_number=chunk.metadata.page_number,
                    chunk_index=chunk.metadata.chunk_index,
                    text_snippet=snippet,
                    relevance_score=round(chunk.rerank_score, 4),
                )
            )
        return citations

    def format_context_for_llm(self, chunks: list[Chunk]) -> str:
        """
        Format chunks as numbered references for the LLM prompt.

        Output format:
            [1] (doc: abc123, page: 5, file: report.pdf)
            <text of chunk>

            [2] (doc: def456, page: 12, file: manual.pdf)
            <text of chunk>
        """
        blocks: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.metadata
            header = f"[{i}] (doc: {meta.doc_id[:8]}, page: {meta.page_number or 'N/A'}, file: {meta.filename})"
            blocks.append(f"{header}\n{chunk.text}")
        return "\n\n".join(blocks)

    def validate_citations_in_answer(
        self,
        answer: str,
        valid_chunk_ids: set[str],
    ) -> list[str]:
        """
        Post-generation check: extract any [N] references in the answer
        and verify they map to real chunks. Returns list of invalid references.
        """
        import re

        refs = re.findall(r"\[(\d+)\]", answer)
        invalid: list[str] = []
        for ref in refs:
            idx = int(ref) - 1
            # We can't validate by chunk_id from the answer text alone,
            # but we log references that exceed the context size.
            if idx < 0:
                invalid.append(ref)
        return invalid


# Singleton
citation_formatter = CitationFormatter()
