from infra.models import PageDoc, Chunk
from infra.tokens import estimate_tokens


def _page_at_offset(boundaries: list[tuple[int, int]], char_offset: int) -> int:
    """Return the page number for a given character offset."""
    page = boundaries[0][1]
    for start, page_num in boundaries:
        if start > char_offset:
            break
        page = page_num
    return page


def chunk_pages(
    pages: list[PageDoc],
    target_tokens: int = 400,
    max_tokens: int = 600,
    overlap_fraction: float = 0.10,
) -> list[Chunk]:
    """Chunk page texts into overlapping segments with page metadata."""
    if not pages:
        return []

    target_chars = target_tokens * 4
    overlap_chars = int(target_chars * overlap_fraction)

    # Concatenate all pages, tracking page boundaries by char offset
    full_text = ""
    page_boundaries: list[tuple[int, int]] = []
    for p in pages:
        page_boundaries.append((len(full_text), p.page))
        full_text += p.text + "\n\n"

    pdf_id = pages[0].pdf_id
    chunks: list[Chunk] = []
    pos = 0
    chunk_idx = 0

    while pos < len(full_text):
        end = min(pos + target_chars, len(full_text))

        # Try to break at a paragraph or sentence boundary
        if end < len(full_text):
            search_start = pos + int(target_chars * 0.8)
            para_break = full_text.rfind(
                "\n\n", search_start, end + int(target_chars * 0.2)
            )
            if para_break > search_start:
                end = para_break
            else:
                sent_break = full_text.rfind(". ", search_start, end + 100)
                if sent_break > search_start:
                    end = sent_break + 1

        chunk_text = full_text[pos:end].strip()
        if not chunk_text:
            break

        start_page = _page_at_offset(page_boundaries, pos)
        end_page = _page_at_offset(page_boundaries, max(pos, end - 1))

        chunks.append(
            Chunk(
                pdf_id=pdf_id,
                chunk_id=f"{pdf_id}_chunk_{chunk_idx:04d}",
                text=chunk_text,
                start_page=start_page,
                end_page=end_page,
                token_count=estimate_tokens(chunk_text),
            )
        )
        chunk_idx += 1

        # Advance with overlap
        new_pos = end - overlap_chars
        if new_pos <= pos:
            break  # prevent infinite loop
        pos = new_pos

    return chunks
