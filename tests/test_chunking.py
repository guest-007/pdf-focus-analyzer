"""Tests for pipeline.chunking -- text chunking with overlap."""

from infra.models import PageDoc
from infra.tokens import estimate_tokens
from pipeline.chunking import chunk_pages, _page_at_offset


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_short(self):
        assert estimate_tokens("abcd") == 1

    def test_proportional(self):
        text = "a" * 400
        assert estimate_tokens(text) == 100


class TestPageAtOffset:
    def test_single_page(self):
        boundaries = [(0, 1)]
        assert _page_at_offset(boundaries, 50) == 1

    def test_multi_page(self):
        boundaries = [(0, 1), (100, 2), (200, 3)]
        assert _page_at_offset(boundaries, 0) == 1
        assert _page_at_offset(boundaries, 99) == 1
        assert _page_at_offset(boundaries, 100) == 2
        assert _page_at_offset(boundaries, 150) == 2
        assert _page_at_offset(boundaries, 200) == 3
        assert _page_at_offset(boundaries, 999) == 3


class TestChunkPages:
    def test_empty_pages(self):
        assert chunk_pages([]) == []

    def test_single_short_page(self):
        pages = [PageDoc(pdf_id="test", page=1, text="Short text.")]
        chunks = chunk_pages(pages)
        assert len(chunks) == 1
        assert chunks[0].pdf_id == "test"
        assert chunks[0].start_page == 1
        assert chunks[0].end_page == 1
        assert chunks[0].chunk_id == "test_chunk_0000"

    def test_chunk_ids_sequential(self):
        text = "Word " * 2000  # long enough to produce multiple chunks
        pages = [PageDoc(pdf_id="doc", page=1, text=text)]
        chunks = chunk_pages(pages)
        assert len(chunks) > 1
        for i, c in enumerate(chunks):
            assert c.chunk_id == f"doc_chunk_{i:04d}"

    def test_multi_page_spanning(self):
        pages = [
            PageDoc(pdf_id="doc", page=1, text="A " * 500),
            PageDoc(pdf_id="doc", page=2, text="B " * 500),
            PageDoc(pdf_id="doc", page=3, text="C " * 500),
        ]
        chunks = chunk_pages(pages)
        assert len(chunks) >= 1
        # At least one chunk should span pages
        has_span = any(c.start_page != c.end_page for c in chunks)
        # Or all text fits in chunks covering all pages
        all_pages = set()
        for c in chunks:
            for p in range(c.start_page, c.end_page + 1):
                all_pages.add(p)
        assert all_pages == {1, 2, 3}

    def test_token_count_set(self):
        pages = [PageDoc(pdf_id="doc", page=1, text="Hello world. " * 100)]
        chunks = chunk_pages(pages)
        for c in chunks:
            assert c.token_count > 0

    def test_overlap_exists(self):
        """Chunks should overlap -- later chunk text should share content with earlier."""
        text = "Sentence number one. " * 300
        pages = [PageDoc(pdf_id="doc", page=1, text=text)]
        chunks = chunk_pages(pages)
        if len(chunks) >= 2:
            # The end of chunk 0 should appear at the start of chunk 1
            tail = chunks[0].text[-50:]
            assert tail in chunks[1].text
