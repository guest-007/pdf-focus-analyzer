import re
from pathlib import Path

from infra.models import PageDoc


def extract_pages(pdf_path: str, min_chars: int = 50) -> list[PageDoc]:
    """Extract text from each page of a PDF. Skip near-empty pages."""
    import fitz  # PyMuPDF

    pdf_id = Path(pdf_path).stem
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        if len(text) < min_chars:
            continue

        pages.append(
            PageDoc(
                pdf_id=pdf_id,
                page=page_num + 1,  # 1-based
                text=text,
            )
        )

    doc.close()
    return pages
