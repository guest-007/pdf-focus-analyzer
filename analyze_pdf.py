"""Focused PDF Analysis Pipeline -- CLI Entry Point."""

import argparse
import sys
from pathlib import Path

from infra.config import default_config
from pipeline import run_pipeline

INPUT_DIR = Path("input")
DEFAULT_FOCUS = INPUT_DIR / "focus.md"


def _read_focus(value: str) -> str:
    """Read focus from a .md file if it exists, otherwise return as inline text."""
    path = Path(value)
    if path.suffix == ".md" and path.exists():
        return path.read_text(encoding="utf-8").strip()
    return value


def _pick_pdf() -> str:
    """List PDFs in input/ and let the user choose interactively."""
    pdfs = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in {INPUT_DIR}/", file=sys.stderr)
        sys.exit(1)

    print(f"\nAvailable PDFs in {INPUT_DIR}/:")
    for i, p in enumerate(pdfs, 1):
        print(f"  {i}) {p.name}")

    choice = input("\nSelect PDF (number): ").strip()
    try:
        idx = int(choice) - 1
        if not 0 <= idx < len(pdfs):
            raise ValueError
    except ValueError:
        print(f"Invalid choice: {choice}", file=sys.stderr)
        sys.exit(1)

    return str(pdfs[idx])


PROVIDERS = [
    ("openai", f"OpenAI ({default_config.openai_chat.model_name})"),
    ("lmstudio", f"LM Studio ({default_config.lmstudio_chat.model_name})"),
]


def _pick_provider() -> str:
    """Let the user choose a provider interactively."""
    print("\nProvider:")
    for i, (_, label) in enumerate(PROVIDERS, 1):
        print(f"  {i}) {label}")

    choice = input("\nSelect provider (number) [1]: ").strip()
    if not choice:
        return PROVIDERS[0][0]
    try:
        idx = int(choice) - 1
        if not 0 <= idx < len(PROVIDERS):
            raise ValueError
    except ValueError:
        print(f"Invalid choice: {choice}", file=sys.stderr)
        sys.exit(1)

    return PROVIDERS[idx][0]


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a PDF with a custom focus prompt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_pdf.py
  python analyze_pdf.py --pdf input/report.pdf
  python analyze_pdf.py --pdf input/report.pdf --focus input/focus_example.md
  python analyze_pdf.py --focus "ESG strategy" --provider lmstudio
        """,
    )
    parser.add_argument(
        "--pdf",
        help="Path to PDF file (default: interactive picker from input/)",
    )
    parser.add_argument(
        "--focus",
        default=str(DEFAULT_FOCUS),
        help="Focus prompt: path to .md file or inline text (default: input/focus.md)",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "lmstudio"],
        help="LLM and embedding provider (default: interactive picker)",
    )
    parser.add_argument(
        "--top-k", type=int, default=20, help="Number of chunks to retrieve"
    )
    parser.add_argument("--out", default="out", help="Output directory")

    args = parser.parse_args()

    # Resolve PDF and provider
    pdf_path = args.pdf or _pick_pdf()
    provider = args.provider or _pick_provider()

    # Resolve focus prompt
    focus_prompt = _read_focus(args.focus)
    if not focus_prompt:
        print(
            f"Error: Focus file '{args.focus}' is empty.\n"
            f"Write your analysis focus in {DEFAULT_FOCUS} or pass --focus 'your prompt'.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        summary = run_pipeline(
            pdf_path=pdf_path,
            focus_prompt=focus_prompt,
            provider=provider,
            top_k=args.top_k,
            output_dir=args.out,
        )

        print("\nKEY FINDINGS:")
        for i, finding in enumerate(summary.key_findings, 1):
            print(f"  {i}. {finding}")
        print(
            f"\nConfidence: {summary.confidence.score:.2f} -- {summary.confidence.why}"
        )

    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    except Exception as e:
        # Unwrap tenacity RetryError to show the actual cause
        cause = e.__cause__ or e
        if hasattr(cause, "last_attempt"):
            cause = cause.last_attempt.exception()
        print(f"\nERROR: {cause}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
