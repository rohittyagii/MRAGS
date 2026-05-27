from __future__ import annotations

from mrags.cli import validate_lmm, ingest, query


def main() -> None:
    """Programmatic demo: validate, ingest `examples/sample.pdf`, then query.

    Note: place `examples/sample.pdf` in the examples/ folder first.
    """
    print("Validating local LMM...")
    validate_lmm()

    pdf_path = "examples/sample.pdf"
    print(f"Ingesting {pdf_path}...")
    ingest(pdf_path)

    print("Showing retrieved context only:")
    query("Give a one-sentence summary.", context_only=True)

    print("Asking the LMM to generate an answer:")
    query("Give a one-sentence summary.")


if __name__ == "__main__":
    main()
