# -*- coding: utf-8 -*-
"""
Batch PDF -> TXT extractor with optional OCR fallback.

Usage:
  python pdf_to_txt_1.py data/corpus -o txt_out --engine auto --ocr never
  python pdf_to_txt_1.py data/corpus -o txt_out --engine auto --ocr auto
"""

from __future__ import annotations
import argparse, os, re, time
from pathlib import Path
from typing import List

# ---------------- text cleanup helpers ----------------
def dehyphenate(text: str) -> str:
    # join hyphenated line breaks: "oxy-\ngen" -> "oxygen"
    return re.sub(r'(\w)-\n(\w)', r'\1\2', text)

def normalize(text: str, mode: str = "paragraphs") -> str:
    if mode == "raw":
        return text
    if mode == "lines":
        # keep original line breaks, trim trailing spaces
        return "\n".join(ln.rstrip() for ln in text.splitlines())
    # paragraphs: join non-empty lines into paragraphs, keep blank lines as paragraph breaks
    lines = text.splitlines()
    out, buf = [], []
    for ln in lines:
        if ln.strip() == "":
            if buf:
                out.append(" ".join(buf))
                buf = []
            out.append("")  # blank line
        else:
            buf.append(ln.strip())
    if buf:
        out.append(" ".join(buf))
    return "\n".join(out)

# ---------------- backends ----------------
def extract_with_pymupdf(pdf_path: Path) -> List[str]:
    import fitz  # PyMuPDF
    pages = []
    with fitz.open(pdf_path) as doc:
        for p in doc:
            txt = p.get_text("text")
            pages.append(txt)
    return pages

def extract_with_pdfminer(pdf_path: Path) -> List[str]:
    from pdfminer.high_level import extract_text
    raw = extract_text(str(pdf_path)) or ""
    parts = raw.split("\f")
    if parts and parts[-1].strip() == "":
        parts = parts[:-1]
    return parts

def ocr_with_tesseract(pdf_path: Path, dpi: int = 300, lang: str = "eng",
                       tesseract_cmd: str | None = None) -> List[str]:
    from pdf2image import convert_from_path
    import pytesseract
    if tesseract_cmd:
        import os as _os
        if Path(tesseract_cmd).exists():
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        else:
            print(f"[warn] tesseract_cmd not found: {tesseract_cmd}")
    images = convert_from_path(str(pdf_path), dpi=dpi)
    pages = []
    for i, img in enumerate(images, 1):
        txt = pytesseract.image_to_string(img, lang=lang)
        pages.append(txt)
    return pages

# ---------------- main convert ----------------
def convert_one(pdf_path: Path, out_dir: Path, engine: str = "auto",
                ocr_mode: str = "never", join: str = "paragraphs",
                tesseract_cmd: str | None = None, ocr_auto_thresh: int = 80) -> Path:
    """
    ocr_mode: 'never' | 'auto' | 'always'
    ocr_auto_thresh: if auto and avg chars/page < thresh, use OCR
    """
    assert pdf_path.suffix.lower() == ".pdf"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (pdf_path.stem + ".txt")

    t0 = time.time()
    pages = []
    used = None

    # 1) text extraction
    try:
        if engine == "pymupdf":
            pages = extract_with_pymupdf(pdf_path); used = "pymupdf"
        elif engine == "pdfminer":
            pages = extract_with_pdfminer(pdf_path); used = "pdfminer"
        else:  # auto
            try:
                pages = extract_with_pymupdf(pdf_path); used = "pymupdf"
            except Exception:
                pages = extract_with_pdfminer(pdf_path); used = "pdfminer"
    except Exception as e:
        print(f"[warn] text extraction failed on {pdf_path.name} with {engine}: {e}")
        pages = []

    # 2) decide OCR
    need_ocr = (ocr_mode == "always")
    if ocr_mode == "auto":
        total_chars = sum(len(p or "") for p in pages)
        num_pages = max(1, len(pages))
        avg_chars = total_chars // num_pages
        if avg_chars < ocr_auto_thresh:
            need_ocr = True

    # 3) OCR fallback if needed
    if need_ocr:
        try:
            pages = ocr_with_tesseract(pdf_path, tesseract_cmd=tesseract_cmd)
            used = (used + "+ocr") if used else "ocr"
        except Exception as e:
            print(f"[warn] OCR failed on {pdf_path.name}: {e}")

    # 4) post-process and write
    if not pages:
        out_path.write_text("", encoding="utf-8")
        print(f"[done] {pdf_path.name} -> {out_path.name} (empty)  in {time.time()-t0:.2f}s")
        return out_path

    joined = []
    for i, pg in enumerate(pages, 1):
        pg = dehyphenate(pg or "")
        pg = normalize(pg, mode=join)
        joined.append(f"[PAGE {i}]\n{pg}".strip())
    txt = "\n\n" + ("\n\n").join(joined) + "\n"

    out_path.write_text(txt, encoding="utf-8")
    print(f"[done] {pdf_path.name} ({used}) -> {out_path.name}  pages={len(pages)}  in {time.time()-t0:.2f}s")
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="PDF file or directory")
    ap.add_argument("-o", "--out", default="txt_out", help="output directory")
    ap.add_argument("--engine", choices=["auto","pymupdf","pdfminer"], default="auto")
    ap.add_argument("--ocr", choices=["never","auto","always"], default="never")
    ap.add_argument("--join", choices=["paragraphs","lines","raw"], default="paragraphs")
    ap.add_argument("--tess", help="full path to tesseract.exe if needed (Windows)")
    ap.add_argument("--thresh", type=int, default=80, help="avg chars/page threshold to trigger OCR in auto mode")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out)
    if in_path.is_file() and in_path.suffix.lower() == ".pdf":
        convert_one(in_path, out_dir, engine=args.engine, ocr_mode=args.ocr,
                    join=args.join, tesseract_cmd=args.tess, ocr_auto_thresh=args.thresh)
    else:
        pdfs = sorted([p for p in in_path.rglob("*.pdf")])
        if not pdfs:
            print("No PDFs found.")
            return
        for p in pdfs:
            convert_one(p, out_dir, engine=args.engine, ocr_mode=args.ocr,
                        join=args.join, tesseract_cmd=args.tess, ocr_auto_thresh=args.thresh)

if __name__ == "__main__":
    main()
