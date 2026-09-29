"""Compile the manuscript and enforce its five-page contract (requires pypdf)."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from pypdf import PdfReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine", help="path to tectonic or pdflatex; autodetected by default"
    )
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    engine = args.engine or shutil.which("tectonic") or shutil.which("pdflatex")
    if not engine:
        raise SystemExit("Install Tectonic or a TeX distribution providing pdflatex.")
    engine = str(Path(engine).resolve()) if Path(engine).exists() else engine
    name = Path(engine).stem.lower()
    build = folder / "build"
    build.mkdir(exist_ok=True)
    if name == "tectonic":
        command = [
            engine,
            "--keep-logs",
            "--keep-intermediates",
            "--outdir",
            str(build),
            "paper.tex",
        ]
        passes = 1  # Tectonic reruns internally to resolve references.
    elif name == "pdflatex":
        command = [
            engine,
            "-no-shell-escape",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory",
            str(build),
            "paper.tex",
        ]
        passes = 2
    else:
        raise SystemExit("Supported engines: tectonic and pdflatex.")
    for _ in range(passes):
        result = subprocess.run(
            command, cwd=folder, capture_output=True, text=True, errors="replace"
        )
        if result.returncode:
            raise SystemExit(result.stdout + result.stderr)
    log = (build / "paper.log").read_text(encoding="utf-8", errors="replace")
    failures = re.findall(
        r"Overfull \\[hv]box[^\n]*|"
        r"[^\n]*(?:undefined references|Citation .* undefined|Reference .* undefined)[^\n]*",
        log,
    )
    if failures:
        raise SystemExit("Manuscript layout/reference checks failed:\n" + "\n".join(failures))
    pdf = build / "paper.pdf"
    reader = PdfReader(pdf)
    pages = len(reader.pages)
    if not 1 <= pages <= 5:
        raise SystemExit(
            f"Manuscript has {pages} pages; required range is 1--5 including references."
        )
    if any(not (page.extract_text() or "").strip() for page in reader.pages):
        raise SystemExit("Manuscript contains an empty text page.")
    inputs = [folder / "paper.tex", folder / "figure.pdf", folder / "plot.py"]
    inputs += sorted((folder.parent / "results").glob("*.csv"))
    inputs += sorted((folder.parent / "results").glob("*.json"))
    report = {
        "schema_version": 1,
        "pages": pages,
        "page_limit": 5,
        "engine": name,
        "overfull_boxes": 0,
        "undefined_references": 0,
        "inputs_sha256": {
            p.relative_to(folder.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in inputs
        },
    }
    shutil.copyfile(pdf, folder / "paper.pdf")
    (folder / "validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Compiled {folder.parent.name}: {pages} pages; "
        "no overfull boxes or undefined references."
    )


if __name__ == "__main__":
    main()
