# Academic manuscript

The complete research note, including figures and references, must be at most five pages.
`paper.tex` is the editable source; `paper.pdf` is the compiled reading copy.
The inline bibliography needs no BibTeX service. `figure.pdf` is an original vector
plot derived from the repository's committed numerical CSVs, regenerated with:

```sh
python paper/plot.py
```

Install the project's plotting dependencies first. To compile, install either
[Tectonic](https://tectonic-typesetting.github.io/) or a TeX distribution with
pdfLaTeX, plus the PDF page-count dependency:

```sh
python -m pip install pypdf==6.10.0
python paper/build.py
# Or explicitly:
python paper/build.py --engine pdflatex
```

Run from any directory; paths resolve relative to this script. The manuscript uses
standard LaTeX packages: geometry, fontenc, lmodern, amsmath, amssymb, graphicx,
booktabs, microtype, caption, hyperref, and fancyhdr. TeX Live's base/recommended
LaTeX and font collections plus lmodern provide them. Tectonic downloads missing
bundle assets on first use. Compiler intermediates remain in ignored `paper/build/`.

The builder fails on compilation errors, undefined citations/references, overfull
boxes, empty pages, or a PDF longer than five pages. It writes input hashes and page
count to `validation.json`. These automated checks do not replace visual review.
The dedicated GitHub Actions workflow compiles independently with pdfLaTeX.
Different engines may produce different PDF bytes or pagination; each must remain
within the page limit. The numerical source snapshot is identified in the paper.
