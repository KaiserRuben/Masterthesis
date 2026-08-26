r"""Render emitted figures to PNG for visual checking.

    PYTHONPATH=$PWD conda run --no-capture-output -n uni \
        python -m analysis.viz.thesis.pgf.preview exp100-budget [more slugs]

Writes <scratch>/preview/<slug>.png.  The page is the thesis text block with a
rule at each edge, so a figure that overhangs the column is visible as such.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from analysis.viz.thesis.pgf.emit import MEASURE_DOC, THESIS

PREVIEW = Path("/private/tmp/claude-501/-Users-kaiser-Desktop-Uni-Masterarbeit-"
               "Master-Thesis-v0-6-0/2a31adb1-39d9-4099-9cd0-0ec54f4842fe/"
               "scratchpad/preview")

DOC = MEASURE_DOC.replace(
    r"\newsavebox{\resbox}",
    "\\setlength{\\parindent}{0pt}\n\\newsavebox{\\resbox}").replace(
    r"\usebox{\resbox}",
    "\\noindent\\rule{\\textwidth}{0.4pt}\\par\\vspace{2pt}\n"
    "\\fboxsep=0pt\\fboxrule=0.4pt\\fbox{\\usebox{\\resbox}}\\par\\vspace{2pt}\n"
    "\\noindent\\rule{\\textwidth}{0.4pt}")


def render(slug: str) -> Path:
    PREVIEW.mkdir(parents=True, exist_ok=True)
    d = PREVIEW / "_build"
    d.mkdir(exist_ok=True)
    (d / "p.tex").write_text(DOC.replace("SLUG", slug), encoding="utf-8")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode",
                        f"-output-directory={d}", str(d / "p.tex")],
                       cwd=THESIS, capture_output=True, text=True,
                       env=dict(os.environ), timeout=600)
    if r.returncode != 0:
        log = (d / "p.log").read_text(errors="replace")
        raise SystemExit("\n".join(ln for ln in log.splitlines()
                                   if ln.startswith("!"))[:1500])
    out = PREVIEW / slug
    subprocess.run(["pdftoppm", "-r", "160", "-png", "-f", "1", "-l", "1",
                    "-x", "60", "-y", "60", "-W", "1300", "-H", "760",
                    str(d / "p.pdf"), str(out)], check=True)
    print(f"wrote {out}-1.png")
    return out


if __name__ == "__main__":
    for s in sys.argv[1:]:
        render(s)
