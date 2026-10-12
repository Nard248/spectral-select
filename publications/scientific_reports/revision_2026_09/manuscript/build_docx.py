"""Convert a manuscript source folder (original_v6 / revised / diff) to an editable Word file with pandoc.

usage: build_docx.py <source_dir> <output.docx> [--diff]

Steps: copy the source to a scratch folder; render the three TikZ figures and Algorithm 1 to PNG with
pdflatex + pdfcrop; replace them by \\includegraphics; strip \\resizebox; turn two-column floats into
plain floats; wrap the body in a minimal preamble that defines the IEEEtran-specific macros; run pandoc
(IEEE CSL, numeric citations). With --diff the latexdiff markup is mapped to Word character styles
"DIFadd" (blue, underlined) and "DIFdel" (red, struck through) via a reference document and a Lua filter.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PANDOC = Path("/private/tmp/claude-501/-Users-narekmeloyan-PycharmProjects-4D-Hyperspectral-Unsupervised-Clustering/f3c6bd32-c8a8-4450-b3df-44c8dd853077/scratchpad/pandoc_env/lib/python3.11/site-packages/pypandoc/files/pandoc")
CSL = Path("/usr/local/texlive/2026/texmf-dist/tex/latex/citation-style-language/styles/ieee.csl")

IEEE_MACROS = r"""
\newcommand{\IEEEPARstart}[2]{#1#2}
\newenvironment{IEEEkeywords}{\par\textbf{Keywords: }}{\par}
\newcommand{\IEEEauthorblockN}[1]{#1}
\newcommand{\IEEEauthorblockA}[1]{#1}
"""
DIF_MACROS = r"""
\newcommand{\DIFadd}[1]{\underline{#1}}
\newcommand{\DIFdel}[1]{\sout{#1}}
\newcommand{\DIFaddFL}[1]{\underline{#1}}
\newcommand{\DIFdelFL}[1]{\sout{#1}}
\newcommand{\DIFaddbegin}{}
\newcommand{\DIFaddend}{}
\newcommand{\DIFdelbegin}{}
\newcommand{\DIFdelend}{}
\newcommand{\DIFaddbeginFL}{}
\newcommand{\DIFaddendFL}{}
\newcommand{\DIFdelbeginFL}{}
\newcommand{\DIFdelendFL}{}
\newcommand{\DIFmodbegin}{}
\newcommand{\DIFmodend}{}
"""
LUA = r"""
function Underline(el) return pandoc.Span(el.content, {['custom-style'] = 'DIFadd'}) end
function Strikeout(el) return pandoc.Span(el.content, {['custom-style'] = 'DIFdel'}) end
"""


def strip_resizebox(out: str) -> str:
    while True:
        i = out.find(r"\resizebox{")
        if i < 0:
            return out
        j = i + len(r"\resizebox")
        for _ in range(2):  # skip {width}{height}
            depth = 0
            while True:
                if out[j] == "{":
                    depth += 1
                elif out[j] == "}":
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                j += 1
        assert out[j] == "{"
        depth, k = 0, j
        while True:
            if out[k] == "{":
                depth += 1
            elif out[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        out = out[:i] + out[j + 1:k] + out[k + 1:]


def render_floats(work: Path, preamble: str, body: str) -> str:
    """Render tikzpicture blocks and the algorithm float to PNG; return body with \\includegraphics."""
    tikz = re.findall(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", body, flags=re.S)
    alg = re.findall(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", body, flags=re.S)
    assert len(tikz) == 3 and len(alg) == 1, (len(tikz), len(alg))
    figs = "\n".join(["\\begin{document}\\pagestyle{empty}"] + [b + "\n\\clearpage" for b in tikz]
                     + [alg[0].replace("[t]", "[H]")] + ["\\end{document}"])
    pre = preamble.replace(r"\usepackage{siunitx}", "\\usepackage{siunitx}\n\\usepackage{float}")
    # the diff preamble carries DIF macro definitions that are harmless for the figure render
    (work / "figs.tex").write_text(pre + "\n" + figs)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "figs.tex"], cwd=work, capture_output=True)
    subprocess.run(["pdfcrop", "--margins", "6", "figs.pdf", "figs_crop.pdf"], cwd=work, capture_output=True)
    subprocess.run(["pdftoppm", "-r", "300", "-png", "figs_crop.pdf", "figures/tikz"], cwd=work, check=True)
    widths = [r"\textwidth", r"\textwidth", r"0.48\textwidth"]
    for n, (blk, w) in enumerate(zip(tikz, widths), start=1):
        body = body.replace(blk, f"\\includegraphics[width={w}]{{figures/tikz-{n}.png}}")
    body = body.replace(alg[0], "\\begin{figure}[H]\\centering\\includegraphics[width=0.9\\columnwidth]{figures/tikz-4.png}"
                                "\\par\\textit{Algorithm 1: MMR Wavelength Selection}\\end{figure}")
    return body


def flatten(src: Path) -> tuple[str, str]:
    """Return (preamble, body) of main.tex with \\input{...} files inlined."""
    main = (src / "main.tex").read_text()
    pre, body = main.split(r"\begin{document}", 1)
    body = body.rsplit(r"\end{document}", 1)[0]

    def repl(m):
        return (src / (m.group(1) + ".tex")).read_text()
    body = re.sub(r"\\input\{([^}]+)\}", repl, body)
    return pre, body


def make_reference_docx(path: Path):
    from docx import Document
    from docx.enum.style import WD_STYLE_TYPE
    from docx.shared import RGBColor
    d = Document()
    s = d.styles.add_style("DIFadd", WD_STYLE_TYPE.CHARACTER)
    s.font.color.rgb = RGBColor(0x00, 0x00, 0xCC)
    s.font.underline = True
    s = d.styles.add_style("DIFdel", WD_STYLE_TYPE.CHARACTER)
    s.font.color.rgb = RGBColor(0xCC, 0x00, 0x00)
    s.font.strike = True
    d.save(path)


def build(src: Path, out: Path, diff: bool):
    work = Path(tempfile.mkdtemp(prefix="docx_", dir=str(HERE)))
    figs = src / "figures" if (src / "figures").exists() else HERE / "revised" / "figures"
    bib = src / "references.bib" if (src / "references.bib").exists() else HERE / "revised" / "references.bib"
    shutil.copytree(figs, work / "figures")
    shutil.copy(bib, work / "references.bib")
    pre, body = flatten(src)
    body = render_floats(work, pre, body)
    body = strip_resizebox(body)
    body = body.replace("\\begin{equation}\n$\n", "\\begin{equation}\n").replace("$,\n\\label{eq:features}", " ,\n\\label{eq:features}")
    for a, b in ((r"\begin{figure*}", r"\begin{figure}"), (r"\end{figure*}", r"\end{figure}"),
                 (r"\begin{table*}", r"\begin{table}"), (r"\end{table*}", r"\end{table}")):
        body = body.replace(a, b)
    body = body.replace(r"\bibliographystyle{IEEEtran}", "").replace(r"\bibliography{references}", r"\section*{References}")
    # abstract environment -> heading
    body = body.replace(r"\begin{abstract}", r"\section*{Abstract}").replace(r"\end{abstract}", "")
    macros = "\n".join(l for l in pre.splitlines() if l.startswith(r"\newcommand"))
    title = re.search(r"\\title\{(.+?)\}", body).group(1)
    body = re.sub(r"\\title\{.+?\}", "", body, count=1)
    i = body.find(r"\author{")
    if i >= 0:  # remove the balanced \author{...} block
        depth, k = 0, i + len(r"\author")
        while True:
            if body[k] == "{":
                depth += 1
            elif body[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        body = body[:i] + body[k + 1:]
    body = body.replace(r"\maketitle", "")
    head = IEEE_MACROS + (DIF_MACROS if diff else "") + macros + "\n"
    head += "\\title{" + title + "}\n\\author{Narek Meloyan and Narine Sarvazyan}\n\\date{"
    head += ("Change-marked revision, 14 September 2026 (blue underline = added, red strikethrough = deleted)" if diff else "Revised manuscript, 14 September 2026") + "}\n"
    (work / "main_docx.tex").write_text(head + "\\begin{document}\n\\maketitle\n" + body + "\n\\end{document}\n")
    cmd = [str(PANDOC), "main_docx.tex", "--from", "latex", "--to", "docx", "--citeproc", "--bibliography", "references.bib",
           "--csl", str(CSL), "--resource-path=.", "-o", str(out)]
    if diff:
        (work / "dif.lua").write_text(LUA)
        make_reference_docx(work / "reference.docx")
        cmd += ["--lua-filter", "dif.lua", "--reference-doc", "reference.docx"]
    r = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    warn = [l for l in r.stderr.splitlines() if l.strip()]
    print(f"pandoc exit {r.returncode}; warnings: {len(warn)}")
    for l in warn[:6]:
        print("  ", l[:160])
    shutil.rmtree(work, ignore_errors=True)
    return r.returncode


if __name__ == "__main__":
    src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    sys.exit(build(src, out, "--diff" in sys.argv))
