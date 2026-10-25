# LaTeX Report: Everyday Waste Sorting via Manual Image Descriptors and Transfer Learning

This directory contains the publication-grade LaTeX report strictly mirroring the structural layout, typography, sectioning, mathematical rigor, and visual styling of the reference project report.

## Directory Structure
- `report.tex`: Main LaTeX source file.
- `references.bib`: BibTeX file containing academic citations.
- `images/`: High-resolution figures and evaluation plots referenced in `report.tex`.

## How to Compile

### Option A: Overleaf (Recommended - 1 Click)
1. Go to [Overleaf](https://www.overleaf.com).
2. Click **New Project** $\rightarrow$ **Upload Project**.
3. Upload `report_latex.zip` (located in the project root directory).
4. Select `pdfLaTeX` or `XeLaTeX` compiler and click **Recompile**.

### Option B: Local Compilation
If you have MacTeX / TeXLive installed on your system:
```bash
cd report
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```
Or using `latexmk`:
```bash
latexmk -pdf report.tex
```
