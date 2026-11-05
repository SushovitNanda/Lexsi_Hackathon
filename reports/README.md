# Build the simple research report

Content source: `../SIMPLE_RESEARCH_REPORT.md`.

```bash
brew install pandoc tectonic poppler
.venv/bin/python scripts/build_research_report.py
```

Run from the repository root. The script converts the Markdown through Pandoc's structured document format, assigns readable table widths, writes standalone LaTeX, and compiles it with Tectonic. Evidence links become absolute GitHub URLs in the PDF. The pinned TeX resource bundle is `tlextras-2022.0r0`; the first build downloads resources into Tectonic's cache.

- Final PDF and standalone LaTeX: `output/pdf/manish_research_report.pdf` and `.tex`.
- Design template: `reports/research_report.template.tex`.
- Build logs and rendered page PNGs: `tmp/pdfs/` (local QA files).

The build used Pandoc 3.13, Tectonic 0.17.0 and Poppler 26.10.0. The report's evidence window is November 4-5, 2026; it was prepared November 6. The medical termination observation, unverified merge parity and pending T4/judge checks are retained explicitly. No experimental outputs are changed by this document build.
