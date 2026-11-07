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

## November 6 handoff to Sushovit

`output/pdf/manish_nov6_handoff.pdf` is the two-page M9/M10 handoff; its standalone LaTeX source is adjacent. Evidence is pinned to commit `2ac0a8c`, with links to all five ZIPs and the manifests. It distinguishes generation completion from pending safety judgments, includes the all-response token-cap finding, and marks phone timing as the resumed 668-response session only. Visual QA is recorded in `reports/nov6_handoff_qa.json`.

Rebuild with `tectonic --bundle https://data1b.fullyjustified.net/tlextras-2022.0r0.tar --outdir output/pdf output/pdf/manish_nov6_handoff.tex`, then render and inspect both pages before sharing changes.

## M14 methods and ROI preparation

`reports/methods/rq5_rq6_roi.tex` is an article-ready LaTeX fragment (requires amsmath/hyperref), drafted November 7 ahead of the November 8 M14 step. It distinguishes completed generation/export mechanics from planned judging, repair, real-map and utility work. `reports/methods/M13_READINESS.md` documents D4 acceptance and the prepared, unexecuted real-map workflow.

The ROI calculator uses explicit scenario inputs, with no bundled vendor prices. Run `.venv/bin/python scripts/mac/roi.py --scenario configs/mac/planning/roi_scenario.example.json --out results/mac/planning/roi_example.json`. The example is arithmetic-only and uses invented assumptions throughout; actual ROI numbers remain pending workload, power, pricing and termination validation. Tests cover monetary units, break-even, capacity rejection, unavailable zero-denominator ratios and invalid assumptions.

Readable M14 draft: `output/pdf/manish_rq5_rq6_roi_methods.pdf`; standalone source adjacent. It has three reviewed pages, with QA in `reports/methods/qa.json`. Compile its standalone source with the same pinned Tectonic bundle used above. The article fragment omits standalone page breaks so it can be included in the final Article layout.
