# Dataset Card: sft_medical

## Summary

- **Run timestamp**: 2026-11-02T23:09:03.554918Z
- **Wall clock time**: 1415.6s
- **Pipeline config hash**: `02bda62287c53ba9`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 4998 | 4998 | — | 0 | 4998 |
| SchemaGate | 4998 | 4998 | 0 | 0 | 4998 |
| ExactDeduplicator | 4998 | 4998 | — | — | 4998 |
| MinHashDeduplicator | 4998 | 4982 | — | — | 4982 |
| TextCleaner | 4982 | 4982 | — | — | 4982 |
| ToxicityGate | 4982 | 4843 | 0 | 139 | 4843 |
| ShareGPTExporter | — | 4843 | — | — | 4843 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 4998/0 passed (N/A), 0 rejected
- **SchemaGate**: 4998/4998 passed (100.0%), 0 rejected
- **ToxicityGate**: 4843/4982 passed (97.2%), 139 rejected


**Rejection reasons:**
  - `toxic_content:classifier:0.163`: 3
  - `toxic_content:classifier:0.286`: 2
  - `toxic_content:classifier:0.441`: 2
  - `toxic_content:classifier:0.175`: 2
  - `toxic_content:classifier:0.740`: 2
  - `toxic_content:classifier:0.164`: 2
  - `toxic_content:classifier:0.176`: 2
  - `toxic_content:classifier:0.107`: 2
  - `toxic_content:classifier:0.555`: 2
  - `toxic_content:classifier:0.101`: 2
  - `toxic_content:classifier:0.259`: 2
  - `toxic_content:classifier:0.162`: 2
  - `toxic_content:classifier:0.422`: 2
  - `toxic_content:classifier:0.127`: 2
  - `toxic_content:classifier:0.261`: 2
  - `toxic_content:classifier:0.108`: 2
  - `toxic_content:classifier:0.125`: 1
  - `toxic_content:classifier:0.206`: 1
  - `toxic_content:classifier:0.897`: 1
  - `toxic_content:classifier:0.904`: 1
  - `toxic_content:classifier:0.285`: 1
  - `toxic_content:classifier:0.293`: 1
  - `toxic_content:classifier:0.123`: 1
  - `toxic_content:classifier:0.210`: 1
  - `toxic_content:classifier:0.794`: 1
  - `toxic_content:classifier:0.513`: 1
  - `toxic_content:classifier:0.529`: 1
  - `toxic_content:classifier:0.131`: 1
  - `toxic_content:classifier:0.478`: 1
  - `toxic_content:classifier:0.896`: 1
  - `toxic_content:classifier:0.313`: 1
  - `toxic_content:classifier:0.424`: 1
  - `toxic_content:classifier:0.200`: 1
  - `toxic_content:classifier:0.161`: 1
  - `toxic_content:classifier:0.226`: 1
  - `toxic_content:classifier:0.615`: 1
  - `toxic_content:classifier:0.653`: 1
  - `toxic_content:classifier:0.111`: 1
  - `toxic_content:classifier:0.410`: 1
  - `toxic_content:classifier:0.317`: 1
  - `toxic_content:classifier:0.171`: 1
  - `toxic_content:classifier:0.388`: 1
  - `toxic_content:classifier:0.959`: 1
  - `toxic_content:classifier:0.288`: 1
  - `toxic_content:classifier:0.601`: 1
  - `toxic_content:classifier:0.295`: 1
  - `toxic_content:classifier:0.198`: 1
  - `toxic_content:classifier:0.825`: 1
  - `toxic_content:classifier:0.817`: 1
  - `toxic_content:classifier:0.223`: 1
  - `toxic_content:classifier:0.600`: 1
  - `toxic_content:classifier:0.579`: 1
  - `toxic_content:classifier:0.777`: 1
  - `toxic_content:classifier:0.252`: 1
  - `toxic_content:classifier:0.703`: 1
  - `toxic_content:classifier:0.661`: 1
  - `toxic_content:classifier:0.935`: 1
  - `toxic_content:classifier:0.138`: 1
  - `toxic_content:classifier:0.114`: 1
  - `toxic_content:classifier:0.586`: 1
  - `toxic_content:classifier:0.143`: 1
  - `toxic_content:classifier:0.135`: 1
  - `toxic_content:classifier:0.890`: 1
  - `toxic_content:classifier:0.617`: 1
  - `toxic_content:classifier:0.282`: 1
  - `toxic_content:classifier:0.733`: 1
  - `toxic_content:classifier:0.518`: 1
  - `toxic_content:classifier:0.649`: 1
  - `toxic_content:classifier:0.868`: 1
  - `toxic_content:classifier:0.103`: 1
  - `toxic_content:classifier:0.408`: 1
  - `toxic_content:classifier:0.801`: 1
  - `toxic_content:classifier:0.235`: 1
  - `toxic_content:classifier:0.876`: 1
  - `toxic_content:classifier:0.491`: 1
  - `toxic_content:classifier:0.666`: 1
  - `toxic_content:classifier:0.418`: 1
  - `toxic_content:classifier:0.121`: 1
  - `toxic_content:classifier:0.236`: 1
  - `toxic_content:classifier:0.119`: 1
  - `toxic_content:classifier:0.151`: 1
  - `toxic_content:classifier:0.688`: 1
  - `toxic_content:classifier:0.697`: 1
  - `toxic_content:classifier:0.173`: 1
  - `toxic_content:classifier:0.372`: 1
  - `toxic_content:classifier:0.157`: 1
  - `toxic_content:classifier:0.459`: 1
  - `toxic_content:classifier:0.936`: 1
  - `toxic_content:classifier:0.657`: 1
  - `toxic_content:classifier:0.526`: 1
  - `toxic_content:classifier:0.329`: 1
  - `toxic_content:classifier:0.254`: 1
  - `toxic_content:classifier:0.102`: 1
  - `toxic_content:classifier:0.932`: 1
  - `toxic_content:classifier:0.189`: 1
  - `toxic_content:classifier:0.191`: 1
  - `toxic_content:classifier:0.501`: 1
  - `toxic_content:classifier:0.339`: 1
  - `toxic_content:classifier:0.403`: 1
  - `toxic_content:classifier:0.168`: 1
  - `toxic_content:classifier:0.124`: 1
  - `toxic_content:classifier:0.137`: 1
  - `toxic_content:classifier:0.370`: 1
  - `toxic_content:classifier:0.256`: 1
  - `toxic_content:classifier:0.297`: 1
  - `toxic_content:classifier:0.583`: 1
  - `toxic_content:classifier:0.887`: 1
  - `toxic_content:classifier:0.106`: 1
  - `toxic_content:classifier:0.213`: 1
  - `toxic_content:classifier:0.853`: 1
  - `toxic_content:classifier:0.141`: 1
  - `toxic_content:classifier:0.160`: 1
  - `toxic_content:classifier:0.844`: 1
  - `toxic_content:classifier:0.182`: 1
  - `toxic_content:classifier:0.749`: 1
  - `toxic_content:classifier:0.540`: 1
  - `toxic_content:classifier:0.387`: 1
  - `toxic_content:classifier:0.117`: 1
  - `toxic_content:classifier:0.970`: 1
  - `toxic_content:classifier:0.242`: 1
  - `toxic_content:classifier:0.374`: 1
  - `toxic_content:classifier:0.684`: 1

## Deduplication

  - near_duplicates_removed: 16
  - surviving_count: 4982

## Known Limitations

- MinHash threshold used: **0.85** — this is a starting point, not a universal answer. Adjust per domain (legal, medical, and other domain-specific corpora may need a lower threshold to avoid losing structurally similar but semantically valid examples).
- Table extraction requires `extract_tables: true` on the PDF reader and MinerU with GPU support. Disabled by default.
- Token counting uses a whitespace tokenizer by default. For exact LLM context window
  enforcement, enable `use_tiktoken: true` in the SchemaGate config.
- Both within-run (MinHash/Diversity) and cross-run (EmbeddingDeduplicator) deduplication
  are available. Cross-run dedup persists an embedding index across runs; use
  `embedding_reset_index: true` to start fresh.

## Reproduction

```bash
pip install curatorkit
curatorkit run pipeline.yaml --output-dir ./out/
```

## Tool Versions

- **curatorkit**: 0.1.21
- **python**: 3.12.10
