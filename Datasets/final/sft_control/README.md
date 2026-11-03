---
tags:
- curatorkit
configs:
- config_name: sft_sharegpt
  default: true
  description: curatorkit sft_sharegpt export, sha256 c1a1d5ac76fb437f15f83f96e0941fc286867fa215c5a28f7735e1821ed1fc2e
  data_files:
  - split: train
    path: sft_sharegpt.jsonl
---

# Dataset Card: sft_control

## Summary

- **Run timestamp**: 2026-11-02T23:19:13.370648Z
- **Wall clock time**: 605.4s
- **Pipeline config hash**: `1bcd89effa64dc12`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 5118 | 5118 | — | 0 | 5118 |
| SchemaGate | 5118 | 5118 | 0 | 0 | 5118 |
| ExactDeduplicator | 5118 | 5118 | — | — | 5118 |
| MinHashDeduplicator | 5118 | 5118 | — | — | 5118 |
| TextCleaner | 5118 | 5118 | — | — | 5118 |
| ToxicityGate | 5118 | 5068 | 0 | 50 | 5068 |
| ShareGPTExporter | — | 5068 | — | — | 5068 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 5118/0 passed (N/A), 0 rejected
- **SchemaGate**: 5118/5118 passed (100.0%), 0 rejected
- **ToxicityGate**: 5068/5118 passed (99.0%), 50 rejected


**Rejection reasons:**
  - `toxic_content:classifier:0.183`: 3
  - `toxic_content:classifier:0.105`: 3
  - `toxic_content:classifier:0.141`: 2
  - `toxic_content:classifier:0.199`: 2
  - `toxic_content:classifier:0.149`: 2
  - `toxic_content:classifier:0.284`: 1
  - `toxic_content:classifier:0.190`: 1
  - `toxic_content:classifier:0.340`: 1
  - `toxic_content:classifier:0.112`: 1
  - `toxic_content:classifier:0.977`: 1
  - `toxic_content:classifier:0.114`: 1
  - `toxic_content:classifier:0.792`: 1
  - `toxic_content:classifier:0.717`: 1
  - `toxic_content:classifier:0.102`: 1
  - `toxic_content:classifier:0.440`: 1
  - `toxic_content:classifier:0.734`: 1
  - `toxic_content:classifier:0.513`: 1
  - `toxic_content:classifier:0.324`: 1
  - `toxic_content:classifier:0.346`: 1
  - `toxic_content:classifier:0.461`: 1
  - `toxic_content:classifier:0.101`: 1
  - `toxic_content:classifier:0.111`: 1
  - `toxic_content:classifier:0.118`: 1
  - `toxic_content:classifier:0.347`: 1
  - `toxic_content:classifier:0.246`: 1
  - `toxic_content:classifier:0.193`: 1
  - `toxic_content:classifier:0.245`: 1
  - `toxic_content:classifier:0.232`: 1
  - `toxic_content:classifier:0.205`: 1
  - `toxic_content:classifier:0.255`: 1
  - `toxic_content:classifier:0.123`: 1
  - `toxic_content:classifier:0.242`: 1
  - `toxic_content:classifier:0.176`: 1
  - `toxic_content:classifier:0.195`: 1
  - `toxic_content:classifier:0.723`: 1
  - `toxic_content:classifier:0.212`: 1
  - `toxic_content:classifier:0.150`: 1
  - `toxic_content:classifier:0.110`: 1
  - `toxic_content:classifier:0.720`: 1
  - `toxic_content:classifier:0.446`: 1
  - `toxic_content:classifier:0.104`: 1
  - `toxic_content:classifier:0.137`: 1
  - `toxic_content:classifier:0.204`: 1

## Deduplication

  - near_duplicates_removed: 0
  - surviving_count: 5118

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

## Configs

Load one with `datasets.load_dataset("<this folder or Hub repo>", "<config>")`.

| Config | Splits (rows) | Columns |
|--------|---------------|---------|
| `sft_sharegpt` | train (5068) | `conversations`: list of {from, value}; from is system, human or gpt |
