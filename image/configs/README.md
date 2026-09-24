# Evaluation protocols and prompt banks

The generators are contract-driven: a run names an `--evaluation-protocol`, and
the matching JSON here is verified by SHA-256 before any image is drawn. Each
protocol pins the dataset, the prompt-CSV hash, and (for SD3.5-Medium) the model
revision and component-config hashes.

## Protocols (`protocols/`)

| file | used for |
|---|---|
| `coco2017_val5k_sd3m_seeded_v1.json` | SD3-Medium quality, 5k COCO captions (512²) |
| `coco2017_val5k_1024_sd3m_seeded_v1.json` | SD3-Medium quality at 1024² |
| `coco2017_val5k_sd35m_seeded_v1.json` | SD3.5-Medium quality, 5k (512²) |
| `coco2017_val5k_1024_sd35m_seeded_v1.json` | SD3.5-Medium quality, 5k (1024²) |
| `coco2017_val1k_1024_sd35m_seeded_v1.json` | SD3.5-Medium quality, 1k (1024²) |
| `sd35m1024_diversity_100x10_v1.json` | diversity bank: 100 COCO prompts × 10 seeds |
| `sd35m1024_portrait_3prompts_25seeds_v1.json` | Figure 1 portraits, 40-step teacher |
| `sd35m1024_portrait_3prompts_25seeds_teacher60_v1.json` | Figure 1 portraits, 60-step teacher |

## The prompt CSVs are not vendored

Each protocol references an external `prompt_csv` (the actual COCO-2017
validation captions and per-case seeds) by path and SHA-256. Those CSVs are
large and derived from COCO, so they are **not** committed here. To reproduce:

1. Build the caption/seed CSV from COCO-2017-val so its SHA-256 matches the
   `prompt_csv_sha256` in the protocol (the diversity bank is the first 100
   captions of COCO-2017-val5k with 10 seeds each; the quality banks are the
   first 1k / 5k with one seed each).
2. Point the generator's `--prompts` at the protocol JSON; it resolves and hashes
   the CSV, and aborts on any mismatch.

## Prompt banks (`evaluation/`)

- `flux_portrait_3prompts_25seeds_v1.csv` — the SenseFlow-FLUX / SD3.5-M portrait
  cases (self-contained: prompts + seeds).
- `sd3m_frozen_suffix_prompts_v1.txt` — prompts for the frozen-suffix diagnostic.
