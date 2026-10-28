# Evaluation protocols and prompt banks

The SD3 generators use `--evaluation-protocol` to select a configuration.
Each configuration records the dataset, prompt-CSV hash, and, for
SD3.5-Medium, the model revision and component configuration hashes.

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

## External prompt CSVs

Each protocol references an external `prompt_csv` (the actual COCO-2017
validation captions and per-case seeds) by path and SHA-256. These CSVs are
not included in this repository. Reproducing the fixed protocols requires the
exact referenced files, including their caption selection, order, and seeds;
an arbitrary COCO subset will not match the recorded hash.

Place the matching CSV at the path specified by `prompt_csv`, relative to
the protocol JSON. Pass that JSON to the generator's `--prompts` option.
The generator checks the CSV hash and stops on a mismatch. The JSON files
alone do not provide the complete evaluation prompt banks.

## Prompt banks (`evaluation/`)

- `flux_portrait_3prompts_25seeds_v1.csv` — the SenseFlow-FLUX / SD3.5-M portrait
  cases (self-contained: prompts + seeds).
- `sd3m_frozen_suffix_prompts_v1.txt` — prompts for the frozen-suffix diagnostic.
