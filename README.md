# LexBrief

**Rhetorical-role-aware extractive briefs for Indian court judgments.**

Indian judgments are long, and their structure is only implicit. LexBrief labels every sentence
with one of 13 rhetorical roles (facts, arguments, statutes, precedents, ratio, ruling and so on)
using InLegalBERT plus a document-level BiLSTM-CRF. It then collapses those roles into 7 brief
sections and assembles an extractive brief, giving each section a word budget learned from IN-Ext
expert summaries. The result is a brief that reads like a lawyer's case note, with every sentence
traceable to the judgment.

```
judgment -> sentences -> InLegalBERT -> BiLSTM-CRF -> 13 roles -> 7 sections -> budgeted brief
```

## Setup (Windows 11, PowerShell, NVIDIA GPU)

Requirements: Python 3.11 and an NVIDIA driver with CUDA 12.x support.

```powershell
git clone <your-repo-url> lexbrief
cd lexbrief
py -3.11 -m venv .venv                      # skipped automatically if .venv exists
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_env.ps1                     # torch (cu121), requirements, -e ., spaCy, pre-commit
.\.venv\Scripts\Activate.ps1
lexbrief check-env                          # GPU, VRAM, bf16, versions
pytest -q
ruff check .
```

Configuration lives in `configs/*.yaml`. Any value can be overridden from the CLI, for example
`train.lr=3e-5`.

## Roadmap

- [x] **Prompt 1: Foundation.** Env setup, packaging, config, label maps, utils, CLI `check-env`
- [ ] **Prompt 2: Data.** Download BUILD and IN-Ext, sentence splitting, alignment, splits, stats
- [ ] **Prompt 3: Sentence classifiers.** TF-IDF baseline, BERT-base, InLegalBERT, context model
- [ ] **Prompt 4: Hierarchical model.** Embedding cache and BiLSTM-CRF (plus end-to-end variant)
- [ ] **Prompt 5: Brief generation.** Learned per-role budgets, scorer, selector, baselines, oracle
- [ ] **Prompt 6: Evaluation.** ROUGE, per-role ROUGE, role coverage, significance tests
- [ ] **Prompt 7: Experiments.** Full grid, gold-vs-predicted roles and noise study, figures/tables
- [ ] **Prompt 8: Demo and report.** Streamlit app and final write-up

## Repository layout

```
src/lexbrief/   package (labels, config, data, models, training, brief, evaluation, utils, cli)
configs/        YAML configs for every stage and model
scripts/        PowerShell helpers
app/            Streamlit demo
tests/          pytest suite
report/         report outline, figures, tables
data/ models/   local only (git-ignored)
outputs/        results/ and figures/ tracked; logs/ and runs/ ignored
```
