# CLAUDE.md: LexBrief

## Project summary
LexBrief is a rhetorical-role-aware extractive brief generator for Indian court judgments.

1. **Role classification:** classify every sentence of a judgment into 13 rhetorical roles
   (BUILD / InRhetoricalRoles labels) with InLegalBERT (`law-ai/InLegalBERT`), then run a
   BiLSTM-CRF over the document's sentence sequence.
2. **Collapse:** map the 13 fine roles to 7 coarse brief sections (plus `DROP`).
3. **Brief:** build an extractive brief with a per-role word budget learned from IN-Ext
   expert summaries.
4. **Evaluate:** report ROUGE, per-role ROUGE, role coverage, and a gold-vs-predicted-roles study.
5. **Demo:** a Streamlit app (`app/streamlit_app.py`).

Layout: `src/lexbrief/` (package), `configs/` (YAML), `tests/`, `scripts/` (PowerShell),
`app/` (Streamlit), `report/`, `data/` and `models/` (local only), `outputs/`.

## Hard constraints
- **GPU: 6 GB VRAM** (dev machine: RTX 4050 Laptop, compute 8.9, bf16 supported; target
  spec RTX 3050 6 GB). Keep batch sizes small; use gradient accumulation,
  `max_length` of 128 or less for sentence encoding, and gradient checkpointing if needed. Cache
  embeddings for the BiLSTM-CRF stage. Free memory between stages.
- **Windows 11 + PowerShell.** All scripts are `.ps1`. Use `pathlib`, never hard-coded `/` or `\`.
- **DataLoader `num_workers=0` always** (Windows multiprocessing + CUDA is fragile).
- **Mixed precision is bf16 autocast** (`lexbrief.utils.gpu.autocast_ctx`). Don't use fp16 or
  GradScaler.
- **Never commit** `data/`, `models/`, `.venv/`, checkpoints (`*.pt`, `*.bin`, `*.safetensors`),
  `outputs/logs/` or `outputs/runs/`. Only the `.gitkeep` placeholders are tracked there.
- Python 3.11 venv at `.venv`. torch (>= 2.6, **cu126** index) is NOT in requirements.txt.
  torch >= 2.6 is mandatory because transformers 5.x refuses `.bin` checkpoints such as
  InLegalBERT on older torch (CVE-2025-32434), and the cu121 index stops at 2.5.1. Use the
  transformers v5 APIs (e.g. `dtype=` rather than `torch_dtype=`).

## Conventions
- Type hints on every function and Google-style docstrings on public functions and classes.
- Use `logging` (`logger = logging.getLogger(__name__)`), **never `print`**. CLI output may use
  `typer.echo`. Configure logging via `lexbrief.utils.logging.setup_logging`.
- **Every path and hyperparameter comes from `configs/*.yaml`**, loaded with
  `lexbrief.config.load_config(path, overrides)`. CLI overrides look like `train.lr=3e-5`.
  Don't put literal paths or magic numbers in code; add a field to the dataclasses in
  `config.py` instead.
- Seed via `lexbrief.utils.seed.set_seed(cfg.seed)` at the start of every entry point.
- JSON/JSONL I/O via `lexbrief.utils.io`.
- Label strings and maps come only from `lexbrief.labels`. Never re-type them.
- Quality gate: `ruff check .`, `ruff format .`, and `pytest -q` must pass before handoff.
- Fill the existing skeleton files and don't rename them.

## Label maps (`src/lexbrief/labels.py`)
`FINE_LABELS` (fixed order, index = class id):

| id | fine | coarse |
|---:|---|---|
| 0 | PREAMBLE | DROP |
| 1 | FAC | FACTS |
| 2 | RLC | FACTS |
| 3 | ISSUE | ISSUES |
| 4 | ARG_PETITIONER | ARGUMENTS |
| 5 | ARG_RESPONDENT | ARGUMENTS |
| 6 | ANALYSIS | REASONING |
| 7 | STA | STATUTE |
| 8 | PRE_RELIED | PRECEDENT |
| 9 | PRE_NOT_RELIED | PRECEDENT |
| 10 | RATIO | REASONING |
| 11 | RPC | RULING |
| 12 | NONE | DROP |

`INEXT_TO_COARSE`: FAC→FACTS, ARG→ARGUMENTS, STA→STATUTE, PRE→PRECEDENT, Ratio→REASONING,
RPC→RULING.

`BRIEF_SECTION_ORDER`: FACTS, ISSUES, ARGUMENTS, STATUTE, PRECEDENT, REASONING, RULING.

`COARSE_COLORS` (Okabe-Ito): FACTS #0072B2, ISSUES #E69F00, ARGUMENTS #56B4E9,
STATUTE #009E73, PRECEDENT #CC79A7, REASONING #D55E00, RULING #F0E442, DROP #999999.

Helpers: `fine_to_coarse`, `inext_to_coarse`, `fine_id`, `fine_label`, `coarse_color`,
`is_dropped`, `collapse_sequence`.

## Git handoff (mandatory)
Every task must end with a **"Git handoff"** section that gives the user exact PowerShell
commands to run: `git status` (and what to check), `git add ...`, `git commit -m "<conventional
message>"`, `git push`, and what to verify on GitHub afterwards. Don't commit or push on the
user's behalf unless they ask.
