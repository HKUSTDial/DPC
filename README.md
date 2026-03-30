<div align="center">

# DPC-SQL

<p>
  <b>Dual-Program Consistency for Text-to-SQL Selection</b><br/>
  ACL 2026 Companion Codebase
</p>

<p>
  <a href="#-overview">Overview</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-reproducing-experiments">Experiments</a> •
  <a href="#-project-structure">Project Structure</a>
</p>

<p>
  ⚖️ SQL-as-candidate selection &nbsp;|&nbsp; 🐍 Python-as-proxy verification &nbsp;|&nbsp; 🧪 Distinguishing synthetic data
</p>

</div>

---

## ✨ Overview

**DPC-SQL** is an inference-time selection framework for Text-to-SQL.
Instead of asking an LLM judge to directly decide which SQL is correct, DPC-SQL builds a second reasoning channel in **Python/Pandas**, generates **distinguishing test data**, and compares candidate SQLs against a cross-program proxy answer.

This repository is the research codebase accompanying our ACL 2026 work on **Dual-Program Consistency (DPC)** for Text-to-SQL selection.

<table>
  <tr>
    <td><b>Problem</b></td>
    <td>LLM-as-a-Judge for SQL often suffers from the same reasoning errors as SQL generation itself.</td>
  </tr>
  <tr>
    <td><b>Key Idea</b></td>
    <td>Validate candidate SQLs through a second program form with different failure modes: Python/Pandas.</td>
  </tr>
  <tr>
    <td><b>What DPC Does</b></td>
    <td>Pick champion/challenger SQLs, synthesize test data where they diverge, solve the question in Python, and choose the candidate closer to the Python answer.</td>
  </tr>
  <tr>
    <td><b>Scope</b></td>
    <td>Inference-time selection only. No model finetuning or task-specific training is required.</td>
  </tr>
</table>

---

## 🧠 Method At A Glance

1. **Phase 1: Candidate Selection**
   Group candidate SQLs and identify a **champion** and **challenger**.
   The codebase currently supports:
   - `execution`: cluster by execution result on the original DB
   - `llm_prompt`: group by prompt-based logical equivalence

2. **Phase 2: Schema Slicing**
   Keep only the tables and columns needed for the duel, then validate the slice with dry-run checks.

3. **Phase 3: Distinguishing Data Generation**
   Generate a small synthetic database slice where the two SQLs produce different answers.

4. **Phase 4: Cross-Program Verification**
   Ask a Python solver to answer the same NL question over the synthetic data and compare SQL outputs against the Python result.

5. **Phase 5: Final Decision**
   Select the SQL that is more consistent with the Python proxy answer.

<details>
<summary><b>Why this is different from standard self-consistency</b></summary>

Standard self-consistency still votes within the same program form: SQL.
DPC-SQL explicitly introduces a second program modality with different inductive biases and different error modes, which helps reduce blind voting over the same logical mistake.

</details>

---

## 🚀 What’s Included

- **Main DPC pipeline**
  - `SlicerAgent`, `TesterAgent`, `PythonSolverAgent`, `EquivalenceGrouperAgent`
- **Baselines**
  - SC, MCS, USC, EX-guided, Random
- **Evaluation utilities**
  - execution accuracy
  - Pass@K / candidate upper bound
  - majority-analysis and solver-reliability analysis
- **Research-oriented experiment runners**
  - resumable batch processing
  - local artifact management under `artifacts/`
  - checked-in snapshot results under `results/`

---

## 🛠 Installation

This project uses **uv** for dependency management.

```bash
uv sync
```

That creates a local `.venv/`.
All shell wrappers in [`scripts/`](./scripts) automatically prefer `.venv/bin/python` if it exists, so after `uv sync` you can usually run them directly with `bash`.

If you want to run Python entry points directly, use:

```bash
uv run python baseline/run_dpc_selection.py --help
```

### Dependencies

The main runtime dependencies are:

- `openai`
- `pandas`
- `numpy`
- `scipy`
- `tabulate`
- `tqdm`
- `chardet`

### LLM Backend

The repository uses an **OpenAI-compatible chat API** wrapper.
You can provide credentials in either of these ways:

- shell wrappers: `API_KEY`, `BASE_URL`
- direct Python entry points: `--api_key`, `--base_url`
- fallback env vars used by the LLM wrapper: `OPENAI_API_KEY`, `OPENAI_BASE_URL`

---

## ⚙️ Configuration

Most experiments are driven by shell wrappers under [`scripts/`](./scripts).
These wrappers expose configuration through environment variables and write local outputs into `artifacts/` by default.

### Common Environment Variables

| Variable | Meaning |
|---|---|
| `MODEL_NAME` | LLM model name used by generation / selection / verification |
| `API_KEY` | API key for the OpenAI-compatible backend |
| `BASE_URL` | Base URL for the backend |
| `ARTIFACT_ROOT` | Root directory for generated local outputs, default `artifacts` |
| `PYTHON_BIN` | Explicit Python interpreter override for shell wrappers |
| `PHASE1_SELECTION_MODE` | `execution` or `llm_prompt` in DPC |
| `NUM_GROUPING_ATTEMPTS` | Self-consistency samples for prompt-based SQL grouping |
| `EVAL_METRIC` | `bs_f1` or `ex` for DPC verification |

### Output Layout

- `artifacts/`: local experiment outputs generated by your runs
- `results/`: curated or checked-in result snapshots already included in the repository

---

## ⚡ Quick Start

The default workflow is:

1. Generate candidate SQLs
2. Run a selector
3. Evaluate predictions

### 1. Generate Candidate SQLs

```bash
MODEL_NAME=gpt-5-chat-latest \
API_KEY=your_api_key \
BASE_URL=your_base_url \
bash scripts/run_gen_baseline.sh
```

This writes candidates to:

```text
artifacts/candidates/...
```

### 2. Run DPC Selection

Execution-based Phase 1:

```bash
MODEL_NAME=gpt-5-chat-latest \
API_KEY=your_api_key \
BASE_URL=your_base_url \
PHASE1_SELECTION_MODE=execution \
EVAL_METRIC=bs_f1 \
bash scripts/run_dpc_selection.sh
```

Prompt-based Phase 1:

```bash
MODEL_NAME=gpt-5-chat-latest \
API_KEY=your_api_key \
BASE_URL=your_base_url \
PHASE1_SELECTION_MODE=llm_prompt \
NUM_GROUPING_ATTEMPTS=3 \
bash scripts/run_dpc_selection.sh
```

### 3. Evaluate Predictions

Execution accuracy:

```bash
bash scripts/run_eval_ex.sh
```

Pass@K over candidate groups:

```bash
PASS_K=2 bash scripts/run_pass_n_eval.sh
```

---

## 🔬 Reproducing Experiments

### Main Entry Points

| Goal | Command |
|---|---|
| Generate generic candidate SQLs | `bash scripts/run_gen_baseline.sh` |
| Generate OmniSQL candidates | `bash scripts/run_gen_omnisql.sh` |
| Generate XiYan candidates | `bash scripts/run_gen_xiyan.sh` |
| Run DPC | `bash scripts/run_dpc_selection.sh` |
| Run SC | `bash scripts/run_sc_selection.sh` |
| Run MCS | `bash scripts/run_mcs_selection.sh` |
| Run EX-guided baseline | `bash scripts/run_ex_guided_selection.sh` |
| Run Random baseline | `bash scripts/run_random_selection.sh` |
| Run EX evaluation | `bash scripts/run_eval_ex.sh` |
| Run Pass@K evaluation | `bash scripts/run_pass_n_eval.sh` |
| Compare DPC vs SC over multiple candidate counts | `bash scripts/run_candidates_analysis.sh` |

### Additional Research Utilities

These are useful for deeper analysis and paper ablations:

| Utility | Entry Point |
|---|---|
| USC selection | `uv run python baseline/run_usc_selection.py ...` |
| MCS without execution clustering | `uv run python baseline/run_mcs_selection_wo_execution.py ...` |
| MDD generation only | `uv run python baseline/run_mdd_generation.py ...` |
| Post-hoc MDD stats | `uv run python baseline/run_mdd_posthoc_stats.py ...` |
| Solver reliability analysis | `uv run python baseline/run_solver_reliability_experiment.py ...` |
| Majority-analysis between SC and DPC | `bash scripts/run_majority_analysis.sh` |

### Direct Python Usage

If you want full control beyond the shell wrappers:

```bash
uv run python baseline/run_dpc_selection.py \
  --dataset_type bird \
  --data_path data/bird/dev/dev.json \
  --db_root_path data/bird/dev/dev_databases \
  --pred_sqls_path artifacts/candidates/GPT-4o.json \
  --output_path artifacts/selected/DPC_Results.json \
  --model_name gpt-4o \
  --phase1_selection_mode execution \
  --eval_metric bs_f1 \
  --num_workers 8
```

---

## 📊 Input / Output Format

### Candidate SQL File

The repository expects candidate SQLs in JSON format:

```json
{
  "question_id_0": [
    "SELECT count(*) FROM head WHERE age > 56",
    "SELECT count(*) FROM head WHERE age >= 56"
  ],
  "question_id_1": [
    "SELECT ..."
  ]
}
```

### Selection Output

Selectors typically write:

```json
{
  "question_id_0": "SELECT ..."
}
```

### Notes

- DPC and some analysis scripts save incrementally and support resume.
- `artifacts/` is intended for your local runs.
- `results/` contains repository snapshots and should not be treated as the default write target.

---

## 🧪 Supported Datasets

The current codebase includes dataset loaders for:

- **BIRD**
- **Spider**

Relevant modules:

- [`dpc/datasets/bird_loader.py`](./dpc/datasets/bird_loader.py)
- [`dpc/datasets/spider_loader.py`](./dpc/datasets/spider_loader.py)

---

## 🗂 Project Structure

```text
DPC-SQL/
├── artifacts/          # Local experiment outputs (gitignored)
├── baseline/           # Main runners for generation, baselines, and analysis
├── data/               # Dataset files and database directories
├── dpc/
│   ├── agents/         # Slicer / Tester / Solver / Selector agents
│   ├── core/           # DPC pipeline orchestration
│   ├── datasets/       # Spider / BIRD dataset loaders
│   ├── eval/           # BS-F1 / EX evaluation logic
│   ├── llm/            # OpenAI-compatible LLM wrapper
│   ├── prompts/        # Prompt templates
│   └── utils/          # DB execution, schema extraction, parsing, clustering
├── evaluation/         # Evaluation scripts
├── results/            # Checked-in snapshots / curated outputs
├── scripts/            # Experiment shell wrappers
├── pyproject.toml      # uv dependency definition
└── uv.lock             # Reproducible dependency lockfile
```

---

## 📝 Notes For Paper Release

- This repository is organized as a **research codebase**, not as a packaged library.
- The preferred user-facing interface is the combination of:
  - `uv sync`
  - `bash scripts/...`
- Prompt-based paths depend on an available OpenAI-compatible LLM backend and may incur API cost.

---

## 📚 Citation

Citation metadata can be added here after the paper metadata is finalized.

```bibtex
@misc{dpc_sql_2026,
  title  = {DPC-SQL: Dual-Program Consistency for Text-to-SQL Selection},
  year   = {2026},
  note   = {ACL 2026 companion codebase}
}
```

---

## 📄 License

This project is released under the MIT License.
