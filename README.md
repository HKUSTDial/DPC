<div align="center">

<h1>DPC</h1>

<p>
  <strong>DPC: Training-Free Text-to-SQL Candidate Selection via Dual-Paradigm Consistency</strong>
</p>

<p>
  Official research code for our ACL 2026 Main Track paper
</p>

<p>
  Boyan Li · Ou Ocean Kun Hei · Yue Yu · Yuyu Luo
</p>

<p>
  <a href="https://arxiv.org/abs/2604.15163">
    <img src="https://img.shields.io/badge/Paper-arXiv%3A2604.15163-b31b1b?logo=arxiv&logoColor=white" alt="Paper: arXiv 2604.15163" />
  </a>
  <img src="https://img.shields.io/badge/Venue-ACL%202026%20Main%20Track-1f6feb" alt="Venue: ACL 2026 Main Track" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/Package-uv-6C47FF" alt="Package manager: uv" />
</p>

<p>
  <a href="https://arxiv.org/abs/2604.15163">Paper</a> •
  <a href="#-overview">Overview</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-reproducing-experiments">Experiments</a> •
  <a href="#-citation">Citation</a>
</p>

</div>

<table>
  <tr>
    <td align="center"><b>Task</b><br/>Training-free Text-to-SQL candidate selection</td>
    <td align="center"><b>Core Idea</b><br/>Verify SQL with a Python/Pandas solution on a minimal distinguishing database</td>
    <td align="center"><b>Outcome</b><br/>Reduce selection errors caused by shared LLM blind spots and self-consistency bias</td>
  </tr>
</table>

---

## 🔔 News

- `2026-04-17`: Our paper is available on arXiv: [2604.15163](https://arxiv.org/abs/2604.15163).
- `2026`: DPC is accepted to **ACL 2026 Main Track**.

---

## ✨ Overview

**DPC** is a training-free method for **inference-time** Text-to-SQL candidate selection.
Instead of asking an LLM judge to directly decide which SQL is correct, DPC introduces a second reasoning channel in **Python/Pandas**, constructs a **Minimal Distinguishing Database (MDD)**, and chooses the SQL candidate whose execution is more consistent with the Python solution.

> DPC reframes SQL selection from "guess which candidate is right on hidden data" into "verify which candidate survives on visible distinguishing data."

This repository accompanies the paper **[DPC: Training-Free Text-to-SQL Candidate Selection via Dual-Paradigm Consistency](https://arxiv.org/abs/2604.15163)**, which reports consistent gains on **BIRD** and **Spider**, including up to **+2.2 absolute accuracy** over strong training-free baselines.

<table>
  <tr>
    <td><b>Problem</b></td>
    <td>LLM-as-a-Judge for SQL often shares the same reasoning failures as SQL generation itself.</td>
  </tr>
  <tr>
    <td><b>Key Idea</b></td>
    <td>Cross-check candidate SQLs with a second program form that has different failure modes: Python/Pandas.</td>
  </tr>
  <tr>
    <td><b>Verification Target</b></td>
    <td>Construct a Minimal Distinguishing Database where competing SQLs diverge, then compare them against a Python answer.</td>
  </tr>
  <tr>
    <td><b>Scope</b></td>
    <td>Inference-time selection only. No task-specific finetuning or verifier training is required.</td>
  </tr>
</table>

---

## 🧠 Method At A Glance

1. **Candidate Selection**
   Group candidate SQLs and identify a **champion** and **challenger**.
   Supported grouping modes:
   - `execution`: cluster by execution result on the original database
   - `llm_prompt`: group by prompt-based logical equivalence

2. **Schema Slicing**
   Keep only the tables and columns needed for the duel, then validate the slice with dry-run checks.

3. **Minimal Distinguishing Database Construction**
   Generate a compact synthetic database slice where the two SQLs produce different answers.

4. **Cross-Paradigm Verification**
   Ask a Python solver to answer the same NL question over the synthetic data and compare both SQL outputs against the Python result.

5. **Final Decision**
   Select the SQL candidate that is more consistent with the Python proxy answer.

<details>
<summary><b>Why this is different from standard self-consistency</b></summary>

Standard self-consistency still votes within the same program form: SQL.
DPC explicitly introduces a second program modality with different inductive biases and failure modes, which helps avoid consensus over the same logical mistake.

</details>

---

## 🚀 What’s Included

- **Main DPC pipeline**
  - `SlicerAgent`, `TesterAgent`, `PythonSolverAgent`, `EquivalenceGrouperAgent`
- **Training-free baselines**
  - SC, MCS, USC, EX-guided, Random
- **Evaluation utilities**
  - execution accuracy
  - Pass@K / candidate upper bound
  - majority-analysis and solver-reliability analysis
- **Research-oriented runners**
  - resumable batch processing
  - local artifact management under `artifacts/`
  - checked-in result snapshots under `results/`

---

## 🛠 Installation

This project uses **uv** for dependency management.

```bash
uv sync
```

This creates a local `.venv/`.
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

## 📝 Notes

- This repository is organized as a **research codebase**, not as a packaged library.
- The preferred user-facing workflow is:
  - `uv sync`
  - `bash scripts/...`
- Prompt-based paths depend on an available OpenAI-compatible LLM backend and may incur API cost.

---

## 📚 Citation

If you find this repository useful, please cite our paper.
Until the ACL Anthology entry is available, we recommend using the arXiv-form BibTeX below.

```bibtex
@misc{li2026dpc,
  title         = {DPC: Training-Free Text-to-SQL Candidate Selection via Dual-Paradigm Consistency},
  author        = {Boyan Li and Ou Ocean Kun Hei and Yue Yu and Yuyu Luo},
  year          = {2026},
  eprint        = {2604.15163},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DB},
  doi           = {10.48550/arXiv.2604.15163},
  url           = {https://arxiv.org/abs/2604.15163},
  note          = {Accepted to ACL 2026 Main Track}
}
```

Paper link: [https://arxiv.org/abs/2604.15163](https://arxiv.org/abs/2604.15163)

---

## 📄 License

This project is released under the MIT License. See [`LICENSE`](./LICENSE) for details.
