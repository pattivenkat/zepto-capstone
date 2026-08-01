# Zepto Data & AI Platform
### Capstone Project — Certificate Program in Artificial Intelligence and Machine Learning

---

## Project Structure

```
zepto-capstone/
├── README.md                    ← this file (root)
│
├── data_pipeline/               ← Module 1 (25 marks)
│   ├── pipeline.ipynb           ← main notebook
│   ├── books.db                 ← SQLite database (generated on run)
│   ├── gen_notebook.py          ← notebook generator helper
│   └── README.md
│
├── analytics/                   ← Module 2 (50 marks) — coming soon
│
└── support_assistant/           ← Module 3 (25 marks) — coming soon
```

---

## Module Summaries

### Module 1 — Data Pipeline (`/data_pipeline`) · 25 marks

End-to-end ETL pipeline:
- **Scrape** book data from `books.toscrape.com` (≥ 60 books, ≥ 3 categories) using `requests` + `BeautifulSoup`
- **Clean** fields into proper types (`price_gbp` float, `rating` int 1–5, `in_stock` bool)
- **Convert** GBP → INR using fixed project rate: **1 GBP = 105.50 INR**
- **Load** into a normalised SQLite database (two-table PK/FK schema)
- **Query** with 6 SQL queries covering all required clauses + JOIN
- **Compare** `pd.read_sql` vs `pd.merge` for equivalence

See [`data_pipeline/README.md`](data_pipeline/README.md) for full setup and run instructions.

### Module 2 — Analytics Pipeline (`/analytics`) · 50 marks

*In progress — will cover EDA, feature engineering, model training, and evaluation.*

### Module 3 — GenAI Support Assistant (`/support_assistant`) · 25 marks

*In progress — will cover RAG-based policy Q&A using Zepto's documents.*

---

## Setup

### Dependencies per module

Each module has its own dependencies. Install them individually:

```bash
# Module 1
pip install requests beautifulsoup4 pandas

# Module 2 (coming soon)
# pip install pandas numpy scikit-learn matplotlib seaborn

# Module 3 (coming soon)
# pip install langchain chromadb openai sentence-transformers
```

Or install everything at once once all modules are complete:

```bash
pip install -r requirements.txt
```

### Python version

Python 3.9 or higher recommended. `sqlite3` is bundled with the standard library.

---

## Running Each Module

### Module 1 — Data Pipeline

```bash
cd data_pipeline
jupyter notebook pipeline.ipynb
# Kernel → Restart & Run All
```

### Module 2 — Analytics (coming soon)

```bash
cd analytics
jupyter notebook analytics.ipynb
```

### Module 3 — Support Assistant (coming soon)

```bash
cd support_assistant
python app.py
```

---

## Key Design Decisions

### Module 1

| Decision | Choice | Rationale |
|---|---|---|
| Scraping scope | Category-by-category until ≥ 60 books & ≥ 3 categories | Avoids hardcoding; handles pagination cleanly |
| `price_gbp` parse failure | Median imputation | Keeps rows; median is outlier-robust |
| `rating` parse failure | Drop row | Can't meaningfully impute ordinal values |
| Currency conversion | Fixed rate: 1 GBP = 105.50 INR | Project-defined constant; no API needed |
| DB normalisation | `categories` + `books` with FK | Eliminates string repetition; enforces referential integrity |

---

## Git Workflow

This repository follows the feature-branch workflow required by the assignment:

- `main` — stable, submission-ready code
- `feature/data-pipeline` — Module 1 development branch (merged into main)
- Additional feature branches per module as development proceeds

The commit history shows at least one feature branch created, committed to
twice, and merged back into `main` — verifiable via:

```bash
git log --graph --all --oneline
```

---

## Academic Integrity

All code, analysis, and written interpretations are original work authored for
this submission. Standard library and framework documentation were referenced;
all reasoning and implementation are my own.

---

*Submission deadline: as communicated on the LMS.*
