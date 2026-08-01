# Module 1 — Data Pipeline

**Zepto Capstone Project · Certificate Program in AI & ML**

## Overview

End-to-end data-engineering pipeline: scrape raw book data from a public
practice site, clean and type-coerce every field, enrich with a fixed currency
conversion, load into a normalised SQLite database, and query it using both
SQL and pandas — identical mechanics to a production catalog-intelligence
pipeline.

---

## Fixed Currency Conversion Rate

> **1 GBP = 105.50 INR**

This is an **artificial, project-defined constant** supplied by the assignment.
It is **not** a live or historical market rate. No API call, network access, or
date reference is required or used.

---

## Repository Layout

```
data_pipeline/
├── pipeline.ipynb       ← main notebook (run this)
├── books.db             ← SQLite database (generated on first run)
├── gen_notebook.py      ← helper that regenerates pipeline.ipynb
└── README.md            ← this file
```

---

## Setup

```bash
# From the repo root (or data_pipeline/ folder)
pip install requests beautifulsoup4 pandas

# No other dependencies — sqlite3 is part of the Python standard library.
```

Python ≥ 3.9 recommended.

---

## How to Run

### Option A — Jupyter (recommended)

```bash
pip install jupyter
cd data_pipeline
jupyter notebook pipeline.ipynb
# Run All Cells  (Kernel → Restart & Run All)
```

### Option B — VS Code

Open `pipeline.ipynb` in VS Code with the Jupyter extension, then
**Run All Cells**.

### Option C — Regenerate from scratch

```bash
cd data_pipeline
python gen_notebook.py          # recreates pipeline.ipynb
jupyter nbconvert --to notebook --execute pipeline.ipynb --output pipeline.ipynb
```

---

## What Each Section Does

| Section | Description |
|---|---|
| **0 · Imports & Config** | Libraries, project constants, fixed conversion rate |
| **1 · Scraping** | `requests` + `BeautifulSoup`; ≥ 3 categories, ≥ 60 books |
| **2 · Cleaning** | `price_gbp` (float), `rating` (int 1–5), `in_stock` (bool) |
| **3 · Conversion** | `price_inr = price_gbp × 105.50` |
| **4 · SQLite** | Two-table schema (`categories` + `books`), FK enforced |
| **5 · SQL Queries** | 6 queries covering SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN, IN, JOIN |
| **6 · pandas** | `pd.read_sql` vs `pd.merge` — equivalence proof |
| **7 · Summary** | Final stats and acceptance-criteria recap |

---

## Design Decisions

### Cleaning strategy

| Field | Parse failure action | Rationale |
|---|---|---|
| `price_gbp` | **Median imputation** | Keeps the row; median is robust to outliers and price failures are rare encoding issues |
| `rating` | **Drop row** | Ordinal field — no meaningful substitute value exists; a wrong rating corrupts downstream analysis |
| `in_stock` | **Default `False`** | Safe-fail: ambiguous availability is treated conservatively |

### Database normalisation

Separating `categories` into its own table eliminates repeated string storage,
enables `JOIN`-based aggregation, and enforces referential integrity via the
`FOREIGN KEY` constraint (activated with `PRAGMA foreign_keys = ON`).

### Scraping approach

Categories are scraped in sidebar order until both conditions are met:
≥ 60 books **and** ≥ 3 categories. This avoids hardcoding specific categories
and handles pagination automatically within each category.

---

## Acceptance Criteria

- [x] ≥ 60 books across ≥ 3 categories
- [x] `price_gbp` (`float`), `rating` (`int` 1–5), `in_stock` (`bool`), `price_inr` (`float`) all present and correctly typed
- [x] `price_inr` computed from fixed rate **1 GBP = 105.50 INR** (stated above)
- [x] SQLite database with two-table PK/FK schema
- [x] ≥ 5 SQL queries covering SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, IN, BETWEEN, JOIN
- [x] `pd.read_sql` and `pd.merge` outputs shown side-by-side and confirmed equivalent
- [x] README documents install/run steps, design decisions, and fixed conversion rate
