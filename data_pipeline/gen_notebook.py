#!/usr/bin/env python3
"""
gen_notebook.py
Generates pipeline.ipynb for Module 1 — Data Pipeline.
Run once:  python gen_notebook.py
Then open: pipeline.ipynb in Jupyter or VS Code.
"""
import json
import uuid
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline.ipynb")


# ── helpers ───────────────────────────────────────────────────────────────────
def uid():
    return uuid.uuid4().hex[:8]


def _src(text):
    """Split a code/markdown string into the source list format ipynb expects."""
    lines = text.split("\n")
    result = [l + "\n" for l in lines[:-1]]
    if lines[-1]:
        result.append(lines[-1])
    return result


def md(text):
    return {"cell_type": "markdown", "id": uid(), "metadata": {}, "source": _src(text)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uid(),
        "metadata": {},
        "outputs": [],
        "source": _src(text),
    }


# ── cells ─────────────────────────────────────────────────────────────────────
C = []

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
C.append(md("""\
# Module 1 — Data Pipeline (`/data_pipeline`)
### Zepto Capstone Project — Certificate Program in AI & ML

| Item | Detail |
|---|---|
| Pipeline | scrape → clean → convert → store → query |
| Data source | [books.toscrape.com](http://books.toscrape.com/) — no login, no API key |
| Fixed conversion rate | **1 GBP = 105.50 INR** *(project-defined constant — not a live market rate)* |
| Database | SQLite (`books.db`) — two-table normalised schema (PK/FK) |
| Minimum scope | ≥ 60 books · ≥ 3 categories |

---"""))

# ---------------------------------------------------------------------------
# 0. Imports & Configuration
# ---------------------------------------------------------------------------
C.append(md("## 0 · Imports & Configuration"))

C.append(code("""\
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import time
import os
import re
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", 60)

print("Libraries loaded successfully.")"""))

C.append(code("""\
# ── Project-level constants ────────────────────────────────────────────────
#
# Fixed conversion rate (capstone-defined constant; stated here and in README.md):
#   1 GBP = 105.50 INR
# This is NOT a live or historical market rate; it is an artificial constant
# supplied by the assignment.  No API call or date reference is needed.
#
FIXED_RATE_GBP_TO_INR: float = 105.50

BASE_URL        = "http://books.toscrape.com/"
HEADERS         = {"User-Agent": "Mozilla/5.0 (compatible; ZeptoCapstoneBot/1.0; educational)"}
RATING_MAP      = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
MIN_BOOKS       = 60
MIN_CATEGORIES  = 3
REQUEST_DELAY   = 0.5   # seconds between requests — polite scraping
DB_PATH         = os.path.join(os.path.dirname(os.path.abspath("__file__")), "books.db")

print(f"Fixed GBP → INR rate : {FIXED_RATE_GBP_TO_INR}")
print(f"DB path              : {DB_PATH}")"""))

# ---------------------------------------------------------------------------
# 1. Scraping
# ---------------------------------------------------------------------------
C.append(md("""\
---
## 1 · Scraping — `books.toscrape.com`

**Strategy:** Fetch all category links from the homepage sidebar, then scrape
each category (paginating through all its pages) until we have ≥ 60 books
across ≥ 3 categories.  Each book page supplies: `title`, `price` (GBP),
`star_rating` (text class), `availability`, and `category` (the section we
are in).

`requests` fetches raw HTML; `BeautifulSoup` parses it.  A 0.5 s sleep between
requests keeps load on the practice site polite."""))

C.append(code("""\
def get_category_links(base_url: str) -> list:
    '''Return [{name, url}] for every leaf category in the sidebar.'''
    resp = requests.get(base_url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    nav = soup.find("ul", class_="nav-list")
    categories = []
    if nav:
        # The inner <ul> holds the actual category <li> items
        inner = nav.find("ul")
        if inner:
            for li in inner.find_all("li"):
                a = li.find("a")
                if a:
                    name = a.text.strip()
                    href = a["href"]
                    categories.append({"name": name, "url": base_url + href})
    return categories


def scrape_listing_page(page_url: str) -> tuple:
    '''
    Scrape one listing page.
    Returns (list_of_book_dicts, next_page_url_or_None).
    '''
    resp = requests.get(page_url, headers=HEADERS, timeout=10)
    if resp.status_code != 200:
        print(f"  [WARN] HTTP {resp.status_code} for {page_url} — skipping page")
        return [], None

    soup  = BeautifulSoup(resp.text, "html.parser")
    books = []

    for article in soup.find_all("article", class_="product_pod"):
        try:
            title       = article.h3.a["title"]
            price_text  = article.find("p", class_="price_color").text.strip()

            # Star rating stored as second CSS class: <p class="star-rating Three">
            rating_tag  = article.find("p", class_="star-rating")
            rating_text = rating_tag["class"][1] if rating_tag else "Unknown"

            avail_tag   = (article.find("p", class_="instock availability")
                           or article.find("p", class_=re.compile("availability")))
            availability = avail_tag.text.strip() if avail_tag else "Unknown"

            books.append({
                "title":        title,
                "price_raw":    price_text,
                "star_rating":  rating_text,
                "availability": availability,
            })
        except Exception as exc:
            print(f"  [WARN] Row skipped — {exc}")

    # Follow pagination: <li class="next"><a href="page-2.html">
    next_btn = soup.find("li", class_="next")
    next_url = None
    if next_btn:
        next_href = next_btn.find("a")["href"]
        base_dir  = page_url.rsplit("/", 1)[0]
        next_url  = base_dir + "/" + next_href

    return books, next_url


def scrape_to_dataframe(categories: list, min_books: int = MIN_BOOKS,
                        min_cats: int = MIN_CATEGORIES) -> pd.DataFrame:
    '''
    Iterate through categories until min_books AND min_cats are satisfied.
    Returns a raw DataFrame with a category column added.
    '''
    all_rows   = []
    cats_done  = 0

    for cat in categories:
        name     = cat["name"]
        page_url = cat["url"]
        cat_rows = []

        print(f"  Scraping '{name}' …", end="", flush=True)
        while page_url:
            rows, next_url = scrape_listing_page(page_url)
            cat_rows.extend(rows)
            time.sleep(REQUEST_DELAY)
            page_url = next_url

        for r in cat_rows:
            r["category"] = name

        all_rows.extend(cat_rows)
        cats_done += 1
        print(f" {len(cat_rows)} books  (running total: {len(all_rows)})")

        if cats_done >= min_cats and len(all_rows) >= min_books:
            print(f"\\n  ✓ Target met — {len(all_rows)} books across {cats_done} categories")
            break

    return pd.DataFrame(all_rows)

print("Scraping functions defined.")"""))

C.append(code("""\
print("=" * 55)
print("STEP 1 — Scraping books.toscrape.com")
print("=" * 55)

categories = get_category_links(BASE_URL)
print(f"Found {len(categories)} categories in sidebar.")
print("First 8:", [c['name'] for c in categories[:8]])

df_raw = scrape_to_dataframe(categories)

print(f"\\nRaw DataFrame shape : {df_raw.shape}")
print(f"Columns             : {df_raw.columns.tolist()}")
df_raw.head(5)"""))

# ---------------------------------------------------------------------------
# 2. Data Cleaning
# ---------------------------------------------------------------------------
C.append(md("""\
---
## 2 · Data Cleaning

| Field | Raw | Target type | Failure strategy |
|---|---|---|---|
| `price_gbp` | `'£12.99'` | `float` | Median imputation — keeps rows, preserves dataset size |
| `rating` | `'Three'` | `int` 1–5 | Drop row — can't meaningfully impute ordinal meaning |
| `in_stock` | `'In stock'` | `bool` | Defaults to `False` for any unrecognised text (safe-fail) |

**Rationale for median imputation on `price_gbp`:** price failures are extremely
rare (unexpected currency encoding) and the median is a robust central tendency
measure unaffected by outliers; it keeps the row in the dataset.
**Rationale for dropping on `rating`:** rating is ordinal — there is no
meaningful substitute value to assign; a wrong rating would corrupt downstream
analysis."""))

C.append(code("""\
def parse_price(raw) -> float:
    '''Strip non-numeric chars (currency symbols, spaces) and return float.'''
    try:
        return float(re.sub(r"[^\\d.]", "", str(raw)))
    except (ValueError, TypeError):
        return None


def parse_rating(text) -> int:
    '''Map word rating (One … Five) to integer 1–5; None if unrecognised.'''
    return RATING_MAP.get(str(text).strip(), None)


def parse_in_stock(text) -> bool:
    '''True iff "in stock" appears in the availability string.'''
    return "in stock" in str(text).lower()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── price_gbp (float) ──────────────────────────────────────────────────
    df["price_gbp"] = df["price_raw"].apply(parse_price)
    n_null_price = df["price_gbp"].isna().sum()
    if n_null_price:
        med = df["price_gbp"].median()
        df["price_gbp"].fillna(med, inplace=True)
        print(f"  [INFO] price_gbp: median-imputed {n_null_price} row(s) with £{med:.2f}")
    df["price_gbp"] = df["price_gbp"].astype(float)

    # ── rating (int 1–5) ───────────────────────────────────────────────────
    df["rating"] = df["star_rating"].apply(parse_rating)
    n_drop_rating = df["rating"].isna().sum()
    if n_drop_rating:
        print(f"  [INFO] rating: dropped {n_drop_rating} row(s) with unrecognised text")
        df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].astype(int)

    # ── in_stock (bool) ────────────────────────────────────────────────────
    df["in_stock"] = df["availability"].apply(parse_in_stock)

    # ── Drop raw helper columns ────────────────────────────────────────────
    df = df.drop(columns=["price_raw", "star_rating", "availability"])
    df = df.reset_index(drop=True)
    return df

print("Cleaning functions defined.")"""))

C.append(code("""\
print("=" * 55)
print("STEP 2 — Cleaning")
print("=" * 55)

df_clean = clean_dataframe(df_raw)

print(f"\\nCleaned shape : {df_clean.shape}")
print("\\nColumn dtypes:")
print(df_clean.dtypes)
print("\\nNull counts:")
print(df_clean.isnull().sum())
df_clean.head(5)"""))

# ---------------------------------------------------------------------------
# 3. Currency Conversion
# ---------------------------------------------------------------------------
C.append(md("""\
---
## 3 · Currency Conversion — `price_gbp` → `price_inr`

The project specifies a **fixed baseline conversion rate**:

> **1 GBP = 105.50 INR**

This is an artificial, project-defined constant — not a live or historical
market rate.  No external API call is required.  The rate is also stated
explicitly in `README.md`."""))

C.append(code("""\
print("=" * 55)
print("STEP 3 — Currency Conversion")
print("=" * 55)
print(f"Fixed rate applied: 1 GBP = {FIXED_RATE_GBP_TO_INR} INR")

df_final = df_clean.copy()
df_final["price_inr"] = (df_final["price_gbp"] * FIXED_RATE_GBP_TO_INR).round(2)

print("\\nSample conversions:")
print(df_final[["title", "price_gbp", "price_inr"]].head(8).to_string(index=False))

# ── Acceptance checks ──────────────────────────────────────────────────────
assert len(df_final) >= 60,                    f"Need ≥60 books, got {len(df_final)}"
assert df_final["category"].nunique() >= 3,    "Need ≥3 categories"
assert df_final["price_gbp"].dtype == float,   "price_gbp must be float"
assert df_final["rating"].dtype   in [int, object] or str(df_final["rating"].dtype).startswith("int"), "rating must be int"
assert "price_inr" in df_final.columns,        "price_inr column missing"

print(f"\\n✓ Acceptance checks passed")
print(f"  Books     : {len(df_final)}")
print(f"  Categories: {df_final['category'].nunique()} → {df_final['category'].unique().tolist()}")
print(f"  price_inr range: ₹{df_final['price_inr'].min():.2f} – ₹{df_final['price_inr'].max():.2f}")"""))

# ---------------------------------------------------------------------------
# 4. SQLite Schema + Load
# ---------------------------------------------------------------------------
C.append(md("""\
---
## 4 · SQLite Schema & Data Load

**Normalised two-table design** with a primary-key / foreign-key relationship:

```
categories (category_id PK, category_name UNIQUE)
    │
    └─ books (book_id PK, title, price_gbp, price_inr,
              rating, in_stock, category_id FK → categories)
```

Separating categories eliminates repeated string storage and makes
`JOIN`-based queries straightforward.  `FOREIGN KEY` enforcement is enabled
via `PRAGMA foreign_keys = ON`."""))

C.append(code("""\
SCHEMA_SQL = (
    "PRAGMA foreign_keys = ON; "
    "CREATE TABLE IF NOT EXISTS categories ("
    "    category_id   INTEGER PRIMARY KEY AUTOINCREMENT,"
    "    category_name TEXT    NOT NULL UNIQUE"
    "); "
    "CREATE TABLE IF NOT EXISTS books ("
    "    book_id     INTEGER PRIMARY KEY AUTOINCREMENT,"
    "    title       TEXT    NOT NULL,"
    "    price_gbp   REAL    NOT NULL,"
    "    price_inr   REAL    NOT NULL,"
    "    rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),"
    "    in_stock    INTEGER NOT NULL DEFAULT 1,"
    "    category_id INTEGER NOT NULL,"
    "    FOREIGN KEY (category_id) REFERENCES categories (category_id)"
    ");"
)


def create_database(db_path: str) -> sqlite3.Connection:
    '''Drop-and-recreate the database, return open connection.'''
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    # executescript requires statements separated by semicolons
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS categories ("
        "category_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "category_name TEXT NOT NULL UNIQUE)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS books ("
        "book_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "title TEXT NOT NULL,"
        "price_gbp REAL NOT NULL,"
        "price_inr REAL NOT NULL,"
        "rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),"
        "in_stock INTEGER NOT NULL DEFAULT 1,"
        "category_id INTEGER NOT NULL,"
        "FOREIGN KEY (category_id) REFERENCES categories (category_id))"
    )
    conn.commit()
    return conn


def load_data(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    '''Insert categories then books, using a name→id map to resolve FK.'''
    cur = conn.cursor()

    # 1. Insert unique categories
    unique_cats = df["category"].unique().tolist()
    cur.executemany("INSERT OR IGNORE INTO categories (category_name) VALUES (?)",
                    [(c,) for c in unique_cats])
    conn.commit()

    # 2. Build name → category_id lookup
    cat_map = dict(cur.execute(
        "SELECT category_name, category_id FROM categories").fetchall())

    # 3. Insert books
    rows = [
        (row.title, float(row.price_gbp), float(row.price_inr),
         int(row.rating), int(row.in_stock), cat_map[row.category])
        for row in df.itertuples(index=False)
    ]
    cur.executemany(
        "INSERT INTO books "
        "(title, price_gbp, price_inr, rating, in_stock, category_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()

print("Database functions defined.")"""))

C.append(code("""\
print("=" * 55)
print("STEP 4 — Create SQLite database & load data")
print("=" * 55)

conn = create_database(DB_PATH)
load_data(conn, df_final)

n_books = pd.read_sql_query("SELECT COUNT(*) AS n FROM books", conn).iloc[0, 0]
n_cats  = pd.read_sql_query("SELECT COUNT(*) AS n FROM categories", conn).iloc[0, 0]

print(f"✓ Loaded {n_books} books into {n_cats} category rows")
print(f"  DB file : {os.path.abspath(DB_PATH)}")

# Show schema
print("\\n-- categories table (sample) --")
print(pd.read_sql_query("SELECT * FROM categories", conn).to_string(index=False))
print("\\n-- books table (first 5 rows) --")
print(pd.read_sql_query("SELECT * FROM books LIMIT 5", conn).to_string(index=False))"""))

# ---------------------------------------------------------------------------
# 5. SQL Queries
# ---------------------------------------------------------------------------
C.append(md("""\
---
## 5 · SQL Queries

Six queries collectively cover every required clause:
`SELECT`/`WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `IN`, `BETWEEN`, and at
least one `JOIN`.  Each query is executed via `pd.read_sql_query` and the
result is printed inline."""))

C.append(code("""\
def run_query(label: str, sql: str) -> pd.DataFrame:
    '''Execute a SQL query, print the label + results, return DataFrame.'''
    print(f"\\n{'─'*60}")
    print(f"▶ {label}")
    print(f"  SQL: {sql.strip()[:120]}...")
    df = pd.read_sql_query(sql, conn)
    print(f"  Rows returned: {len(df)}")
    print(df.head(15).to_string(index=False))
    return df

print("Query helper defined.")"""))

C.append(code("""\
# ── Q1 · SELECT + WHERE + ORDER BY ────────────────────────────────────────
q1_sql = '''
    SELECT title, price_gbp, price_inr, rating, in_stock
    FROM   books
    WHERE  price_gbp < 10.0
      AND  in_stock = 1
    ORDER  BY price_gbp ASC
'''
df_q1 = run_query("Q1 — Affordable in-stock books  (SELECT + WHERE + ORDER BY)", q1_sql)"""))

C.append(code("""\
# ── Q2 · ORDER BY + LIMIT ─────────────────────────────────────────────────
q2_sql = '''
    SELECT title, price_gbp, price_inr, rating
    FROM   books
    ORDER  BY price_gbp DESC
    LIMIT  10
'''
df_q2 = run_query("Q2 — Top 10 most expensive books  (ORDER BY + LIMIT)", q2_sql)"""))

C.append(code("""\
# ── Q3 · DISTINCT + JOIN + GROUP BY ──────────────────────────────────────
q3_sql = '''
    SELECT DISTINCT c.category_name,
           COUNT(b.book_id)            AS total_books,
           ROUND(AVG(b.price_gbp), 2)  AS avg_price_gbp,
           ROUND(AVG(b.rating), 2)     AS avg_rating
    FROM   categories c
    JOIN   books b ON c.category_id = b.category_id
    GROUP  BY c.category_name
    ORDER  BY total_books DESC
'''
df_q3 = run_query("Q3 — Category summary (DISTINCT + JOIN + GROUP BY)", q3_sql)"""))

C.append(code("""\
# ── Q4 · BETWEEN ─────────────────────────────────────────────────────────
q4_sql = '''
    SELECT title, price_gbp, price_inr, rating
    FROM   books
    WHERE  rating    BETWEEN 3 AND 5
      AND  price_gbp BETWEEN 10.0 AND 35.0
    ORDER  BY rating DESC, price_gbp ASC
'''
df_q4 = run_query("Q4 — Mid-price, well-rated books  (BETWEEN)", q4_sql)"""))

C.append(code("""\
# ── Q5 · IN + LIMIT ──────────────────────────────────────────────────────
q5_sql = '''
    SELECT title, price_gbp, price_inr, rating
    FROM   books
    WHERE  rating IN (4, 5)
    ORDER  BY rating DESC, price_gbp DESC
    LIMIT  20
'''
df_q5 = run_query("Q5 — Top-rated books in selected buckets  (IN + LIMIT)", q5_sql)"""))

C.append(code("""\
# ── Q6 · Full JOIN — used for pd.merge comparison in Section 6 ────────────
q6_sql = '''
    SELECT c.category_name,
           b.title,
           b.price_gbp,
           b.price_inr,
           b.rating,
           b.in_stock
    FROM   books b
    JOIN   categories c ON b.category_id = c.category_id
    ORDER  BY c.category_name ASC, b.rating DESC, b.price_gbp ASC
'''
df_q6 = run_query("Q6 — Full books + category JOIN  (used for pd.merge comparison)", q6_sql)"""))

# ---------------------------------------------------------------------------
# 6. pd.read_sql vs pd.merge
# ---------------------------------------------------------------------------
C.append(md("""\
---
## 6 · `pd.read_sql` vs `pd.merge` — Equivalence Demonstration

Two approaches to the same join:

| Approach | Mechanism |
|---|---|
| **A** | `pd.read_sql_query(sql_with_JOIN, conn)` — lets the DB engine do the join |
| **B** | Read both tables into memory, then `pd.merge(books, categories, on='category_id')` |

Both should produce identical DataFrames."""))

C.append(code("""\
print("=" * 60)
print("APPROACH A — pd.read_sql  (SQL JOIN executed in SQLite)")
print("=" * 60)

join_sql = '''
    SELECT c.category_name,
           b.title,
           b.price_gbp,
           b.price_inr,
           b.rating
    FROM   books b
    JOIN   categories c ON b.category_id = c.category_id
    ORDER  BY c.category_name ASC, b.rating DESC, b.price_gbp ASC
'''
df_sql = pd.read_sql_query(join_sql, conn)
print(f"Shape: {df_sql.shape}")
print(df_sql.head(10).to_string(index=False))"""))

C.append(code("""\
print("=" * 60)
print("APPROACH B — pd.merge  (in-memory join on category_id)")
print("=" * 60)

df_books_mem = pd.read_sql_query("SELECT * FROM books", conn)
df_cats_mem  = pd.read_sql_query("SELECT * FROM categories", conn)

df_merge = (
    df_books_mem
    .merge(df_cats_mem, on="category_id", how="inner")
    [["category_name", "title", "price_gbp", "price_inr", "rating"]]
    .sort_values(["category_name", "rating", "price_gbp"],
                 ascending=[True, False, True])
    .reset_index(drop=True)
)
print(f"Shape: {df_merge.shape}")
print(df_merge.head(10).to_string(index=False))"""))

C.append(code("""\
print("=" * 60)
print("EQUIVALENCE CHECK")
print("=" * 60)

df_a = df_sql.reset_index(drop=True)
df_b = df_merge.reset_index(drop=True)

# Align column order for comparison
df_b = df_b[df_a.columns]

match = df_a.equals(df_b)
print(f"DataFrames identical: {match}")

if not match:
    print("\\nDifferences (first 5):")
    diff = df_a.compare(df_b)
    print(diff.head(5))
else:
    print("\\n✓ Both approaches return the same rows in the same order.")
    print("  pd.read_sql (SQL JOIN) == pd.merge (in-memory join)")"""))

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
C.append(md("---\n## 7 · Pipeline Summary"))

C.append(code("""\
conn.close()

print("=" * 60)
print("MODULE 1 — PIPELINE COMPLETE")
print("=" * 60)
print(f"  Total books loaded    : {len(df_final)}")
print(f"  Categories            : {df_final['category'].nunique()}")
cats_list = df_final['category'].unique().tolist()
for c in cats_list:
    n = (df_final['category'] == c).sum()
    print(f"    • {c:<30} {n} books")
print(f"  price_gbp range       : £{df_final['price_gbp'].min():.2f} – £{df_final['price_gbp'].max():.2f}")
print(f"  price_inr range       : ₹{df_final['price_inr'].min():.2f} – ₹{df_final['price_inr'].max():.2f}")
print(f"  Rating distribution   :")
print(df_final['rating'].value_counts().sort_index().to_string())
print(f"  In-stock ratio        : {df_final['in_stock'].mean():.1%}")
print(f"  SQLite DB             : {os.path.abspath(DB_PATH)}")
print(f"  Fixed GBP→INR rate    : {FIXED_RATE_GBP_TO_INR} (project constant)")
print("=" * 60)"""))


# ── Assemble and write notebook ────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": C,
}

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(notebook, fh, indent=1, ensure_ascii=False)

print(f"✓ Notebook written to: {OUT}")
print(f"  Cells: {len(C)}")
