#!/usr/bin/env python3
"""Rebuild the known-fusion catalogue from the Mitelman Database.

GATFuse ships a catalogue snapshot, so this script is only needed to refresh it.
The output is the two-partner-plus-count TSV that ``--known-fusions`` expects:

    gene_a  gene_b  n_cases

    python scripts/update_known_fusions.py -o mitelman_fusions.tsv

Only the standard library is used, so it runs without installing GATFuse.

The Mitelman Database of Chromosome Aberrations and Gene Fusions in Cancer is
curated by Mitelman F, Johansson B and Mertens F and hosted by the NCI and
ISB-CGC (https://mitelmandatabase.isb-cgc.org). Cite it when you use the
catalogue, and respect its terms of use.

Note that a model trained against one snapshot expects that snapshot's feature
distribution; refreshing the catalogue means retraining.
"""

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from itertools import combinations

BASE_URL = "https://mitelmandatabase.isb-cgc.org"
PAGE_QUERY_URL = f"{BASE_URL}/page_query"
RESULT_URL = f"{BASE_URL}/result"

# What the search form submits with no filter applied, i.e. every gene fusion.
SEARCH_CRITERIA = {
    "abnorm_op": "a",
    "break_op": "a",
    "gene_op": "a",
    "op": "M",
    "search_type": "mb",
}
TABLE_ID = "mb_result"
PAGE_SIZE = 5000

HEADERS = {
    "User-Agent": "GATFuse known-fusion updater",
    "Content-Type": "application/x-www-form-urlencoded",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": RESULT_URL,
}


def _post_json(url, payload, timeout=60, retries=3):
    request = urllib.request.Request(
        url, data=urllib.parse.urlencode(payload).encode(), headers=HEADERS
    )
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt < retries:
                delay = 5 * attempt
                print(f"  request failed ({error}); retrying in {delay}s", file=sys.stderr)
                time.sleep(delay)
    raise RuntimeError(f"Could not fetch {url} after {retries} attempts: {last_error}")


def _open_session(timeout=30):
    """Submit the search form so the server materialises the result set."""
    payload = {
        "search_type": "mb",
        "genes_mb": "",
        "abNormOptions": "a",
        "brOptions": "a",
        "geneRadios": "a",
    }
    request = urllib.request.Request(
        RESULT_URL,
        data=urllib.parse.urlencode(payload).encode(),
        headers={
            "User-Agent": HEADERS["User-Agent"],
            "Content-Type": HEADERS["Content-Type"],
            "Referer": f"{BASE_URL}/mb_search",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
    except (urllib.error.URLError, TimeoutError) as error:
        print(f"  session initialisation failed ({error}); continuing", file=sys.stderr)


def fetch_records(delay=1.0):
    """Download every fusion case, one page at a time."""
    print("Opening session", file=sys.stderr)
    _open_session()

    def page(start, draw):
        return _post_json(PAGE_QUERY_URL, {
            "criteria": json.dumps(SEARCH_CRITERIA),
            "table_id": TABLE_ID,
            "start": start,
            "length": PAGE_SIZE,
            "draw": draw,
        })

    first = page(0, 1)
    total = int(first.get("recordsTotal", 0))
    if total == 0:
        raise RuntimeError(
            "The server returned no records. The database interface may have changed."
        )
    print(f"Records available: {total:,}", file=sys.stderr)

    records = list(first["data"])
    start, draw = PAGE_SIZE, 2
    while start < total:
        print(f"Fetching {start + 1:,}-{min(start + PAGE_SIZE, total):,}", file=sys.stderr)
        time.sleep(delay)
        records.extend(page(start, draw)["data"])
        start += PAGE_SIZE
        draw += 1
    return records


def parse_gene_pairs(gene_short):
    """Extract gene pairs from one record's fusion description.

    Fusions in a case are comma-separated and partners are joined by '::'.
    A fusion naming three or more genes contributes every pairwise combination.
    """
    text = (gene_short or "").strip()
    if not text or text in {"N/A", "-"}:
        return []

    pairs = []
    for fusion in text.split(","):
        genes = [gene.strip() for gene in fusion.split("::") if gene.strip()]
        if len(genes) < 2:
            continue
        pairs.extend(combinations(sorted(genes), 2))
    return pairs


def aggregate(records):
    """Count the cases reported for each unordered gene pair."""
    counts = defaultdict(int)
    for record in records:
        for pair in parse_gene_pairs(record.get("GeneShort", "")):
            counts[tuple(sorted(gene.upper() for gene in pair))] += 1
    return counts


def write_tsv(counts, path):
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["gene_a", "gene_b", "n_cases"])
        for (gene_a, gene_b), cases in ordered:
            writer.writerow([gene_a, gene_b, cases])
    print(f"Wrote {len(ordered):,} gene pairs to {path}", file=sys.stderr)
    return ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("-o", "--output", default="mitelman_fusions.tsv",
                        help="output TSV (default: mitelman_fusions.tsv)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="seconds between page requests (default: 1.0)")
    args = parser.parse_args()

    try:
        counts = aggregate(fetch_records(delay=args.delay))
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if not counts:
        print("error: no gene pairs could be parsed from the response", file=sys.stderr)
        return 1

    ordered = write_tsv(counts, args.output)
    print("Most recurrent pairs:", file=sys.stderr)
    for (gene_a, gene_b), cases in ordered[:5]:
        print(f"  {gene_a}::{gene_b}  {cases} cases", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
