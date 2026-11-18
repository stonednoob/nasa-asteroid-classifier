"""Download near-Earth asteroids from NASA NeoWs (browse endpoint) and build asteroids.csv.

The API key is read from the NASA_API_KEY environment variable and is never
written to disk or printed. Each page is cached in data/cache/ so reruns do not
call the API again.

Usage:  python fetch_data.py [--first-page 0] [--last-page 299]
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://api.nasa.gov/neo/rest/v1/neo/browse"
CACHE = Path("data/cache")
CSV = Path("asteroids.csv")
PAGE_SIZE = 20
MAX_RETRIES = 3


class StopDownload(RuntimeError):
    pass


def redact(text: str) -> str:
    return re.sub(r"api_key=[^&\"\s]+", "api_key=REDACTED", text)


def fetch_page(session, page, key):
    path = CACHE / f"page_{page:03d}.json"
    if path.exists():
        return json.loads(path.read_text()), False
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(URL, params={"page": page, "size": PAGE_SIZE, "api_key": key}, timeout=30)
        except requests.RequestException as exc:
            # Do not print the exception text: it can contain the request URL (with the key).
            print(f"  page {page}: network error ({type(exc).__name__}), attempt {attempt}/{MAX_RETRIES}")
            time.sleep(2 * attempt)
            continue
        if r.status_code in (401, 403):
            raise StopDownload(f"HTTP {r.status_code}: key missing or invalid. Stopping.")
        if r.status_code == 429:
            raise StopDownload("HTTP 429: rate limit reached. Stopping; cached pages are kept.")
        if r.status_code == 200:
            # NeoWs echoes the key inside every "links" URL; redact it before caching.
            text = redact(r.text)
            path.write_text(text)
            return json.loads(text), True
        print(f"  page {page}: HTTP {r.status_code}, attempt {attempt}/{MAX_RETRIES}")
        time.sleep(2 * attempt)
    raise StopDownload(f"page {page}: failed after {MAX_RETRIES} attempts. Stopping.")


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def to_row(neo):
    km = neo.get("estimated_diameter", {}).get("kilometers", {})
    approaches = neo.get("close_approach_data") or []
    miss = [num(a.get("miss_distance", {}).get("astronomical")) for a in approaches]
    speed = [num(a.get("relative_velocity", {}).get("kilometers_per_second")) for a in approaches]
    orb = neo.get("orbital_data") or {}
    return {
        "id": neo.get("id"),
        "name": neo.get("name"),
        "absolute_magnitude_h": num(neo.get("absolute_magnitude_h")),
        "diameter_min_km": num(km.get("estimated_diameter_min")),
        "diameter_max_km": num(km.get("estimated_diameter_max")),
        "closest_miss_au": min(miss) if miss else float("nan"),
        "fastest_velocity_km_s": max(speed) if speed else float("nan"),
        "n_close_approaches": len(approaches),
        "moid_au": num(orb.get("minimum_orbit_intersection")),
        "eccentricity": num(orb.get("eccentricity")),
        "inclination_deg": num(orb.get("inclination")),
        "semi_major_axis_au": num(orb.get("semi_major_axis")),
        "perihelion_distance_au": num(orb.get("perihelion_distance")),
        "aphelion_distance_au": num(orb.get("aphelion_distance")),
        "orbital_period_days": num(orb.get("orbital_period")),
        "is_hazardous": bool(neo.get("is_potentially_hazardous_asteroid")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-page", type=int, default=0)
    ap.add_argument("--last-page", type=int, default=299)
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    pages = range(args.first_page, args.last_page + 1)
    need_api = any(not (CACHE / f"page_{p:03d}.json").exists() for p in pages)
    key = os.environ.get("NASA_API_KEY", "").strip()
    if need_api and not key:
        sys.exit("Set the NASA_API_KEY environment variable first (see README).")

    rows, calls, stopped = [], 0, None
    with requests.Session() as session:
        for page in pages:
            try:
                data, called = fetch_page(session, page, key)
            except StopDownload as exc:
                stopped = str(exc)
                break
            if called:
                calls += 1
                time.sleep(1)  # be polite to the API
            rows.extend(to_row(n) for n in data.get("near_earth_objects", []))
            if page % 25 == 0:
                print(f"page {page}: {len(rows)} asteroids so far ({calls} API calls)")

    if stopped:
        print(stopped)
    df = pd.DataFrame(rows).drop_duplicates("id")
    df.to_csv(CSV, index=False)
    print(f"API calls this run: {calls}")
    print(f"Saved {len(df)} asteroids to {CSV}")
    if stopped:
        sys.exit(1)


if __name__ == "__main__":
    main()
