"""
Data plumbing: cached fetchers for every free source.

Each fetcher writes one JSON file per (source, sector) under data/raw/<source>/.
If the file already exists it is reused, so re-running a notebook never
re-downloads. Delete a file to force a refresh.

Run a whole source from the shell (useful for the slow, rate-limited ones):
    python -m radar.fetch gtrends
"""
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import quote

import requests

from .config import (END_YEAR, GT_TIMEFRAME, RAW, SEA_GEOS, SEA_WIKIS, SECTORS,
                     SF_GEO, START_YEAR, US_GEO, USER_AGENT, WB_COUNTRIES,
                     WB_INDICATORS)

S = requests.Session()
S.headers["User-Agent"] = USER_AGENT


def _cache(source, key):
    d = RAW / source
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def _get(url, params=None, tries=6, **kw):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=90, **kw)
            if r.status_code == 200:
                return r
            if r.status_code in (403, 429, 500, 502, 503):
                time.sleep(min(300, 10 * 2 ** i))
                continue
            return r
        except requests.RequestException:
            time.sleep(5 * 2 ** i)
    raise RuntimeError(f"failed: {url} {params}")


# ---------------------------------------------------------------- Y Combinator
def fetch_yc():
    """Full YC company directory (yc-oss mirror of the public YC directory)."""
    f = _cache("yc", "all_companies")
    if not f.exists():
        r = _get("https://yc-oss.github.io/api/companies/all.json")
        f.write_text(r.text)
    return json.loads(f.read_text())


# ---------------------------------------------------------------- Hacker News
def _quarters():
    for y in range(START_YEAR, END_YEAR + 1):
        for q in range(4):
            start = datetime(y, 3 * q + 1, 1, tzinfo=timezone.utc)
            end = datetime(y + (q == 3), (3 * q + 3) % 12 + 1, 1, tzinfo=timezone.utc)
            if start > datetime(2026, 9, 30, tzinfo=timezone.utc):
                return
            yield f"{y}Q{q+1}", int(start.timestamp()), int(end.timestamp())


def _hn_count(phrase, t0, t1):
    r = _get("https://hn.algolia.com/api/v1/search", params={
        "query": f'"{phrase}"', "advancedSyntax": "true", "typoTolerance": "false",
        "tags": "(story,comment)", "hitsPerPage": 0,
        "numericFilters": f"created_at_i>={t0},created_at_i<{t1}"})
    return r.json()["nbHits"]


def _years():
    for y in range(START_YEAR, END_YEAR + 1):
        t0 = datetime(y, 1, 1, tzinfo=timezone.utc)
        t1 = datetime(y + 1, 1, 1, tzinfo=timezone.utc) if y < END_YEAR else datetime(y, 10, 1, tzinfo=timezone.utc)
        yield str(y), int(t0.timestamp()), int(t1.timestamp())


def fetch_hn(sector):
    """Yearly count of HN stories+comments mentioning any of the sector phrases.
    (The first sectors fetched used quarterly keys like '2015Q3'; notebook 01 handles both.)"""
    f = _cache("hn", sector)
    if f.exists():
        return json.loads(f.read_text())
    out = {}
    for phrase in SECTORS[sector]["hn"]:
        for y, t0, t1 in _years():
            out.setdefault(y, {})[phrase] = _hn_count(phrase, t0, t1)
            time.sleep(0.3)                       # stay well under Algolia's 10k/hour
    f.write_text(json.dumps(out))
    return out


# ---------------------------------------------------------------- Google Trends
def _gt_multi(items):
    """One Google Trends request comparing up to 5 (term, geo) pairs.
    Returns {geo: {YYYY-MM: value}}. Values share one 0-100 scale inside the request;
    only each series' *shape* is used downstream, so that is fine."""
    from pytrends.request import TrendReq
    py = TrendReq(hl="en-US", tz=0, timeout=(10, 60), retries=0)
    py.kw_list = [items[0][0]]
    py.geo = items[0][1]
    py.token_payload = {"hl": py.hl, "tz": py.tz,
                        "req": json.dumps({"comparisonItem": [{"keyword": k, "time": GT_TIMEFRAME, "geo": g} for k, g in items],
                                           "category": 0, "property": ""})}
    py._tokens()
    py.kw_list = [f"{k}|{g}" for k, g in items]
    df = py.interest_over_time()
    out = {}
    for i, (k, g) in enumerate(items):
        col = df.columns[i] if not df.empty else None
        out[g] = {} if col is None else {d.strftime("%Y-%m"): int(v) for d, v in df[col].items()}
    return out


def fetch_gtrends(sector):
    """Monthly Google Trends interest for SF, US and 6 SEA countries in 2 batched requests."""
    f = _cache("gtrends", sector)
    data = json.loads(f.read_text()) if f.exists() else {}
    cfg = SECTORS[sector]
    batches = [[(cfg["gt_sf"], SF_GEO), (cfg["gt_sf"], US_GEO), (cfg["gt_sea"], "SG")],
               [(cfg["gt_sea"], g) for g in SEA_GEOS if g != "SG"]]
    for items in batches:
        if all(g in data for _, g in items):
            continue
        for i in range(8):
            try:
                data.update(_gt_multi(items))
                break
            except Exception as e:
                wait = min(1800, 300 * 2 ** i) + random.uniform(0, 60)
                print(f"  gtrends {sector}: {type(e).__name__}, sleeping {wait:.0f}s", flush=True)
                time.sleep(wait)
        f.write_text(json.dumps(data))
        time.sleep(random.uniform(45, 75))
    return data


def fetch_gtrends_single(sector):
    """Monthly Google Trends interest (0-100, shape only) for SF, US and 6 SEA countries."""
    from pytrends.request import TrendReq
    f = _cache("gtrends", sector)
    data = json.loads(f.read_text()) if f.exists() else {}
    cfg = SECTORS[sector]
    jobs = [(SF_GEO, cfg["gt_sf"]), (US_GEO, cfg["gt_sf"])] + [(g, cfg["gt_sea"]) for g in SEA_GEOS]
    for geo, term in jobs:
        if geo in data:
            continue
        # the US series is only a fallback for when the SF metro series is too sparse
        if geo == US_GEO and data.get(SF_GEO) and sum(v == 0 for v in data[SF_GEO].values()) < 0.3 * len(data[SF_GEO]):
            continue
        for i in range(8):
            try:
                py = TrendReq(hl="en-US", tz=0, timeout=(10, 60))
                py.build_payload([term], timeframe=GT_TIMEFRAME, geo=geo)
                df = py.interest_over_time()
                data[geo] = {} if df.empty else {d.strftime("%Y-%m"): int(v) for d, v in df[term].items()}
                break
            except Exception as e:                # 429s are common; back off hard
                wait = min(900, 120 * 2 ** i) + random.uniform(0, 30)
                print(f"  gtrends {sector}/{geo}: {type(e).__name__}, sleeping {wait:.0f}s", flush=True)
                time.sleep(wait)
        else:
            continue
        f.write_text(json.dumps(data))
        time.sleep(random.uniform(30, 45))      # Google blocks bursts; slow and steady wins
    return data


# ---------------------------------------------------------------- Wikipedia
def _wikidata_sitelinks(title):
    r = _get("https://www.wikidata.org/w/api.php", params={
        "action": "wbgetentities", "sites": "enwiki", "titles": title,
        "props": "sitelinks", "format": "json", "redirects": "yes"})
    ent = list(r.json()["entities"].values())[0]
    return {k[:-4]: v["title"] for k, v in ent.get("sitelinks", {}).items()
            if k[:-4] in ["en"] + SEA_WIKIS and k.endswith("wiki")}


def fetch_wiki(sector):
    """Monthly human pageviews (Jul 2015 onward) in English + SEA-language Wikipedias."""
    f = _cache("wiki", sector)
    if f.exists():
        return json.loads(f.read_text())
    out = {}
    for lang, title in _wikidata_sitelinks(SECTORS[sector]["wiki"]).items():
        url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
               f"{lang}.wikipedia/all-access/user/{quote(title.replace(' ', '_'), safe='')}"
               "/monthly/20150701/20260930")
        r = _get(url)
        if r.status_code == 200:
            out[lang] = {i["timestamp"][:4] + "-" + i["timestamp"][4:6]: i["views"] for i in r.json()["items"]}
        time.sleep(0.2)
    f.write_text(json.dumps(out))
    return out


# ---------------------------------------------------------------- GitHub
def fetch_github(sector):
    """Yearly count of new public repos matching the sector phrase (builder activity)."""
    f = _cache("github", sector)
    data = json.loads(f.read_text()) if f.exists() else {}
    phrase = SECTORS[sector]["gh"]
    for y in range(2009, END_YEAR + 1):
        if str(y) in data:
            continue
        end = f"{y}-09-30" if y == END_YEAR else f"{y}-12-31"
        q = f'"{phrase}" in:name,description,topics created:{y}-01-01..{end}'
        for i in range(8):
            r = S.get("https://api.github.com/search/repositories",
                      params={"q": q, "per_page": 1}, timeout=60)
            if r.status_code == 200:
                data[str(y)] = r.json()["total_count"]
                break
            reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
            time.sleep(max(5, reset - time.time() + 2))
        f.write_text(json.dumps(data))
        time.sleep(6.5)                            # 10 searches/min unauthenticated
    return data


# ---------------------------------------------------------------- OpenAlex
def fetch_openalex(sector):
    """Yearly count of research works with the phrase in title/abstract."""
    f = _cache("openalex", sector)
    if f.exists():
        return json.loads(f.read_text())
    r = _get("https://api.openalex.org/works", params={
        "filter": f'title_and_abstract.search:"{SECTORS[sector]["oa"]}"',
        "group_by": "publication_year", "per_page": 200})
    out = {g["key"]: g["count"] for g in r.json()["group_by"]}
    f.write_text(json.dumps(out))
    time.sleep(0.5)
    return out


# ---------------------------------------------------------------- World Bank
def fetch_worldbank():
    f = _cache("worldbank", "indicators")
    if f.exists():
        return json.loads(f.read_text())
    out = []
    for code, name in WB_INDICATORS.items():
        r = _get(f"https://api.worldbank.org/v2/country/{';'.join(WB_COUNTRIES)}/indicator/{code}",
                 params={"format": "json", "date": "2005:2025", "per_page": 1000})
        for row in r.json()[1] or []:
            if row["value"] is not None:
                out.append({"country": row["countryiso3code"], "year": int(row["date"]),
                            "indicator": name, "value": row["value"]})
    f.write_text(json.dumps(out))
    return out


# ---------------------------------------------------------------- Denominators
def fetch_denominators():
    """Total platform activity per period, so sector counts become *shares*.
    Without this, every topic looks like a 'trend' simply because HN, GitHub
    and the research literature all grew enormously since 2007."""
    f = _cache("denominators", "totals")
    data = json.loads(f.read_text()) if f.exists() else {}
    if "hn" not in data:
        data["hn"] = {}
        for q, t0, t1 in _quarters():
            r = _get("https://hn.algolia.com/api/v1/search", params={
                "tags": "(story,comment)", "hitsPerPage": 0,
                "numericFilters": f"created_at_i>={t0},created_at_i<{t1}"})
            data["hn"][q] = r.json()["nbHits"]
            time.sleep(0.4)
        f.write_text(json.dumps(data))
    if "openalex" not in data:
        r = _get("https://api.openalex.org/works", params={"group_by": "publication_year", "per_page": 200})
        data["openalex"] = {g["key"]: g["count"] for g in r.json()["group_by"]}
        f.write_text(json.dumps(data))
    if "github" not in data:
        data["github"] = {}
        for y in range(2009, END_YEAR + 1):
            end = f"{y}-09-30" if y == END_YEAR else f"{y}-12-31"
            # whole-year counts exceed GitHub's search cap, so sample one day mid-year
            q = f"created:{y}-06-15..{y}-06-15"
            r = S.get("https://api.github.com/search/repositories", params={"q": q, "per_page": 1}, timeout=60)
            while r.status_code != 200:
                time.sleep(65)
                r = S.get("https://api.github.com/search/repositories", params={"q": q, "per_page": 1}, timeout=60)
            data["github"][str(y)] = r.json()["total_count"]
            time.sleep(6.5)
        f.write_text(json.dumps(data))
    if "wiki" not in data:
        data["wiki"] = {}
        for lang in ["en"] + SEA_WIKIS:
            r = _get("https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate/"
                     f"{lang}.wikipedia/all-access/user/monthly/2015070100/2026093000")
            data["wiki"][lang] = {i["timestamp"][:4] + "-" + i["timestamp"][4:6]: i["views"] for i in r.json()["items"]}
        f.write_text(json.dumps(data))
    return data


FETCHERS = {"hn": fetch_hn, "gtrends": fetch_gtrends, "wiki": fetch_wiki,
            "github": fetch_github, "openalex": fetch_openalex}


def fetch_all(source, sectors=None, workers=1):
    sectors = sectors or list(SECTORS)
    fn = FETCHERS[source]

    def run(s):
        t = time.time()
        fn(s)
        print(f"[{source}] {s} done in {time.time()-t:.0f}s", flush=True)

    if workers == 1:
        for s in sectors:
            run(s)
    else:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(run, sectors))


if __name__ == "__main__":
    src = sys.argv[1]
    if src == "yc":
        fetch_yc()
    elif src == "worldbank":
        fetch_worldbank()
    elif src == "denominators":
        fetch_denominators()
    else:
        fetch_all(src, workers=int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    print(f"[{src}] ALL DONE", flush=True)
