"""
diagnose_fetch.py – standalone connectivity/download check for the NSE/BSE
fetcher. Run this directly (not through the pipeline) to see exactly which
step is failing.

Usage:
    python diagnose_fetch.py --ticker TCS --company "Tata Consultancy Services"
"""
import argparse
import sys
from datetime import datetime, timedelta

p = argparse.ArgumentParser()
p.add_argument("--ticker", required=True)
p.add_argument("--company", required=True)
p.add_argument("--scripcode", default=None,
               help="BSE scrip code (e.g. 532540 for TCS). Bypasses the fuzzy "
                    "getScripCode() name lookup, matching what the pipeline does.")
args = p.parse_args()

print("=" * 70)
print("STEP 1 — NSE cookie handshake")
print("=" * 70)
try:
    from nse import NSE
    nse = NSE(download_folder="./data/nse_cache")
    print("✅ NSE client initialised (cookies obtained)")
except Exception as e:
    nse = None
    print(f"❌ NSE init failed: {e!r}")

print()
print("=" * 70)
print("STEP 2 — NSE announcements()")
print("=" * 70)
if nse:
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=180)
        anns = nse.announcements(symbol=args.ticker, from_date=start, to_date=end)
        print(f"✅ Got {len(anns)} announcements")
        for a in anns[:3]:
            print(f"   - {a.get('an_dt')}  {a.get('desc')}  {a.get('attchmntFile')}")
    except Exception as e:
        anns = []
        print(f"❌ NSE announcements() failed: {e!r}")
else:
    anns = []
    print("skipped (step 1 failed)")

print()
print("=" * 70)
print("STEP 3 — BSE scrip code lookup")
print("=" * 70)
scripcode = None
try:
    from bse import BSE
    bse = BSE(download_folder="./data/bse_cache")
    if args.scripcode:
        scripcode = args.scripcode
        print(f"✅ Using scrip code (provided): {scripcode}")
    else:
        scripcode = bse.getScripCode(args.company)
        print(f"✅ Resolved scrip code (fuzzy lookup): {scripcode}")
except Exception as e:
    print(f"❌ BSE scrip code lookup failed: {e!r}")

print()
print("=" * 70)
print("STEP 4 — BSE announcements()")
print("=" * 70)
bse_items = []
if scripcode:
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=180)
        data = bse.announcements(page_no=1, from_date=start, to_date=end, scripcode=str(scripcode))
        bse_items = data.get("Table") or []
        print(f"✅ Got {len(bse_items)} announcements")
        for r in bse_items[:3]:
            print(f"   - {r.get('NEWS_DT')}  {r.get('HEADLINE')}  {r.get('ATTACHMENTNAME')}")
    except Exception as e:
        print(f"❌ BSE announcements() failed: {e!r}")
else:
    print("skipped (step 3 failed)")

print()
print("=" * 70)
print("STEP 5 — PDF download")
print("=" * 70)
import httpx

test_urls = []
if anns:
    u = anns[0].get("attchmntFile")
    if u:
        test_urls.append(("NSE", u))
if bse_items:
    name = bse_items[0].get("ATTACHMENTNAME")
    if name:
        test_urls.append(("BSE", f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{name}"))
        test_urls.append(("BSE-hist", f"https://www.bseindia.com/xml-data/corpfiling/AttachHis/{name}"))

if not test_urls:
    print("No attachment URLs available to test (steps 2/4 returned nothing).")
else:
    for src, url in test_urls:
        try:
            with httpx.Client(follow_redirects=True, timeout=25) as client:
                resp = client.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                    "Referer": "https://www.bseindia.com/" if src.startswith("BSE") else "https://www.nseindia.com/",
                    "Accept": "application/pdf,*/*",
                })
            ok = resp.status_code == 200 and resp.content[:4] == b"%PDF"
            print(f"{'✅' if ok else '❌'} [{src}] {url}")
            print(f"    status={resp.status_code} bytes={len(resp.content)} content_type={resp.headers.get('content-type')}")
            if not ok:
                print(f"    first 200 bytes: {resp.content[:200]!r}")
        except Exception as e:
            print(f"❌ [{src}] {url} -> {e!r}")

print()
print("Done.")
