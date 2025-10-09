# scripts/build_manifest_msrvtt.py
import csv, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset

def try_probe(url: str) -> tuple[str, str]:
    # Return (url, "ok"|"fail")
    try:
        # lightweight probe: use yt_dlp to extract without downloading
        import yt_dlp
        ydl_opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist"}
        cookie_file = os.environ.get("YTDLP_COOKIE_FILE")
        if cookie_file and os.path.exists(cookie_file):
            ydl_opts["cookiefile"] = cookie_file

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=False)
        return url, "ok"
    except Exception:
        return url, "fail"

def main(split="train", out_csv="data/msrvtt_manifest.csv"):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    ds = load_dataset("AlexZigma/msr-vtt", split=split)
    rows = []
    urls = [r["url"] for r in ds]

    # parallel but not too aggressive
    max_workers = 8
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(try_probe, u): u for u in urls}
        for i, fut in enumerate(as_completed(futs), 1):
            url, status = fut.result()
            rows.append((url, status))
            if i % 100 == 0:
                print(f"Probed {i}/{len(urls)}")

    # write csv with url + status; later we join back to the HF records
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["url", "status"])
        w.writerows(rows)
    print(f"Wrote {out_csv} with {len(rows)} rows")

if __name__ == "__main__":
    # usage: python scripts/build_manifest_msrvtt.py [train|val] [out_csv]
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    out = sys.argv[2] if len(sys.argv) > 2 else ("data/msrvtt_manifest_train.csv" if split=="train" else "data/msrvtt_manifest_val.csv")
    main(split, out)
