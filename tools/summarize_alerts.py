import argparse, csv, json
from collections import defaultdict

def summarize(csv_path, out_json, gap_s=0.5):
    rows=[]
    with open(csv_path) as f:
        r=csv.DictReader(f)
        for row in r:
            try:
                rows.append({
                    "ts": float(row["ts"]),
                    "tid": int(row["track_id"]),
                    "frames": int(row["frames"]),
                    "p": float(row["shoplift_prob"]),
                    "fired": int(row["fired"]),
                    "src": row.get("source","")
                })
            except Exception:
                pass

    if not rows:
        print("No rows in", csv_path)
        return 0

    by_src = defaultdict(list)
    for r in rows:
        by_src[r["src"]].append(r)

    events=[]
    for src, group in by_src.items():
        group.sort(key=lambda x: (x["tid"], x["ts"]))
        t0 = group[0]["ts"]
        cur=None
        for r in group:
            if r["fired"]==1:
                if (not cur) or (cur["tid"]!=r["tid"]) or ((r["ts"]-cur["end_ts"])>gap_s):
                    cur={"src":src, "tid":r["tid"], "start_ts":r["ts"], "end_ts":r["ts"], "max_p":r["p"], "t0":t0}
                    events.append(cur)
                else:
                    cur["end_ts"]=r["ts"]
                    if r["p"]>cur["max_p"]: cur["max_p"]=r["p"]

    for e in events:
        t0=e.pop("t0")
        e["start_offset_s"]=round(e["start_ts"]-t0,2)
        e["end_offset_s"]=round(e["end_ts"]-t0,2)
        e["duration_s"]=round(e["end_ts"]-e["start_ts"],2)

    with open(out_json,"w") as f:
        json.dump({"events":events}, f, indent=2)

    print(f"Wrote {out_json} with {len(events)} events")
    for e in events:
        print(f"{e['src']} | Track {e['tid']}: {e['start_offset_s']}s → {e['end_offset_s']}s "
              f"(dur {e['duration_s']}s, max p={e['max_p']:.3f})")
    return len(events)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="output/alert_log.csv")
    ap.add_argument("--out", default="output/alert_events.json")
    ap.add_argument("--gap", type=float, default=0.5)
    args = ap.parse_args()
    summarize(args.csv, args.out, gap_s=args.gap)
