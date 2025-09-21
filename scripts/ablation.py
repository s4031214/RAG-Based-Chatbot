import os, json, itertools, subprocess, time, argparse, pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
EVAL = os.path.join(ROOT, "scripts", "07_evaluate.py")
RUNS = os.path.join(ROOT, "data", "eval", "runs")
os.makedirs(RUNS, exist_ok=True)

def run_cfg(models, gold, outprefix, top_k, context_k, judge_model=None):
    args = [
        "python", EVAL,
        "--models", *models,
        "--gold", gold,
        "--outprefix", outprefix,
        "--top_k", str(top_k),
        "--context_k", str(context_k)
    ]
    if judge_model:
        args += ["--judge_model", judge_model]
    print(">>", " ".join(args))
    subprocess.run(args, check=True)

def find_latest_summary(prefix: str):
    # find newest dir that starts with prefix under data/eval/runs
    base = os.path.join(RUNS)
    candidates = []
    for d in os.listdir(base):
        if d.startswith(prefix):
            path = os.path.join(base, d, "summary_by_model.csv")
            if os.path.exists(path):
                ts = os.path.getmtime(path)
                candidates.append((ts, path))
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--gold", default=os.path.join(ROOT,"data","eval","questions.jsonl"))
    ap.add_argument("--judge_model", default=None)
    ap.add_argument("--k_list", nargs="+", type=int, default=[4,8,12])
    ap.add_argument("--c_list", nargs="+", type=int, default=[2,4,6])
    ap.add_argument("--tag", default="grid")
    args = ap.parse_args()

    summaries = []
    for k, c in itertools.product(args.k_list, args.c_list):
        outp = f"abl_{args.tag}_k{k}_c{c}"
        run_cfg(args.models, args.gold, outp, k, c, judge_model=args.judge_model)
        # collect latest summary file
        path = find_latest_summary(outp)
        if not path:
            continue
        df = pd.read_csv(path)
        df["top_k"] = k
        df["context_k"] = c
        summaries.append(df)

    if not summaries:
        print("No summaries found.")
        return
    big = pd.concat(summaries, ignore_index=True)
    out_csv = os.path.join(RUNS, f"ablations_{args.tag}.csv")
    big.to_csv(out_csv, index=False)
    print("✅ Wrote ablation summary:", out_csv)
    print(big.groupby(["model","top_k","context_k"])[["UA%","Acc@Gold%","Recall@8%","Faithfulness(mean)","Completeness(mean)","Latency_p95_s"]].mean().reset_index().to_string(index=False))

if __name__ == "__main__":
    main()
