import argparse, sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--target", required=True, help="YYYY-MM-DD")
    ap.add_argument("--input_dir", default="data/input")
    ap.add_argument("--output_dir", default="output")
    args = ap.parse_args()

    T = datetime.strptime(args.target, "%Y-%m-%d")
    window_days = [T - timedelta(days=d) for d in range(30, -1, -1)]
    window_str = [d.strftime("%Y-%m-%d") for d in window_days]

    ontology = {
        "Delivery issue": ["late delivery","delayed order","delivery stuck","no show","wrong route","late","delay"],
        "Food stale": ["stale","cold food","spoiled","rotten","bad smell","soggy","stinky"],
        "Delivery partner rude": ["rude","impolite","abusive","misbehaved","behaved badly"],
        "Maps not working properly": ["map not working","location wrong","gps issue","pin incorrect","map issue","gps"],
        "Instamart should be open all night": ["open all night","24x7 instamart","night delivery instamart","instamart 24x7"],
        "Bring back 10 minute bolt delivery": ["10 minute delivery","bolt delivery","instant delivery feature","10 min"],
        "Coupon not applied": ["coupon not working","promo code failed","discount not applied","offer not applied","coupon failed"],
        "Refund taking too long": ["refund pending","refund delay","money not refunded","refund not received"],
        "App keeps crashing": ["app crash","keeps crashing","app closes","force close","hangs"],
    }

    def extract_topic(text):
        t = str(text).lower()
        for topic, kws in ontology.items():
            for k in kws:
                if k in t:
                    return topic
        if "late" in t or "delay" in t:
            return "Delivery issue"
        if "stale" in t or "cold" in t or "rotten" in t:
            return "Food stale"
        if "rude" in t or "abusive" in t:
            return "Delivery partner rude"
        if "gps" in t or "map" in t or "location" in t:
            return "Maps not working properly"
        if "coupon" in t or "promo" in t or "discount" in t or "offer" in t:
            return "Coupon not applied"
        if "refund" in t:
            return "Refund taking too long"
        if "crash" in t or "hang" in t or "force close" in t or "close" in t:
            return "App keeps crashing"
        return "Other feedback"

    topic_counts = {t:{d:0 for d in window_str} for t in list(ontology.keys())+["Other feedback"]}

    input_dir = Path(args.input_dir)
    for d in window_str:
        f = input_dir/f"{d}.csv"
        if not f.exists(): 
            continue
        df = pd.read_csv(f)
        df["topic"] = df["reviewText"].map(extract_topic)
        counts = df.groupby("topic")["reviewId"].count().to_dict()
        for topic, c in counts.items():
            if topic not in topic_counts:
                topic_counts[topic] = {dd:0 for dd in window_str}
            topic_counts[topic][d] = int(c)

    import matplotlib.pyplot as plt
    import numpy as np

    trend_df = pd.DataFrame(topic_counts).T[window_str]
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir/f"trend_{args.package}_{T.strftime('%Y-%m-%d')}.csv"
    png_path = out_dir/f"trend_{args.package}_{T.strftime('%Y-%m-%d')}.png"

    trend_df.to_csv(csv_path)

    plt.figure(figsize=(12,6))
    plt.imshow(trend_df.values, aspect='auto')
    plt.xticks(ticks=np.arange(len(trend_df.columns)), labels=[d[5:] for d in trend_df.columns], rotation=90)
    plt.yticks(ticks=np.arange(len(trend_df.index)), labels=trend_df.index)
    plt.title(f"31-day Topic Trend for {args.package} (T={T.strftime('%Y-%m-%d')})")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    print(str(csv_path))

if __name__ == "__main__":
    main()