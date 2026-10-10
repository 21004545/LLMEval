"""Make the report figures from the results files.

Run: python charts.py results_china30.tsv [results_us30.tsv]   ->  charts/*.png
"""

import csv
import os
import sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

csv.field_size_limit(10**9)
MODELS = [("openai-gpt-nano", "ChatGPT"), ("gemini-3.1-flash-lite", "Gemini"),
          ("deepseek-v4-flash", "DeepSeek"), ("kimi-k2", "Kimi")]
LANGS = [("English", "English prompt", "#4a5a6a"), ("Mandarin", "Mandarin prompt", "#b4543a")]
BIAS = ["factual_balance", "framing_bias", "attribution_of_responsibility", "loaded_language"]
SOURCE = "Source: project data, {n} responses ({sets}). Restricted = blocked by the provider, or judge-scored avoidance of 4-5 out of 5."

plt.rcParams.update({
    "font.family": ["Segoe UI", "Arial", "sans-serif"], "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888888",
    "axes.labelcolor": "#333333", "xtick.color": "#444444", "ytick.color": "#333333",
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def restricted(r):
    return r["status"] == "blocked" or (r["status"] == "success" and int(r["political_avoidance"]) >= 4)


def load(paths):
    rows = []
    for p in paths:
        rows += [r for r in csv.DictReader(open(p, encoding="utf-8"), delimiter="\t") if r["status"] in ("success", "blocked")]
    return rows


def title(ax, text, sub=None):
    ax.set_title(text, loc="left", fontsize=12, fontweight="bold", color="#222222", pad=22 if sub else 10)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9.5, color="#555555")


def note(fig, text):
    fig.text(0.01, -0.02, text, fontsize=7.5, color="#777777", ha="left", va="top", wrap=True)


def rate_bars(rows, path, heading, sub, sets):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    h = 0.36
    for i, (key, name) in enumerate(MODELS):
        for j, (lang, label, colour) in enumerate(LANGS):
            g = [r for r in rows if r["responder"] == key and r["language"] == lang]
            v = 100 * sum(map(restricted, g)) / len(g) if g else 0
            y = i + (j - 0.5) * h
            ax.barh(y, v, height=h * 0.9, color=colour)
            ax.text(v + 0.6, y, f"{v:.0f}%", va="center", fontsize=8.5, color="#333333")
    ax.set_yticks(range(len(MODELS)), [m[1] for m in MODELS])
    ax.invert_yaxis()
    ax.set_xlabel("Share of responses restricted (%)")
    ax.xaxis.grid(True, color="#e5e5e5")
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(35, ax.get_xlim()[1]))
    # Direct labels in place of a legend box (top right is empty: the US models restrict little)
    ax.text(0.99, 0.93, "English prompt", color=LANGS[0][2], transform=ax.transAxes, ha="right", fontsize=9, fontweight="bold")
    ax.text(0.99, 0.86, "Mandarin prompt", color=LANGS[1][2], transform=ax.transAxes, ha="right", fontsize=9, fontweight="bold")
    title(ax, heading, sub)
    note(fig, SOURCE.format(n=len(rows), sets=sets))
    fig.savefig(path)
    plt.close(fig)


def event_grid(rows, path):
    events, names = defaultdict(lambda: defaultdict(int)), {}
    for r in rows:
        names[r["Event ID"]] = r["Event"]
        if restricted(r):
            events[r["Event ID"]][(r["responder"], r["language"])] += 1
    ids = sorted(events, key=lambda e: -sum(events[e].values()))
    cols = [(m, l) for m, _ in MODELS for l, _, _ in LANGS]
    data = [[events[e].get(c, 0) for c in cols] for e in ids]
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(ids) + 1.2))
    cmap = LinearSegmentedColormap.from_list("w2r", ["#ffffff", "#f1d3c9", "#b4543a"])
    ax.imshow(data, cmap=cmap, vmin=0, vmax=5, aspect="auto")
    for y, row in enumerate(data):
        for x, v in enumerate(row):
            if v:
                ax.text(x, y, v, ha="center", va="center", fontsize=8.5, color="white" if v >= 4 else "#333333")
    ax.set_xticks(range(len(cols)), [("English" if l == "English" else "Mandarin") for _, l in cols], fontsize=8)
    for i, (_, name) in enumerate(MODELS):
        ax.text(2 * i + 0.5, len(ids) + 0.35, name, ha="center", va="top", fontsize=9.5, fontweight="bold", color="#333333")
    labels = [names[e].removeprefix("The ") for e in ids]
    ax.set_yticks(range(len(ids)), [l[0].upper() + l[1:] for l in labels], fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for x in (1.5, 3.5, 5.5):
        ax.axvline(x, color="#999999", lw=0.8)
    ax.set_xticks([x + 0.5 for x in range(len(cols) - 1)], minor=True)
    ax.set_yticks([y + 0.5 for y in range(len(ids) - 1)], minor=True)
    ax.grid(which="minor", color="#eeeeee", lw=0.6)
    ax.tick_params(which="both", length=0)
    title(ax, "Which China-related events get restricted",
          "How many of the 5 question framings were restricted. Events with none are left out.")
    note(fig, SOURCE.format(n=len(rows), sets="30 China-related events"))
    fig.savefig(path)
    plt.close(fig)


def bias_dumbbell(rows, path, sets):
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    for i, (key, name) in enumerate(MODELS):
        means = []
        for lang, _, colour in LANGS:
            g = [r for r in rows if r["responder"] == key and r["language"] == lang and r["status"] == "success"]
            means.append(sum(int(r[s]) for r in g for s in BIAS) / (len(BIAS) * len(g)))
        for m, (lang, _, colour), side in zip(means, LANGS, (-1, 1)):
            ax.plot(m, i, "o", color=colour, ms=8, zorder=3)
            ax.text(m + side * 0.09, i, f"{m:.2f}", ha="right" if side < 0 else "left", va="center",
                    fontsize=8, color=colour)
        ax.plot(means, [i, i], color="#bbbbbb", lw=2, zorder=2)
    ax.set_yticks(range(len(MODELS)), [m[1] for m in MODELS])
    ax.invert_yaxis()
    ax.set_xlim(1, 5)
    ax.set_ylim(len(MODELS) - 0.4, -0.6)
    ax.set_xlabel("Mean bias score of answered questions (1 = least, 5 = most)")
    ax.xaxis.grid(True, color="#e5e5e5")
    ax.set_axisbelow(True)
    ax.text(0.99, 0.12, "English prompt", color=LANGS[0][2], transform=ax.transAxes, ha="right", fontsize=9, fontweight="bold")
    ax.text(0.99, 0.03, "Mandarin prompt", color=LANGS[1][2], transform=ax.transAxes, ha="right", fontsize=9, fontweight="bold")
    title(ax, "Mandarin answers from the Chinese models score as more biased",
          "Average of factual balance, framing, attribution and loaded language, as scored by the AI judge.")
    fig.text(0.01, -0.1, f"Source: project data ({sets}). Blocked responses have no score and are excluded.", fontsize=7.5, color="#777777")
    fig.savefig(path)
    plt.close(fig)


def framing_bars(rows, path):
    framings = [("tell_me", "What can you tell me about...?"), ("led_up_to", "What led up to...?"),
                ("opinion", "What do you think about...?"), ("politics", "Describe the politics of..."),
                ("viewed_today", "How is ... viewed today?")]
    cn = [r for r in rows if r["responder"] in ("deepseek-v4-flash", "kimi-k2")]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    h = 0.36
    for i, (key, label) in enumerate(framings):
        for j, (lang, _, colour) in enumerate(LANGS):
            g = [r for r in cn if r["Framing"] == key and r["language"] == lang]
            v = 100 * sum(map(restricted, g)) / len(g)
            y = i + (j - 0.5) * h
            ax.barh(y, v, height=h * 0.9, color=colour)
            ax.text(v + 0.5, y, f"{v:.0f}%", va="center", fontsize=8.5, color="#333333")
    ax.set_yticks(range(len(framings)), [f[1] for f in framings])
    ax.invert_yaxis()
    ax.set_xlabel("Share of responses restricted (%)")
    ax.xaxis.grid(True, color="#e5e5e5")
    ax.set_axisbelow(True)
    ax.set_xlim(0, 35)
    ax.text(0.99, 0.08, "English prompt", color=LANGS[0][2], transform=ax.transAxes, ha="right", fontsize=9, fontweight="bold")
    ax.text(0.99, 0.01, "Mandarin prompt", color=LANGS[1][2], transform=ax.transAxes, ha="right", fontsize=9, fontweight="bold")
    title(ax, "Language matters more than question wording",
          "Restriction rate for the Chinese models (DeepSeek and Kimi) by question framing.")
    note(fig, f"Source: project data, {len(cn)} DeepSeek and Kimi responses (30 China-related events). Restricted = blocked, or avoidance scored 4-5.")
    fig.savefig(path)
    plt.close(fig)


def main(china, us=None):
    os.makedirs("charts", exist_ok=True)
    c = load([china])
    rate_bars(c, "charts/1_restriction_by_model_language.png",
              "Chinese models restrict more when asked in Mandarin",
              "Share of answers about 30 China-related events that were blocked or evaded.", "30 China-related events")
    event_grid(c, "charts/2_restriction_by_event.png")
    bias_dumbbell(c, "charts/3_bias_by_model_language.png", "30 China-related events")
    framing_bars(c, "charts/4_restriction_by_framing.png")
    if us:
        u = load([us])
        rate_bars(u, "charts/5_restriction_us_events.png",
                  "Kimi also deflects US topics when asked in Mandarin",
                  "Share of answers about 30 US-related events that were blocked or evaded.", "30 US-related events")
    print("charts written to charts/")


if __name__ == "__main__":
    main(*sys.argv[1:3])
