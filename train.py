"""Train two hazardous-asteroid classifiers on asteroids.csv.

Model A: random forest on close-approach and orbit-shape features only
         (MOID, absolute magnitude H and diameter are excluded on purpose).
Model B: depth-2 decision tree on MOID and H only (checks NASA's PHA rule).

Usage:  python train.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

PINK, INK, GREY = "#E8447A", "#3A1C2B", "#C9BFC4"
SEED = 42
OUT = Path("outputs")

FEATURES_A = ["closest_miss_au", "fastest_velocity_km_s", "eccentricity", "inclination_deg",
              "semi_major_axis_au", "perihelion_distance_au", "aphelion_distance_au",
              "orbital_period_days"]
FEATURES_B = ["moid_au", "absolute_magnitude_h"]
LEAKY = {"moid_au", "absolute_magnitude_h", "diameter_min_km", "diameter_max_km"}


def report(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    print(f"\n{name}")
    print(f"  hazardous precision {p:.3f}   recall {r:.3f}")
    print("  confusion matrix (rows = true, cols = predicted)")
    print("                 safe  hazardous")
    print(f"    safe       {tn:6d} {fp:10d}")
    print(f"    hazardous  {fn:6d} {tp:10d}")
    return {"precision": round(p, 4), "recall": round(r, 4),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def tree_rules(tree, names):
    t = tree.tree_
    rules = []
    for node in range(t.node_count):
        if t.children_left[node] != t.children_right[node]:
            rules.append({"node": node, "feature": names[t.feature[node]],
                          "threshold": round(float(t.threshold[node]), 4)})
    return rules


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv("asteroids.csv")
    y = df["is_hazardous"].astype(bool)
    print(f"Asteroids: {len(df)}")
    print("Class balance:")
    print(f"  hazardous     {y.sum():5d} ({y.mean():.1%})")
    print(f"  not hazardous {(~y).sum():5d} ({1 - y.mean():.1%})")

    assert not (set(FEATURES_A) & LEAKY), "a leaky feature slipped into Model A"
    idx_tr, idx_te = train_test_split(df.index, test_size=0.2, stratify=y, random_state=SEED)
    print(f"\nStratified split: {len(idx_tr)} train / {len(idx_te)} test "
          f"({y[idx_te].sum()} hazardous in test)")

    # Model A: the real challenge.
    print(f"Model A features: {FEATURES_A}")
    model_a = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=400, class_weight="balanced", min_samples_leaf=2,
                               random_state=SEED, n_jobs=-1))
    model_a.fit(df.loc[idx_tr, FEATURES_A], y[idx_tr])
    res_a = report("Model A - random forest (no MOID / H / diameter)",
                   y[idx_te], model_a.predict(df.loc[idx_te, FEATURES_A]))

    # Model B: the rule check.
    print(f"\nModel B features: {FEATURES_B}")
    b_tr = df.loc[idx_tr, FEATURES_B].notna().all(axis=1)
    b_te = df.loc[idx_te, FEATURES_B].notna().all(axis=1)
    tr_b, te_b = idx_tr[b_tr.to_numpy()], idx_te[b_te.to_numpy()]
    model_b = DecisionTreeClassifier(max_depth=2, random_state=SEED)
    model_b.fit(df.loc[tr_b, FEATURES_B], y[tr_b])
    res_b = report("Model B - decision tree, depth 2 (MOID + H)",
                   y[te_b], model_b.predict(df.loc[te_b, FEATURES_B]))
    rules = tree_rules(model_b, FEATURES_B)
    print("  split thresholds:")
    for r in rules:
        print(f"    node {r['node']}: {r['feature']} <= {r['threshold']}")

    # NASA's published rule applied directly, as a reference.
    rule = (df["moid_au"] <= 0.05) & (df["absolute_magnitude_h"] <= 22.0)
    agree = float((rule == y).mean())
    print(f"\nNASA rule (MOID <= 0.05 au and H <= 22.0) agrees with the label on {agree:.2%} of rows")

    # Feature importance for Model A.
    rf = model_a[-1]
    imp = pd.Series(rf.feature_importances_, index=FEATURES_A).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(imp.index, imp.values, color=PINK)
    ax.set(xlabel="Mean decrease in impurity", title="Model A feature importance")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=150)
    plt.close(fig)

    # MOID vs H scatter.
    fig, ax = plt.subplots(figsize=(8, 5.5))
    safe, haz = df[~y], df[y]
    ax.scatter(safe["moid_au"], safe["absolute_magnitude_h"], s=6, color=GREY, alpha=0.6,
               label=f"Not hazardous ({len(safe)})")
    ax.scatter(haz["moid_au"], haz["absolute_magnitude_h"], s=9, color=PINK, alpha=0.85,
               label=f"Potentially hazardous ({len(haz)})")
    ax.axvline(0.05, color=INK, ls="--", lw=1)
    ax.axhline(22.0, color=INK, ls="--", lw=1)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlim(left=0)
    ax.invert_yaxis()  # brighter (bigger) objects at the top
    ax.set(xlabel="MOID (au, symlog scale)", ylabel="Absolute magnitude H (lower = bigger)",
           title="NASA PHA rule: MOID <= 0.05 au and H <= 22.0")
    ax.legend(frameon=False, labelcolor=INK, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "moid_vs_h.png", dpi=150)
    plt.close(fig)

    metrics = {"n_asteroids": int(len(df)), "n_hazardous": int(y.sum()),
               "hazardous_share": round(float(y.mean()), 4),
               "model_a": {"features": FEATURES_A, **res_a},
               "model_b": {"features": FEATURES_B, "rules": rules, **res_b},
               "nasa_rule_agreement": round(agree, 4), "seed": SEED}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved charts and metrics to {OUT}/")


if __name__ == "__main__":
    main()
