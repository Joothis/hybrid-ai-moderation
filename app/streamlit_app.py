"""Hybrid AI Moderation - demo app.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moderation.domains import DOMAINS, ROOT, load_costing, load_domain  # noqa: E402
from moderation.llm import OllamaLLM  # noqa: E402
from moderation.ml_model import MLModel  # noqa: E402
from moderation.pipeline import HybridPipeline, LLMOnlyPipeline, MLOnlyPipeline  # noqa: E402
from moderation.rules import RuleEngine  # noqa: E402

PIPE_NAMES = {
    "ml_only": "Traditional ML only",
    "llm_only": "LLM only",
    "rules_ml": "Rules + ML (ablation)",
    "hybrid": "Hybrid: Rules → ML → LLM",
}
PIPE_COLORS = {"ml_only": "#8C8C8C", "llm_only": "#E4572E", "rules_ml": "#A0C4E8", "hybrid": "#2E86AB"}
LAYER_COLORS = {"rules": "#6C757D", "ml": "#2E86AB", "llm": "#E4572E"}

st.set_page_config(page_title="Hybrid AI Moderation", page_icon="🛡️", layout="wide")


# --------------------------------------------------------------------------- cached resources
@st.cache_resource
def components(domain: str):
    cfg = load_domain(domain)
    return cfg, RuleEngine.from_config(cfg["rules"]), MLModel.load(domain)


@st.cache_resource
def llm_client() -> OllamaLLM:
    return OllamaLLM()


@st.cache_data(ttl=30)
def llm_up() -> bool:
    return llm_client().available()


def load_results(domain: str):
    """Benchmark output, or None if missing or written by an older version (no margin-based trade-off)."""
    path = ROOT / "results" / f"{domain}.json"
    if not path.exists():
        return None
    res = json.loads(path.read_text())
    if "margin" not in res or not res.get("tradeoff") or "margin" not in res["tradeoff"][0]:
        return None
    return res


def load_items(domain: str) -> pd.DataFrame | None:
    path = ROOT / "results" / f"{domain}_items.parquet"
    return pd.read_parquet(path) if path.exists() else None


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🛡️ Hybrid Moderation")
    domain = st.selectbox("Use case", DOMAINS, format_func=lambda d: load_domain(d)["display_name"])
    cfg, rules, ml = components(domain)
    st.caption(cfg["description"])
    st.divider()
    st.subheader("Routing")
    margin = st.slider(
        "Uncertainty band (log-odds)", 0.0, 4.0, float(round(ml.margin, 2)), 0.05,
        help="Comments whose ML score is within ± this distance of the decision threshold are escalated to the LLM. "
             f"Default {ml.margin:.2f} = calibrated to a {cfg['routing']['escalation_budget']:.0%} escalation budget.",
    )
    floors = cfg["routing"].get("lo_by_label") or {}
    if floors:
        st.caption("Severity floors (always get an LLM second opinion): " + ", ".join(f"`{k}` > {v}" for k, v in floors.items()))
    st.divider()
    st.subheader("LLM")
    if llm_up():
        st.success(f"Ollama · `{llm_client().model}`", icon="✅")
    else:
        st.error(f"Ollama `{llm_client().model}` not reachable - escalations fall back to the ML decision.", icon="⚠️")

hybrid = HybridPipeline(rules, ml, llm_client(), cfg, margin)

st.title("Hybrid AI Content Moderation")
st.markdown(
    "Send each comment to the **cheapest layer that can decide it confidently**: "
    "deterministic **rules** → **Logistic Regression** → an **LLM** only for the ambiguous rest."
)

tab_live, tab_batch, tab_bench, tab_trade, tab_arch = st.tabs(
    ["🔎 Live moderator", "📦 Batch simulation", "📊 Benchmark", "⚖️ Trade-off", "🏗️ Architecture"]
)


# --------------------------------------------------------------------------- live moderator
def verdict_badge(labels: list[str]) -> str:
    return f":red[**REMOVE**] · {', '.join(labels)}" if labels else ":green[**ALLOW**]"


with tab_live:
    examples = [{"text": "", "note": "(type your own)"}] + cfg.get("demo_examples", [])
    pick = st.selectbox(
        "Demo examples", range(len(examples)),
        format_func=lambda i: examples[i]["note"] if i == 0 else f"[{examples[i].get('expect', '?').upper()}] {examples[i]['text'][:90]}",
    )
    text = st.text_area("Comment", value=examples[pick]["text"], height=100, key=f"text_{domain}_{pick}")
    c1, c2 = st.columns([1, 3])
    go_btn = c1.button("Moderate", type="primary", use_container_width=True, disabled=not text.strip())
    compare = c2.checkbox("Also show what ML-only and LLM-only would decide", value=True)

    if go_btn and text.strip():
        with st.spinner("Moderating..."):
            res = hybrid.moderate(text)
        hit = rules.check(text)
        probs = ml.proba_dict(text)
        route = hybrid.route(probs) if hit is None else None

        st.markdown(f"### Verdict: {verdict_badge(res.labels)}")
        st.caption(f"Decided by **{res.layer.upper()}** · {res.latency_ms:,.0f} ms · "
                   f"{'1 LLM call' if res.llm_called else 'no LLM call'} · {res.reason}")

        l1, l2, l3 = st.columns(3)
        with l1.container(border=True):
            st.markdown("#### ① Rules")
            if hit:
                st.markdown(f"**{'BLOCK' if hit.action == 'block' else 'ALLOW'}** - `{', '.join(hit.rule_names)}`")
                st.caption(hit.reason)
            else:
                st.markdown("No rule matched → pass to ML")
        with l2.container(border=True):
            st.markdown("#### ② Logistic Regression")
            if hit:
                st.caption("Skipped (decided by rules)")
            else:
                dfp = pd.DataFrame({"label": list(probs), "p": list(probs.values()),
                                    "threshold": [ml.thresholds[l] for l in probs]})
                fig = px.bar(dfp, x="p", y="label", orientation="h", range_x=[0, 1], height=200, text_auto=".2f")
                fig.add_scatter(x=dfp["threshold"], y=dfp["label"], mode="markers", marker_symbol="line-ns-open",
                                marker_size=18, marker_color="black", name="threshold")
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False, xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown({"flag": "**Confident violation** → done", "clean": "**Confident clean** → done",
                             "escalate": "**Unsure** → escalate to LLM"}[route])
        with l3.container(border=True):
            st.markdown("#### ③ LLM")
            if res.layer == "llm":
                st.markdown(verdict_badge(res.labels))
                st.info(res.reason)
                st.caption(f"{res.prompt_tokens} prompt + {res.completion_tokens} completion tokens")
            elif res.llm_error:
                st.warning(res.reason)
            else:
                st.caption("Not called - saved one LLM call")

        if compare:
            st.markdown("##### Same comment, other strategies")
            cmp = [("Traditional ML only", MLOnlyPipeline(ml).moderate(text))]
            if llm_up():
                with st.spinner("Asking the LLM (LLM-only)..."):
                    cmp.append(("LLM only", LLMOnlyPipeline(llm_client(), cfg).moderate(text)))
            cmp.append(("Hybrid", res))
            cols = st.columns(len(cmp))
            for col, (name, r) in zip(cols, cmp):
                col.metric(name, r.action, f"{r.latency_ms:,.0f} ms", delta_color="off")
                col.caption(", ".join(r.labels) or "no labels")


# --------------------------------------------------------------------------- batch simulation
with tab_batch:
    test = pd.read_parquet(ROOT / "data" / "processed" / domain / "test.parquet")
    bench = load_results(domain)
    # Same seed as the benchmark so its cached LLM answers are reused; never fewer than 300 comments.
    sample_n = max(bench["sample"] if bench else 0, 300)
    pool = test.sample(n=min(sample_n, len(test)), random_state=bench["seed"] if bench else 0).reset_index(drop=True)
    st.markdown(f"Simulate a slice of the **{load_costing()['daily_volume']:,} comments/day** stream through the hybrid "
                f"pipeline. Comments come from the held-out test set (benchmark sample → LLM answers are cached).")
    lo_n = min(20, len(pool) - 1)
    n = st.slider("Comments to process", lo_n, len(pool), min(200, len(pool)), 10) if len(pool) > 1 else len(pool)
    if st.button("Run simulation", type="primary"):
        texts = pool["text"].head(n).tolist()
        bar = st.progress(0.0, text="starting")
        out = hybrid.moderate_batch(texts, lambda i, k: bar.progress(i / k, text=f"{i}/{k} comments"))
        bar.empty()
        labels = cfg["label_names"]
        df = pool.head(n).copy()
        df["truth"] = df[labels].any(axis=1)
        df["flagged"] = [r.flagged for r in out]
        df["layer"] = [r.layer for r in out]
        df["labels_pred"] = [", ".join(r.labels) for r in out]
        df["reason"] = [r.reason for r in out]
        df["latency_ms"] = [r.latency_ms for r in out]
        layer_counts = df["layer"].value_counts().reindex(["rules", "ml", "llm"]).fillna(0).astype(int)

        k1, k2, k3, k4 = st.columns(4)
        acc = (df["truth"] == df["flagged"]).mean()
        k1.metric("Accuracy vs. labels", f"{acc:.1%}")
        k2.metric("Sent to LLM", f"{layer_counts['llm']} / {n}", f"{layer_counts['llm'] / n:.0%} of traffic", delta_color="off")
        k3.metric("Removed", int(df["flagged"].sum()))
        k4.metric("Median latency", f"{df['latency_ms'].median():.0f} ms")

        f1c, f2c = st.columns([3, 2])
        remaining = n - layer_counts["rules"]
        fig = go.Figure(go.Funnel(
            y=["Incoming", "After rules", "Escalated to LLM"],
            x=[n, remaining, layer_counts["llm"]],
            textinfo="value+percent initial", marker_color=["#6C757D", "#2E86AB", "#E4572E"],
        ))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        f1c.plotly_chart(fig, use_container_width=True)
        pie = px.pie(values=layer_counts.values, names=[f"{k} ({v})" for k, v in layer_counts.items()], hole=0.5,
                     color=layer_counts.index, color_discrete_map=LAYER_COLORS, height=300, title="Decided by")
        pie.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        f2c.plotly_chart(pie, use_container_width=True)

        st.markdown("##### Escalated comments and the LLM's reasoning")
        st.dataframe(df.loc[df["layer"] == "llm", ["text", "truth", "labels_pred", "reason"]],
                     use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- benchmark
def pipeline_table(res: dict) -> pd.DataFrame:
    rows = []
    for key in ["ml_only", "llm_only", "hybrid", "rules_ml"]:
        m = res["metrics"][key]
        rows.append({
            "Pipeline": PIPE_NAMES[key], "key": key,
            "Flag F1": m["flag"]["f1"], "Precision": m["flag"]["precision"], "Recall": m["flag"]["recall"],
            "False positives": m["flag"]["fp"], "False negatives": m["flag"]["fn"],
            "Macro label F1": m["macro_label_f1"],
            "LLM calls": m["llm_calls"], "Escalation %": m["escalation_rate"] * 100,
            "p50 latency (ms)": m["latency_ms"]["p50"], "p95 latency (ms)": m["latency_ms"]["p95"],
            "LLM calls / day": m["cost"]["llm_calls_per_day"],
            "API $ / month": m["cost"]["api_usd_per_month"],
            "LLM compute h / day": m["cost"]["llm_compute_hours_per_day"],
        })
    return pd.DataFrame(rows)


with tab_bench:
    res = load_results(domain)
    if not res:
        st.info("No benchmark yet. Run `python scripts/benchmark.py --sample 300`.")
    else:
        tbl = pipeline_table(res)
        h, ml_r, llm_r = (tbl.set_index("key").loc[k] for k in ("hybrid", "ml_only", "llm_only"))
        st.caption(f"{res['sample']} held-out comments · {res['positive_rate']:.0%} violations · LLM `{res['llm_model']}` · "
                   f"generated {res['generated_at']}")
        c = st.columns(4)
        c[0].metric("Hybrid F1", f"{h['Flag F1']:.3f}", f"{h['Flag F1'] - ml_r['Flag F1']:+.3f} vs ML-only")
        c[1].metric("vs LLM-only F1", f"{llm_r['Flag F1']:.3f}", f"{h['Flag F1'] - llm_r['Flag F1']:+.3f} hybrid gap")
        c[2].metric("LLM calls saved", f"{1 - h['LLM calls'] / max(llm_r['LLM calls'], 1):.0%}",
                    f"{int(h['LLM calls'])} vs {int(llm_r['LLM calls'])}", delta_color="off")
        c[3].metric("API cost / month @10k/day", f"${h['API $ / month']:,.2f}", f"LLM-only ${llm_r['API $ / month']:,.2f}",
                    delta_color="off")

        long = tbl.melt(id_vars=["Pipeline", "key"], value_vars=["Flag F1", "LLM calls", "p50 latency (ms)", "API $ / month"])
        grid = st.columns(2)
        for i, metric in enumerate(["Flag F1", "LLM calls", "p50 latency (ms)", "API $ / month"]):
            d = long[long["variable"] == metric]
            fig = px.bar(d, x="Pipeline", y="value", color="key", color_discrete_map=PIPE_COLORS, text_auto=".3~f",
                         title=metric, height=300, log_y=metric == "p50 latency (ms)")
            fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None, margin=dict(l=0, r=0, t=40, b=0))
            grid[i % 2].plotly_chart(fig, use_container_width=True)

        fpfn = tbl.melt(id_vars=["Pipeline"], value_vars=["False positives", "False negatives"])
        fig = px.bar(fpfn, x="Pipeline", y="value", color="variable", barmode="group", text_auto=True, height=300,
                     title="Errors", color_discrete_sequence=["#F4A259", "#BC4B51"])
        fig.update_layout(xaxis_title=None, yaxis_title=None, margin=dict(l=0, r=0, t=40, b=0), legend_title=None)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(tbl.drop(columns="key"), hide_index=True, use_container_width=True,
                     column_config={"Escalation %": st.column_config.NumberColumn(format="%.1f%%"),
                                    "API $ / month": st.column_config.NumberColumn(format="$%.2f")})
        st.caption(f"Cost model: {res['metrics']['hybrid']['cost']['api_price_note']}; local compute "
                   f"${load_costing()['local_compute']['usd_per_hour']}/h. Edit `config/costing.yaml`.")

        all_res = {d: load_results(d) for d in DOMAINS}
        if all(all_res.values()):
            st.markdown("##### All use cases")
            rows = []
            for d, r in all_res.items():
                for k in ("ml_only", "llm_only", "hybrid"):
                    m = r["metrics"][k]
                    rows.append({"Use case": r["display_name"], "Pipeline": PIPE_NAMES[k], "F1": m["flag"]["f1"],
                                 "FP": m["flag"]["fp"], "FN": m["flag"]["fn"], "LLM calls": m["llm_calls"],
                                 "p50 ms": m["latency_ms"]["p50"], "API $/month": m["cost"]["api_usd_per_month"]})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------- trade-off
with tab_trade:
    res = load_results(domain)
    if not res:
        st.info("Run the benchmark first.")
    else:
        pts = pd.DataFrame(res["tradeoff"])
        pts["Escalated to LLM (%)"] = pts["escalation_rate"] * 100
        llm_f1 = res["metrics"]["llm_only"]["flag"]["f1"]
        ml_f1 = res["metrics"]["ml_only"]["flag"]["f1"]
        fig = px.line(pts, x="Escalated to LLM (%)", y="f1", markers=True, hover_data=["margin", "fp", "fn"],
                      height=450, title="Accuracy vs. LLM usage - every point is a different band width")
        fig.add_hline(y=llm_f1, line_dash="dash", line_color=PIPE_COLORS["llm_only"], annotation_text=f"LLM only ({llm_f1:.3f})")
        fig.add_hline(y=ml_f1, line_dash="dot", line_color=PIPE_COLORS["ml_only"], annotation_text=f"ML only ({ml_f1:.3f})")
        cur = pts.iloc[(pts["margin"] - margin).abs().argmin()]
        fig.add_scatter(x=[cur["Escalated to LLM (%)"]], y=[cur["f1"]], mode="markers", marker_size=16,
                        marker_color=PIPE_COLORS["hybrid"], name=f"current band ±{cur['margin']:.1f}")
        fig.update_layout(yaxis_title="Flag F1", margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)
        best = pts.loc[pts["f1"].idxmax()]
        st.markdown(
            f"- Current band ±{cur['margin']:.1f}: **F1 {cur['f1']:.3f}** with **{cur['Escalated to LLM (%)']:.0f}%** of comments sent to the LLM.\n"
            f"- Best F1 on this sample: **{best['f1']:.3f}** at {best['Escalated to LLM (%)']:.0f}% escalation (band ±{best['margin']:.1f}).\n"
            f"- Move the sidebar slider to see where the live router sits on this curve."
        )
        st.caption("Computed from the stored per-comment outputs of the benchmark (no extra LLM calls). "
                   "Treat as a sensitivity analysis: the production band is set on validation data via the escalation budget.")


# --------------------------------------------------------------------------- architecture
with tab_arch:
    st.graphviz_chart(f"""
digraph G {{
  rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", color="#444444"];
  in [label="Comment", fillcolor="#FFFFFF"];
  r [label="① Rules\\nregex · blocklists · PII\\n~0 ms, $0", fillcolor="#E9ECEF"];
  m [label="② TF-IDF + Logistic Regression\\nper-label probabilities\\n~1-3 ms, $0", fillcolor="#D6EAF8"];
  l [label="③ LLM (Ollama {llm_client().model})\\npolicy prompt + JSON\\n~seconds, tokens", fillcolor="#FADBD8"];
  d [label="Decision\\nALLOW / REMOVE + labels + reason", fillcolor="#FFFFFF"];
  in -> r; r -> d [label="clear-cut"]; r -> m [label="no match"];
  m -> d [label="confident"]; m -> l [label="unsure band\\nor severity floor"]; l -> d;
}}""")
    st.markdown("""
**Why this design**
- **Rules** catch blatant, high-precision cases (links, card numbers, obfuscated profanity) for free and are auditable.
- **Logistic Regression** on word + character n-grams handles the bulk in milliseconds; per-label thresholds are tuned on validation data.
- **The LLM** is reserved for comments where the ML score sits near its decision threshold, measured in log-odds, so one knob (the band width) trades accuracy against LLM spend. The band is set from an *escalation budget*.
- **Severity floors**: for high-risk labels (self-harm) even a faint ML signal gets an LLM second opinion. A missed self-harm post costs far more than one extra call.
- **Graceful degradation**: if the LLM is down or returns invalid JSON, the ML decision is used and the failure is logged.
""")
