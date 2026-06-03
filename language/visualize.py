#!/usr/bin/env python3
"""
visualize.py — render an interactive HTML viewer of top-activating sequences
for each matched neuron across N >= 1 partner models.

Two input modes:

1. Multi-model (run_anchor_pipeline.sh output):
     --manifest path/to/rosetta/manifest.json
   The manifest references rosetta_anchors.json + per-target-model cross
   files + the anchor-side activation collection dir.

2. Pairwise (run_pipeline.sh output):
     --cross-activations path/to/cross_activations.json
   The single cross file is treated as a one-partner-model anchor run.
   Each best-buddy pair becomes a degenerate anchor with one partner.

The on-disk artifacts are unchanged; this script just renders them.
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# -----------------------------------------------------------------------------
# Adapter: multi-model anchor pipeline outputs
# -----------------------------------------------------------------------------


def _load_act_anchor_layers(act_anchor_dir: str) -> Dict[int, Dict[int, dict]]:
    """Load all act_anchor/layer_{L}_activations.json files into {layer: {neuron: entry}}."""
    by_layer: Dict[int, Dict[int, dict]] = {}
    if not act_anchor_dir or not os.path.isdir(act_anchor_dir):
        return by_layer
    for path in sorted(glob.glob(os.path.join(act_anchor_dir, "layer_*_activations.json"))):
        fname = os.path.basename(path)
        try:
            layer = int(fname.split("_")[1])
        except (IndexError, ValueError):
            continue
        try:
            d = _load_json(path)
        except Exception:
            continue
        by_layer[layer] = {int(k): v for k, v in d.items()}
    return by_layer


def _build_from_manifest(manifest_path: str) -> Tuple[dict, List[dict]]:
    manifest = _load_json(manifest_path)
    anchor_model = manifest.get("anchor_model", "Anchor")
    rosetta = _load_json(manifest["rosetta_path"])

    cross_paths = manifest.get("cross_paths", [])
    model_labels = manifest.get("model_labels", [])
    act_anchor_dir = manifest.get("act_anchor_dir") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(manifest_path))), "act_anchor"
    )

    # Load per-partner cross files into (label, {(m1_layer, m1_neuron) -> pair})
    cross_by_label: List[Tuple[str, Dict[Tuple[int, int], dict]]] = []
    for i, path in enumerate(cross_paths):
        data = _load_json(path)
        pairs_by_key = {
            (int(p.get("model1_layer", 0)), int(p.get("model1_neuron", 0))): p
            for p in data.get("pairs", [])
        }
        label = model_labels[i] if i < len(model_labels) else data.get("model2_name", f"Model {i+2}")
        cross_by_label.append((label, pairs_by_key))

    # Anchor activations grouped by (layer, neuron)
    act_by_layer = _load_act_anchor_layers(act_anchor_dir)

    # Build per-anchor records and summaries
    all_anchors: List[dict] = []
    summaries: List[dict] = []
    for anchor_id, r in enumerate(rosetta):
        l1 = int(r.get("model1_layer", 0))
        n1 = int(r.get("model1_neuron", 0))
        avg_corr = _safe_float(r.get("avg_correlation", 0.0))

        # Per-partner header info (matched neuron + correlation)
        per_model = r.get("per_model") or []
        partners_header: List[dict] = []
        for entry in per_model:
            partners_header.append({
                "model": entry.get("model", ""),
                "m2_layer": entry.get("model2_layer", ""),
                "m2_neuron": int(entry.get("model2_neuron", 0)),
                "correlation": _safe_float(entry.get("correlation", 0.0)),
            })

        # Anchor examples come from act_anchor[l1][n1]
        anchor_entry = act_by_layer.get(l1, {}).get(n1, {})
        anchor_examples = anchor_entry.get("examples", []) or []

        # For each partner, get the matching pair's example dict keyed by example_id
        partner_ex_maps: List[Tuple[str, Dict[str, dict]]] = []
        for label, pairs_by_key in cross_by_label:
            pair = pairs_by_key.get((l1, n1), {})
            ex_by_id = {ex.get("example_id"): ex for ex in pair.get("examples", []) if ex.get("example_id")}
            partner_ex_maps.append((label, ex_by_id))

        # Assemble per-anchor examples
        out_examples: List[dict] = []
        for rank, ex in enumerate(anchor_examples):
            exid = ex.get("example_id")
            partner_acts: List[dict] = []
            for label, ex_by_id in partner_ex_maps:
                p_ex = ex_by_id.get(exid, {}) if exid else {}
                partner_acts.append({
                    "model": label,
                    "span": p_ex.get("cross_span", []),
                    "span_pool_activation": p_ex.get("cross_span_pool_activation"),
                    "activation": p_ex.get("cross_activation"),
                    "context_before": p_ex.get("cross_context_before", []),
                    "context_after": p_ex.get("cross_context_after", []),
                })
            out_examples.append({
                "rank": rank,
                "doc_id": ex.get("doc_id", ""),
                "position": ex.get("position", 0),
                "token": ex.get("token", ""),
                "m1_activation": _safe_float(ex.get("activation", 0.0)),
                "m1_context_before": ex.get("context_before", []),
                "m1_context_after": ex.get("context_after", []),
                "partner_activations": partner_acts,
            })

        anchor_record = {
            "anchor_id": anchor_id,
            "anchor_model": anchor_model,
            "m1_layer": l1,
            "m1_neuron": n1,
            "avg_correlation": avg_corr,
            "partners": partners_header,
            "examples": out_examples,
        }
        all_anchors.append(anchor_record)

        summaries.append({
            "anchor_id": anchor_id,
            "m1_layer": l1,
            "m1_neuron": n1,
            "avg_correlation": avg_corr,
            "partners": partners_header,
        })

    master = {
        "anchor_model": anchor_model,
        "partner_models": [lbl for lbl, _ in cross_by_label],
        "num_anchors": len(rosetta),
        "top_k": max((len(a["examples"]) for a in all_anchors), default=0),
        "anchors_index": summaries,
    }
    return master, all_anchors


# -----------------------------------------------------------------------------
# Adapter: pairwise pipeline output (single cross_activations.json)
# -----------------------------------------------------------------------------


def _build_from_pairwise(cross_path: str) -> Tuple[dict, List[dict]]:
    data = _load_json(cross_path)
    anchor_model = data.get("model1_name", "Model 1")
    partner_model = data.get("model2_name", "Model 2")
    pairs = data.get("pairs", []) or []

    all_anchors: List[dict] = []
    summaries: List[dict] = []
    top_k_max = 0

    for anchor_id, pair in enumerate(pairs):
        l1 = int(pair.get("model1_layer", 0))
        n1 = int(pair.get("model1_neuron", 0))
        l2 = int(pair.get("model2_layer", 0))
        n2 = int(pair.get("model2_neuron", 0))
        corr = _safe_float(pair.get("correlation", 0.0))
        examples = pair.get("examples", []) or []
        top_k_max = max(top_k_max, len(examples))

        partners_header = [{
            "model": partner_model,
            "m2_layer": l2,
            "m2_neuron": n2,
            "correlation": corr,
        }]

        out_examples: List[dict] = []
        for rank, ex in enumerate(examples):
            out_examples.append({
                "rank": rank,
                "doc_id": ex.get("doc_id", ""),
                "position": ex.get("position", 0),
                "token": ex.get("token", ""),
                "m1_activation": _safe_float(ex.get("activation", 0.0)),
                "m1_context_before": ex.get("context_before", []),
                "m1_context_after": ex.get("context_after", []),
                "partner_activations": [{
                    "model": partner_model,
                    "span": ex.get("cross_span", []),
                    "span_pool_activation": ex.get("cross_span_pool_activation"),
                    "activation": ex.get("cross_activation"),
                    "context_before": ex.get("cross_context_before", []),
                    "context_after": ex.get("cross_context_after", []),
                }],
            })

        anchor_record = {
            "anchor_id": anchor_id,
            "anchor_model": anchor_model,
            "m1_layer": l1,
            "m1_neuron": n1,
            "avg_correlation": corr,
            "partners": partners_header,
            "examples": out_examples,
        }
        all_anchors.append(anchor_record)

        summaries.append({
            "anchor_id": anchor_id,
            "m1_layer": l1,
            "m1_neuron": n1,
            "avg_correlation": corr,
            "partners": partners_header,
        })

    master = {
        "anchor_model": anchor_model,
        "partner_models": [partner_model],
        "num_anchors": len(pairs),
        "top_k": top_k_max,
        "anchors_index": summaries,
    }
    return master, all_anchors


# -----------------------------------------------------------------------------
# Rendering — embedded CSS + JS, single self-contained HTML.
# -----------------------------------------------------------------------------


CSS = """
:root {
  --bg: #0e1117; --panel: #12161d; --panel2: #161b22; --border: #22272e;
  --text: #fafafa; --muted: #8b949e; --accent1: #7fd3ff; --badge: #1f252d;
}
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; background: var(--bg); color: var(--text);
  font: 13px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.app { display: flex; height: 100vh; }
.sidebar { width: 340px; min-width: 340px; background: var(--panel);
  border-right: 1px solid var(--border); display: flex; flex-direction: column; }
.sidebar .hdr { padding: 12px 14px; border-bottom: 1px solid var(--border); }
.sidebar h1 { font-size: 14px; margin: 0 0 4px 0; }
.sidebar .models { font-size: 11px; color: var(--muted); }
.sidebar .filters { padding: 10px 14px; border-bottom: 1px solid var(--border);
  display: flex; flex-direction: column; gap: 8px; }
.sidebar .filters input {
  background: var(--panel2); color: var(--text); border: 1px solid var(--border);
  padding: 4px 6px; font: inherit; border-radius: 4px; width: 100%; }
.sidebar .filters label { font-size: 11px; color: var(--muted); display: flex;
  flex-direction: column; gap: 2px; }
.sidebar .filters button { background: var(--panel2); color: var(--text);
  border: 1px solid var(--border); padding: 4px 8px; border-radius: 4px;
  cursor: pointer; font: inherit; }
.sidebar .filters .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.count { padding: 6px 14px; font-size: 11px; color: var(--muted);
  border-bottom: 1px solid var(--border); }
.list { overflow-y: auto; flex: 1; }
.pager { padding: 6px 14px; display: flex; justify-content: space-between;
  align-items: center; border-top: 1px solid var(--border); font-size: 11px; color: var(--muted); }
.pager button { background: var(--panel2); color: var(--text);
  border: 1px solid var(--border); padding: 2px 8px; border-radius: 4px;
  cursor: pointer; font: inherit; }
.anchor-item { padding: 8px 14px; border-bottom: 1px solid var(--border); cursor: pointer; }
.anchor-item:hover { background: var(--panel2); }
.anchor-item.selected { background: #1c2230; border-left: 3px solid var(--accent1); padding-left: 11px; }
.anchor-item .title { font-size: 12px; font-weight: 600; }
.anchor-item .corr { float: right; color: var(--accent1); font-weight: 600; }
.anchor-item .meta { font-size: 11px; color: var(--muted); margin-top: 2px; }
.main { flex: 1; overflow-y: auto; padding: 16px 20px; }
.main .empty { color: var(--muted); padding: 40px; text-align: center; }
.anchor-header { padding-bottom: 10px; border-bottom: 1px solid var(--border);
  margin-bottom: 14px; }
.anchor-header h2 { font-size: 18px; margin: 0 0 4px 0; }
.anchor-header .sub { font-size: 12px; color: var(--muted); }
.badges { margin-top: 6px; display: flex; gap: 8px; flex-wrap: wrap; }
.badge { display: inline-block; background: var(--badge); color: var(--text);
  border: 1px solid var(--border); border-radius: 4px; padding: 2px 8px; font-size: 11px; }
.badge.corr { color: var(--accent1); border-color: #22577a; }
.example-block { margin: 14px 0; padding: 10px; background: var(--panel2);
  border: 1px solid var(--border); border-radius: 6px; }
.example-block .rank { font-size: 11px; color: var(--muted); margin-bottom: 6px; }
.example-row { margin: 4px 0; display: flex; align-items: center; gap: 8px; }
.example-row .model-lab { font-size: 11px; color: var(--muted); min-width: 96px;
  font-family: ui-monospace, Menlo, Consolas, monospace; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; }
.tokens { display: flex; flex-wrap: wrap; gap: 3px; align-items: center; flex: 1; }
.tok { padding: 2px 5px; border-radius: 3px; font-family: ui-monospace, Menlo, Consolas, monospace;
  font-size: 12px; white-space: pre-wrap; line-height: 1.6; }
.tok.main { font-weight: 600; }
.act-val { margin-left: 5px; padding: 1px 4px; background: rgba(0,0,0,0.35);
  border-radius: 3px; font-size: 10px; }
"""


JS = r"""
(function() {
  const DATA = window.__ANCHOR_DATA__;
  const MASTER = window.__MASTER__;
  const INLINE_ANCHORS = window.__INLINE_ANCHORS__;  // array indexed by anchor_id
  const PAGE_SIZE = 50;

  const state = {
    q: "", cmin: 0, cmax: 1,
    l1min: 0, l1max: 9999,
    page: 0, selected: null,
  };

  function loadFromHash() {
    const h = (location.hash || "").replace(/^#/, "");
    if (!h) return;
    for (const p of h.split("&")) {
      const [k, v] = p.split("=");
      if (!k) continue;
      const dv = decodeURIComponent(v || "");
      if (k === "q") state.q = dv;
      else if (k === "cmin") state.cmin = parseFloat(dv);
      else if (k === "cmax") state.cmax = parseFloat(dv);
      else if (k === "l1") { const [a,b] = dv.split("-"); state.l1min=+a||0; state.l1max=+b||9999; }
      else if (k === "page") state.page = parseInt(dv, 10) || 0;
      else if (k === "anchor") state.selected = parseInt(dv, 10);
    }
  }
  function saveToHash() {
    const p = [];
    if (state.q) p.push("q=" + encodeURIComponent(state.q));
    if (state.cmin > 0) p.push("cmin=" + state.cmin);
    if (state.cmax < 1) p.push("cmax=" + state.cmax);
    if (state.l1min > 0 || state.l1max < 9999) p.push("l1=" + state.l1min + "-" + state.l1max);
    if (state.page > 0) p.push("page=" + state.page);
    if (state.selected != null) p.push("anchor=" + state.selected);
    history.replaceState(null, "", "#" + p.join("&"));
  }

  function filtered() {
    const q = state.q.trim().toLowerCase();
    const qLnum = q.match(/^l(\d+)$/);
    const qNnum = q.match(/^n(\d+)$/);
    const qComp = q.match(/^(\d+)\/(\d+)$/);
    const out = [];
    for (const idx of DATA) {
      if (idx.avg_correlation < state.cmin || idx.avg_correlation > state.cmax) continue;
      if (idx.m1_layer < state.l1min || idx.m1_layer > state.l1max) continue;
      if (q) {
        let match = false;
        if (qLnum) match = (idx.m1_layer == +qLnum[1]);
        else if (qNnum) match = (idx.m1_neuron == +qNnum[1]);
        else if (qComp) match = (idx.m1_layer == +qComp[1] && idx.m1_neuron == +qComp[2]);
        if (!match) continue;
      }
      out.push(idx);
    }
    // Always sorted by avg correlation desc.
    out.sort((a, b) => b.avg_correlation - a.avg_correlation);
    return out;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  function colorStyle(raw, maxRaw) {
    const t = Math.max(-1, Math.min(1, raw / (Math.abs(maxRaw) || 1)));
    const t2 = Math.abs(t);
    let r, g, b;
    if (t >= 0) { r = 35+Math.round(t2*220); g = 38+Math.round(t2*25); b = 45+Math.round(t2*10); }
    else        { r = 35+Math.round(t2*30);  g = 38+Math.round(t2*60); b = 45+Math.round(t2*210); }
    const fg = t2 > 0.35 ? "#fff" : "#c8c8c8";
    return `background:rgb(${r},${g},${b});color:${fg}`;
  }
  function renderTok(tok, raw, maxRaw) {
    if (tok == null) tok = "";
    return `<span class="tok" style="${colorStyle(raw, maxRaw)}" title="raw=${(+raw).toFixed(4)}">${escapeHtml(tok)}</span>`;
  }
  function renderMainTok(tok, raw, maxRaw, accent, badge) {
    return `<span class="tok main" style="${colorStyle(raw, maxRaw)};box-shadow:0 0 0 2px ${accent}" title="raw=${(+raw).toFixed(4)}">${escapeHtml(tok)}<span class="act-val">${escapeHtml(badge)}</span></span>`;
  }

  function shortModel(m) {
    return String(m || "").split("/").pop();
  }

  function renderAnchorItem(idx, isSel) {
    const title = `L${idx.m1_layer}/N${idx.m1_neuron}`;
    const partnerLine = (idx.partners || []).map(p =>
      `${shortModel(p.model)} L${p.m2_layer}/N${p.m2_neuron} (${(+p.correlation).toFixed(2)})`
    ).join(" &middot; ");
    return `<div class="anchor-item ${isSel ? 'selected' : ''}" data-anchor="${idx.anchor_id}">
      <div><span class="corr">${(+idx.avg_correlation).toFixed(3)}</span>
        <span class="title">${title}</span></div>
      <div class="meta">${partnerLine}</div>
    </div>`;
  }

  function renderExampleBlock(ex, i, maxRaw, anchorModel, partnerModels) {
    // anchor row
    const m1Before = (ex.m1_context_before || []).map(c =>
      renderTok(c.token || "", +c.activation || 0, maxRaw)).join(" ");
    const m1After = (ex.m1_context_after || []).map(c =>
      renderTok(c.token || "", +c.activation || 0, maxRaw)).join(" ");
    const m1Raw = +ex.m1_activation || 0;
    const mainTok = renderMainTok(ex.token || "", m1Raw, maxRaw, "var(--accent1)",
                                   `a=${m1Raw.toFixed(2)}`);
    let rows = `<div class="example-row">
      <span class="model-lab" title="${escapeHtml(anchorModel)}">${shortModel(anchorModel)}</span>
      <div class="tokens">${m1Before} ${mainTok} ${m1After}</div>
    </div>`;

    // partner rows
    for (let m = 0; m < (ex.partner_activations||[]).length; m++) {
      const p = ex.partner_activations[m];
      const modelLabel = partnerModels[m] || p.model || "?";
      rows += renderPartnerRow(p, modelLabel);
    }

    return `<div class="example-block">
      <div class="rank">#${i+1} &middot; doc=${escapeHtml(String(ex.doc_id||""))} pos=${ex.position||0}</div>
      ${rows}
    </div>`;
  }

  function renderPartnerRow(p, modelLabel) {
    const span = p.span || [];
    const before = (p.context_before || []).map(c =>
      renderTok(c.token || "", +c.activation || 0, 1)).join(" ");
    const after = (p.context_after || []).map(c =>
      renderTok(c.token || "", +c.activation || 0, 1)).join(" ");
    const pool = p.span_pool_activation;
    const pa = p.activation;
    const poolOk = (pool != null && !isNaN(+pool));
    const paOk   = (pa   != null && !isNaN(+pa));
    let spanHtml;
    if (span.length > 0) {
      const maxLocal = Math.max(1e-6, ...span.map(s => +s.activation || 0));
      spanHtml = span.map(s => renderTok(s.token || "", +s.activation || 0, maxLocal)).join(" ");
    } else if (paOk) {
      spanHtml = `<span class="tok" style="${colorStyle(+pa, 1)}">?</span>`;
    } else {
      spanHtml = `<span class="tok" style="opacity:.4">no span match</span>`;
    }
    const poolStr = poolOk ? `pool=${(+pool).toFixed(2)}`
                   : paOk   ? `a=${(+pa).toFixed(2)}`
                   : '<span style="opacity:.5">—</span>';
    return `<div class="example-row">
      <span class="model-lab" title="${escapeHtml(modelLabel)}">${shortModel(modelLabel)}</span>
      <div class="tokens">${before} <span class="badge">${poolStr}</span> ${spanHtml} ${after}</div>
    </div>`;
  }

  function fetchAnchor(anchorId) {
    if (INLINE_ANCHORS) return INLINE_ANCHORS[anchorId];
    return null;
  }

  function renderMain() {
    const mainEl = document.getElementById("main");
    if (state.selected == null) {
      mainEl.innerHTML = '<div class="empty">Select an anchor from the sidebar.</div>';
      return;
    }
    const a = fetchAnchor(state.selected);
    if (!a) {
      mainEl.innerHTML = '<div class="empty">Anchor not found.</div>';
      return;
    }
    const maxRaw = Math.max(1e-6, ...((a.examples||[]).map(e => +e.m1_activation || 0)));
    const partnerModelNames = (a.partners||[]).map(p => p.model);
    const partnerDescr = (a.partners||[]).map(p =>
      `<span class="badge">${shortModel(p.model)} L${p.m2_layer}/N${p.m2_neuron} corr ${(+p.correlation).toFixed(3)}</span>`
    ).join("");
    const header = `<div class="anchor-header">
      <h2>Anchor L${a.m1_layer}/N${a.m1_neuron}</h2>
      <div class="sub">${escapeHtml(a.anchor_model || MASTER.anchor_model || '')}</div>
      <div class="badges">
        <span class="badge corr">avg corr ${(+a.avg_correlation).toFixed(3)}</span>
        <span class="badge">anchor_id ${a.anchor_id}</span>
        ${partnerDescr}
      </div></div>`;
    const body = (a.examples || []).map((e,i) =>
      renderExampleBlock(e, i, maxRaw, a.anchor_model || MASTER.anchor_model || 'anchor', partnerModelNames)
    ).join("");
    mainEl.innerHTML = header + body;
  }

  function renderSidebar() {
    const all = filtered();
    document.getElementById("count").textContent = `${all.length} anchors`;
    const start = state.page * PAGE_SIZE;
    const page = all.slice(start, start + PAGE_SIZE);
    const listEl = document.getElementById("list");
    listEl.innerHTML = page.map(idx => renderAnchorItem(idx, idx.anchor_id === state.selected)).join("");
    for (const el of listEl.querySelectorAll(".anchor-item")) {
      el.onclick = () => {
        state.selected = parseInt(el.dataset.anchor, 10);
        saveToHash(); renderSidebar(); renderMain();
      };
    }
    document.getElementById("pgInfo").textContent =
      `page ${state.page+1}/${Math.max(1, Math.ceil(all.length/PAGE_SIZE))}`;
  }

  function wire() {
    const bind = (id, prop, parse=((v)=>v)) => {
      const el = document.getElementById(id);
      el.value = state[prop];
      el.oninput = () => { state[prop] = parse(el.value); state.page = 0; saveToHash(); renderSidebar(); };
    };
    bind("q", "q");
    bind("cmin", "cmin", v => +v || 0);
    bind("cmax", "cmax", v => +v || 1);
    const l1 = document.getElementById("l1");
    l1.value = state.l1min === 0 && state.l1max === 9999 ? "" : `${state.l1min}-${state.l1max}`;
    l1.oninput = () => {
      const m = l1.value.match(/^(\d+)-(\d+)$/);
      if (m) { state.l1min = +m[1]; state.l1max = +m[2]; }
      else { state.l1min = 0; state.l1max = 9999; }
      state.page = 0; saveToHash(); renderSidebar();
    };
    document.getElementById("reset").onclick = () => {
      Object.assign(state, { q:"", cmin:0, cmax:1, l1min:0, l1max:9999, page:0, selected:null });
      location.hash = ""; init();
    };
    document.getElementById("prev").onclick = () => {
      if (state.page > 0) { state.page--; saveToHash(); renderSidebar(); }
    };
    document.getElementById("next").onclick = () => {
      const all = filtered();
      if ((state.page + 1) * PAGE_SIZE < all.length) { state.page++; saveToHash(); renderSidebar(); }
    };
  }

  function init() {
    loadFromHash();
    wire();
    renderSidebar();
    renderMain();
  }
  document.addEventListener("DOMContentLoaded", init);
})();
"""


def _script_safe(s: str) -> str:
    """Neutralize sequences that would prematurely terminate <script> or trigger the HTML parser."""
    return (s.replace("</", "<\\/")
             .replace("<!--", "<\\!--")
             .replace("]]>", "]]\\u003e"))


def render_html(master: dict, all_anchors: List[dict], title: str) -> str:
    anchor_model = html.escape(master.get("anchor_model", ""))
    partner_models = master.get("partner_models", [])
    num_anchors = master.get("num_anchors", len(master.get("anchors_index", [])))
    top_k = master.get("top_k", 0)
    partners_line = " &middot; ".join(html.escape(p) for p in partner_models)
    summaries = master.get("anchors_index", [])

    by_id = {a["anchor_id"]: a for a in all_anchors}
    max_id = max(by_id.keys(), default=-1)
    inline_arr = [by_id.get(i) for i in range(max_id + 1)] if all_anchors else None

    master_js = _script_safe(json.dumps({
        "anchor_model": master.get("anchor_model"),
        "partner_models": partner_models,
        "num_anchors": num_anchors,
        "top_k": top_k,
        "anchors_index": summaries,
    }))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="hdr">
      <h1>{html.escape(title)}</h1>
      <div class="models">{anchor_model} &larr; {partners_line} &middot; {num_anchors} anchors &middot; top-{top_k}</div>
    </div>
    <div class="filters">
      <label>search (L9, N1024, or 9/1024)<input id="q" placeholder="L9 / N1024 / 9/1024"></label>
      <div class="row2">
        <label>min avg corr <input id="cmin" type="number" step="0.01" min="-1" max="1"></label>
        <label>max avg corr <input id="cmax" type="number" step="0.01" min="-1" max="1"></label>
      </div>
      <label>anchor layer range (a-b) <input id="l1" placeholder="e.g. 0-20"></label>
      <button id="reset">reset</button>
    </div>
    <div class="count" id="count">0 anchors</div>
    <div class="list" id="list"></div>
    <div class="pager">
      <button id="prev">&lsaquo; prev</button>
      <span id="pgInfo"></span>
      <button id="next">next &rsaquo;</button>
    </div>
  </aside>
  <main class="main" id="main"></main>
</div>
<script>window.__MASTER__ = {master_js};</script>
<script>window.__ANCHOR_DATA__ = {_script_safe(json.dumps(summaries))};</script>
<script>window.__INLINE_ANCHORS__ = {_script_safe(json.dumps(inline_arr))};</script>
<script>{JS}</script>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Input modes — exactly one required.
    p.add_argument("--manifest", type=str, default=None,
                   help="Multi-model: manifest.json from run_anchor_pipeline.sh")
    p.add_argument("--cross-activations", "--cross_activations", dest="cross_activations",
                   type=str, default=None,
                   help="Pairwise: a single cross_activations.json from run_pipeline.sh")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--title", default="Rosetta Anchor Activations")
    p.add_argument("--num-anchors", "--num_anchors", dest="num_anchors", type=int, default=None,
                   help="Cap on how many top anchors (sorted by avg_correlation) to include in the viewer. "
                        "The disk artifacts are unchanged. Default: include all.")
    args = p.parse_args()

    if bool(args.manifest) == bool(args.cross_activations):
        raise SystemExit("Provide exactly one of --manifest or --cross-activations.")

    if args.cross_activations:
        master, all_anchors = _build_from_pairwise(args.cross_activations)
    else:
        master, all_anchors = _build_from_manifest(args.manifest)

    # Sort by avg_correlation desc and apply optional cap.
    sorted_pairs = sorted(zip(master["anchors_index"], all_anchors),
                          key=lambda t: _safe_float(t[0].get("avg_correlation", 0.0)),
                          reverse=True)
    total = len(sorted_pairs)
    if args.num_anchors is not None and args.num_anchors > 0:
        sorted_pairs = sorted_pairs[: args.num_anchors]
    # Re-assign anchor_id sequentially so the viewer's index lookups are dense.
    new_summaries: List[dict] = []
    new_anchors: List[dict] = []
    for new_id, (summary, anchor) in enumerate(sorted_pairs):
        s = dict(summary)
        a = dict(anchor)
        s["anchor_id"] = new_id
        a["anchor_id"] = new_id
        new_summaries.append(s)
        new_anchors.append(a)
    master["anchors_index"] = new_summaries
    master["num_anchors"] = total

    print(f"[visualize] rendering {len(new_anchors)} / {total} anchors")
    html_out = render_html(master, new_anchors, title=args.title)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
