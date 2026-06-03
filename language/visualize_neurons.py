#!/usr/bin/env python3
"""
viz_activations.py — HTML visualizer for cached-corpus activation dumps

Works with the *collector* output you posted:

output_dir/
  metadata.json
  layer_{L}_activations.json   # dict: neuron_idx(str) -> { layer, neuron_idx, examples, mean_activation, ... }

Fixes vs your current visualizer:
- ✅ Accepts collector schema: uses `examples` (aliases to `top_activations`)
- ✅ Neuron IDs are per-layer (best-buddies): uses `neuronsByLayer[layer]`
- ✅ Random / find / similar-neurons only choose neurons that exist in the selected layer
- ✅ Works even when different layers have different neuron sets

Usage:
  python viz_activations.py --input_dir ./outputs/act_m1 --output_html neuron_explorer.html
"""

from __future__ import annotations

import argparse
import html
import json
import os
from typing import Any, Dict, List, Tuple


def load_collector_dir(input_dir: str) -> Tuple[Dict[int, Dict[int, dict]], dict]:
    meta_path = os.path.join(input_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata.json in {input_dir}")

    meta = json.load(open(meta_path, "r", encoding="utf-8"))

    # infer layers from metadata, else scan files
    layers: List[int] = []
    if isinstance(meta.get("layers"), list):
        layers = [int(x) for x in meta["layers"]]
    else:
        # scan
        for fn in os.listdir(input_dir):
            if fn.startswith("layer_") and fn.endswith("_activations.json"):
                mid = fn[len("layer_") : -len("_activations.json")]
                if mid.isdigit():
                    layers.append(int(mid))
        layers.sort()

    all_data: Dict[int, Dict[int, dict]] = {}
    for L in layers:
        path = os.path.join(input_dir, f"layer_{L}_activations.json")
        if not os.path.exists(path):
            continue
        raw = json.load(open(path, "r", encoding="utf-8"))
        layer_map: Dict[int, dict] = {}
        for k, v in raw.items():
            try:
                n = int(k)
            except Exception:
                continue

            # Normalize entry: ensure top_activations exists (alias examples -> top_activations)
            examples = v.get("top_activations", v.get("examples", [])) or []
            # Some collectors might store float-like strings; sanitize a bit
            v_norm = {
                "layer": int(v.get("layer", L)),
                "neuron_idx": int(v.get("neuron_idx", n)),
                "top_activations": examples,
                "mean_activation": float(v.get("mean_activation", 0.0)),
                "std_activation": float(v.get("std_activation", 0.0)),
                "max_activation": float(v.get("max_activation", 0.0)),
                "min_activation": float(v.get("min_activation", 0.0)),
            }
            layer_map[n] = v_norm
        if layer_map:
            all_data[L] = layer_map

    if not all_data:
        raise ValueError(f"No layer_*_activations.json data found in {input_dir}")

    return all_data, meta


def format_activation_js() -> str:
    # Keep as JS source string (small helper)
    return r"""
function formatActivation(val) {
  if (!isFinite(val)) return "0.00";
  const a = Math.abs(val);
  if (a >= 10) return val.toFixed(1);
  if (a >= 1) return val.toFixed(2);
  return val.toFixed(2);
}
"""


def generate_html(
    all_neuron_data: Dict[int, Dict[int, dict]],
    meta: dict,
    output_path: str,
) -> None:
    layers = sorted(all_neuron_data.keys())
    neurons_by_layer = {str(L): sorted(list(all_neuron_data[L].keys())) for L in layers}

    model_name = str(meta.get("model_name", meta.get("model", "model")))

    # JSON for JS:
    json_data = {str(L): {str(n): all_neuron_data[L][n] for n in all_neuron_data[L]} for L in layers}

    first_layer = layers[0]
    first_neuron = neurons_by_layer[str(first_layer)][0]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Neuron Activation Explorer — {html.escape(model_name)}</title>
<style>
  :root {{
    --bg-primary: #0e1117;
    --bg-secondary: #1a1d24;
    --bg-tertiary: #262a33;
    --text-primary: #fafafa;
    --text-secondary: #a0a0a0;
    --accent: #ff4b4b;
    --accent-hover: #ff6b6b;
    --border: #333;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
    min-height: 100vh;
  }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 2rem; }}
  header {{ margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border); }}
  h1 {{
    font-size: 2.2rem; font-weight: 800; margin-bottom: 0.5rem;
  }}
  .subtitle {{ color: var(--text-secondary); font-size: 1.0rem; }}
  .controls {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
    padding: 1.25rem;
    background: var(--bg-secondary);
    border-radius: 12px;
    border: 1px solid var(--border);
  }}
  .control-group {{ display: flex; flex-direction: column; gap: 0.5rem; }}
  .control-group label {{ font-size: 0.9rem; color: var(--text-secondary); font-weight: 600; }}
  .input-wrapper {{ display: flex; gap: 0.5rem; align-items: center; }}
  input[type="number"] {{
    flex: 1;
    padding: 0.7rem 0.9rem;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 1rem;
  }}
  input[type="number"]:focus {{ outline: none; border-color: var(--accent); }}
  .btn {{
    padding: 0.7rem 1.1rem;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 0.95rem;
    cursor: pointer;
  }}
  .btn:hover {{ background: var(--bg-secondary); border-color: var(--text-secondary); }}
  .btn-primary {{
    background: var(--accent);
    border-color: var(--accent);
    font-weight: 700;
  }}
  .btn-primary:hover {{ background: var(--accent-hover); border-color: var(--accent-hover); }}
  .btn-group {{ display: flex; gap: 0.5rem; flex-wrap: wrap; }}
  .toggle-group {{ display: flex; align-items: center; gap: 0.75rem; }}
  .toggle {{
    position: relative;
    width: 48px; height: 26px;
    background: var(--bg-tertiary);
    border-radius: 13px;
    cursor: pointer;
    transition: background 0.2s;
    border: 1px solid var(--border);
  }}
  .toggle.active {{ background: var(--accent); border-color: var(--accent); }}
  .toggle::after {{
    content: "";
    position: absolute;
    top: 3px; left: 3px;
    width: 18px; height: 18px;
    background: white;
    border-radius: 50%;
    transition: transform 0.2s;
  }}
  .toggle.active::after {{ transform: translateX(22px); }}
  .toggle-label {{ font-size: 0.9rem; color: var(--text-secondary); }}
  .main-content {{
    display: grid;
    grid-template-columns: 1fr 350px;
    gap: 1.5rem;
  }}
  @media (max-width: 1000px) {{
    .main-content {{ grid-template-columns: 1fr; }}
  }}
  .features-panel {{
    background: var(--bg-secondary);
    border-radius: 12px;
    border: 1px solid var(--border);
    overflow: hidden;
  }}
  .panel-header {{
    padding: 1rem 1.25rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
  }}
  .panel-title {{ font-size: 1.05rem; font-weight: 800; color: var(--accent); }}
  .neuron-info {{ font-size: 0.85rem; color: var(--text-secondary); }}
  .features-content {{
    padding: 1.25rem;
    max-height: 720px;
    overflow-y: auto;
  }}
  .feature-item {{
    margin-bottom: 1.25rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--border);
  }}
  .feature-item:last-child {{ margin-bottom: 0; padding-bottom: 0; border-bottom: none; }}
  .feature-label {{ font-size: 0.85rem; color: var(--accent); margin-bottom: 0.6rem; font-weight: 700; }}
  .token-display {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.2rem;
    align-items: baseline;
    line-height: 2.2;
  }}
  .token {{
    display: inline-flex;
    align-items: center;
    padding: 0.18rem 0.35rem;
    border-radius: 4px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-size: 0.9rem;
    white-space: pre;
  }}
  .token:hover {{ outline: 1px solid rgba(255,255,255,0.15); }}
  .token-value {{ font-size: 0.65rem; margin-left: 0.25rem; opacity: 0.85; font-weight: 700; }}
  .token-main {{ box-shadow: 0 0 0 2px var(--accent); font-weight: 900; }}
  .sidebar {{ display: flex; flex-direction: column; gap: 1.25rem; }}
  .stats-panel, .similar-panel {{
    background: var(--bg-secondary);
    border-radius: 12px;
    border: 1px solid var(--border);
    padding: 1.25rem;
  }}
  .stats-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-top: 0.75rem;
  }}
  .stat-item {{
    background: var(--bg-tertiary);
    padding: 0.9rem;
    border-radius: 8px;
  }}
  .stat-label {{ font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.25rem; }}
  .stat-value {{ font-size: 1.1rem; font-weight: 900; }}
  .similar-title {{ font-size: 1.05rem; font-weight: 900; margin-bottom: 0.25rem; }}
  .similar-subtitle {{ font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.75rem; }}
  .similar-item {{
    background: var(--bg-tertiary);
    border-radius: 8px;
    padding: 0.85rem;
    margin-bottom: 0.65rem;
    cursor: pointer;
    border: 1px solid transparent;
  }}
  .similar-item:hover {{ border-color: var(--accent); transform: translateX(3px); }}
  .similar-item-header {{ font-size: 0.9rem; font-weight: 800; margin-bottom: 0.5rem; color: var(--accent); }}
  .similar-tokens {{ display: flex; flex-wrap: wrap; gap: 0.25rem; }}
  .similar-token {{
    font-size: 0.8rem;
    padding: 0.12rem 0.3rem;
    border-radius: 3px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    white-space: pre;
  }}
  .empty-state {{ text-align: center; padding: 2rem; color: var(--text-secondary); }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Neuron Activation Explorer</h1>
    <p class="subtitle">Explore top-activating contexts — {html.escape(model_name)}</p>
  </header>

  <div class="controls">
    <div class="control-group">
      <label>Layer</label>
      <div class="input-wrapper">
        <input type="number" id="layer-input" value="{first_layer}">
      </div>
    </div>

    <div class="control-group">
      <label>Neuron (per-layer)</label>
      <div class="input-wrapper">
        <input type="number" id="neuron-input" value="{first_neuron}">
      </div>
    </div>

    <div class="control-group">
      <label>Actions</label>
      <div class="btn-group">
        <button class="btn btn-primary" onclick="findNeuron()">Find</button>
        <button class="btn" onclick="randomNeuron()">Random</button>
        <button class="btn" onclick="goBack()">← Back</button>
      </div>
    </div>

    <div class="control-group">
      <label>Display</label>
      <div class="toggle-group">
        <div class="toggle active" id="show-values-toggle" onclick="toggleShowValues()"></div>
        <span class="toggle-label">Show activation values</span>
      </div>
    </div>
  </div>

  <div class="main-content">
    <div class="features-panel">
      <div class="panel-header">
        <span class="panel-title" id="panel-title"></span>
        <span class="neuron-info" id="neuron-info"></span>
      </div>
      <div class="features-content" id="features-content"></div>
    </div>

    <div class="sidebar">
      <div class="stats-panel">
        <div class="panel-title">Neuron Statistics</div>
        <div class="stats-grid">
          <div class="stat-item"><div class="stat-label">Max</div><div class="stat-value" id="stat-max">-</div></div>
          <div class="stat-item"><div class="stat-label">Mean</div><div class="stat-value" id="stat-mean">-</div></div>
          <div class="stat-item"><div class="stat-label">Std</div><div class="stat-value" id="stat-std">-</div></div>
          <div class="stat-item"><div class="stat-label">Min</div><div class="stat-value" id="stat-min">-</div></div>
        </div>
      </div>

      <div class="similar-panel">
        <div class="similar-title">Other Neurons</div>
        <div class="similar-subtitle">Click to explore (same/nearby layers)</div>
        <div id="similar-content"><div class="empty-state">Loading...</div></div>
      </div>
    </div>
  </div>
</div>

<script>
const neuronData = {json.dumps(json_data)};
const availableLayers = {json.dumps([int(x) for x in layers])};
const neuronsByLayer = {json.dumps({int(k): v for k, v in neurons_by_layer.items()})};

let history = [];
let currentLayer = {first_layer};
let currentNeuron = {first_neuron};
let showValues = true;

{format_activation_js()}

function layerNeurons(layer) {{
  return neuronsByLayer[layer] || [];
}}

function closestLayer(layerInput) {{
  let layer = availableLayers[0];
  for (const l of availableLayers) {{
    if (Math.abs(l - layerInput) < Math.abs(layer - layerInput)) layer = l;
  }}
  return layer;
}}

function clampNeuronToLayer(layer, neuron) {{
  const arr = layerNeurons(layer);
  if (arr.length === 0) return 0;
  if (arr.includes(neuron)) return neuron;
  let closest = arr[0];
  let minDiff = Math.abs(neuron - closest);
  for (const n of arr) {{
    const d = Math.abs(neuron - n);
    if (d < minDiff) {{ minDiff = d; closest = n; }}
  }}
  return closest;
}}

function updateNeuronInputBounds() {{
  const arr = layerNeurons(currentLayer);
  const el = document.getElementById('neuron-input');
  if (!el) return;
  if (arr.length === 0) {{
    el.min = 0; el.max = 0;
  }} else {{
    el.min = Math.min(...arr);
    el.max = Math.max(...arr);
  }}
}}

function getActivationColor(activation, maxAct, minAct) {{
  const range = maxAct - minAct;
  const normalized = range > 0 ? (activation - minAct) / range : 0.5;
  const clamped = Math.min(1, Math.max(0, normalized));
  const intensity = Math.pow(clamped, 0.6);

  const r = Math.round(50 + intensity * 205);
  const g = Math.round(50 - intensity * 40);
  const b = Math.round(55 - intensity * 45);
  return `rgb(${r}, ${g}, ${b})`;
}}

function getTextColor(activation, maxAct, minAct) {{
  const range = maxAct - minAct;
  const normalized = range > 0 ? (activation - minAct) / range : 0.5;
  return normalized > 0.25 ? '#fff' : '#bbb';
}}

function escapeHtml(text) {{
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}}

function findNeuron() {{
  const layerInput = parseInt(document.getElementById('layer-input').value);
  const neuronInput = parseInt(document.getElementById('neuron-input').value);

  const layer = closestLayer(layerInput);
  const neuron = clampNeuronToLayer(layer, neuronInput);

  history.push({{layer: currentLayer, neuron: currentNeuron}});
  currentLayer = layer;
  currentNeuron = neuron;

  document.getElementById('layer-input').value = currentLayer;
  document.getElementById('neuron-input').value = currentNeuron;

  updateDisplay();
}}

function randomNeuron() {{
  history.push({{layer: currentLayer, neuron: currentNeuron}});

  currentLayer = availableLayers[Math.floor(Math.random() * availableLayers.length)];
  const arr = layerNeurons(currentLayer);
  currentNeuron = arr[Math.floor(Math.random() * arr.length)];

  document.getElementById('layer-input').value = currentLayer;
  document.getElementById('neuron-input').value = currentNeuron;

  updateDisplay();
}}

function goBack() {{
  if (history.length === 0) return;
  const prev = history.pop();
  currentLayer = prev.layer;
  currentNeuron = prev.neuron;

  document.getElementById('layer-input').value = currentLayer;
  document.getElementById('neuron-input').value = currentNeuron;

  updateDisplay();
}}

function navigateToNeuron(layer, neuron) {{
  history.push({{layer: currentLayer, neuron: currentNeuron}});
  currentLayer = layer;
  currentNeuron = neuron;

  document.getElementById('layer-input').value = currentLayer;
  document.getElementById('neuron-input').value = currentNeuron;

  updateDisplay();
}}

function toggleShowValues() {{
  showValues = !showValues;
  document.getElementById('show-values-toggle').classList.toggle('active', showValues);
  updateDisplay();
}}

function updateDisplay() {{
  updateNeuronInputBounds();

  const data = neuronData[currentLayer]?.[currentNeuron];
  if (!data) {{
    document.getElementById('panel-title').textContent = `Layer ${currentLayer}, Neuron ${currentNeuron}`;
    document.getElementById('neuron-info').textContent = `No data`;
    document.getElementById('features-content').innerHTML = '<div class="empty-state">No data available for this neuron (try Find/Random)</div>';
    document.getElementById('similar-content').innerHTML = '<div class="empty-state">No other neurons found</div>';
    document.getElementById('stat-max').textContent = '-';
    document.getElementById('stat-mean').textContent = '-';
    document.getElementById('stat-std').textContent = '-';
    document.getElementById('stat-min').textContent = '-';
    return;
  }}

  const acts = data.top_activations || [];
  document.getElementById('panel-title').textContent = `Layer ${currentLayer}, Neuron ${currentNeuron}`;
  document.getElementById('neuron-info').textContent = `${acts.length} top activations`;

  document.getElementById('stat-max').textContent = formatActivation(data.max_activation);
  document.getElementById('stat-mean').textContent = formatActivation(data.mean_activation);
  document.getElementById('stat-std').textContent = formatActivation(data.std_activation);
  document.getElementById('stat-min').textContent = formatActivation(data.min_activation);

  const maxAct = data.max_activation;
  const minAct = data.min_activation;

  let featuresHtml = '';
  if (acts.length === 0) {{
    featuresHtml = '<div class="empty-state">No activations recorded for this neuron</div>';
  }} else {{
    acts.forEach((item, idx) => {{
      const peak = item.activation ?? 0.0;
      featuresHtml += `
        <div class="feature-item">
          <div class="feature-label">Example ${idx + 1} (peak: ${formatActivation(peak)})</div>
          <div class="token-display">
      `;

      // Context before
      (item.context_before || []).forEach(ctx => {{
        const a = ctx.activation ?? 0.0;
        const color = getActivationColor(a, maxAct, minAct);
        const textColor = getTextColor(a, maxAct, minAct);
        const escaped = escapeHtml(ctx.token ?? "");
        const valueHtml = showValues ? `<span class="token-value">${formatActivation(a)}</span>` : '';
        featuresHtml += `<span class="token" style="background: ${color}; color: ${textColor}">${escaped}${valueHtml}</span>`;
      }});

      // Main token
      const mainColor = getActivationColor(peak, maxAct, minAct);
      const mainTextColor = getTextColor(peak, maxAct, minAct);
      const escapedMain = escapeHtml(item.token ?? "");
      const mainValueHtml = showValues ? `<span class="token-value">${formatActivation(peak)}</span>` : '';
      featuresHtml += `<span class="token token-main" style="background: ${mainColor}; color: ${mainTextColor}">${escapedMain}${mainValueHtml}</span>`;

      // Context after
      (item.context_after || []).forEach(ctx => {{
        const a = ctx.activation ?? 0.0;
        const color = getActivationColor(a, maxAct, minAct);
        const textColor = getTextColor(a, maxAct, minAct);
        const escaped = escapeHtml(ctx.token ?? "");
        const valueHtml = showValues ? `<span class="token-value">${formatActivation(a)}</span>` : '';
        featuresHtml += `<span class="token" style="background: ${color}; color: ${textColor}">${escaped}${valueHtml}</span>`;
      }});

      featuresHtml += `</div></div>`;
    }});
  }}

  document.getElementById('features-content').innerHTML = featuresHtml;
  updateSimilarNeurons();
}}

function updateSimilarNeurons() {{
  let similarHtml = '';
  const nearbyLayers = availableLayers.filter(l => Math.abs(l - currentLayer) <= 2);

  let count = 0;
  for (const layer of nearbyLayers) {{
    const arr = layerNeurons(layer);
    for (const neuron of arr) {{
      if (count >= 8) break;
      if (layer === currentLayer && neuron === currentNeuron) continue;

      const data = neuronData[layer]?.[neuron];
      if (!data) continue;
      const acts = data.top_activations || [];
      if (acts.length === 0) continue;

      const topTokens = acts.slice(0, 4);
      const maxAct = data.max_activation;
      const minAct = data.min_activation;

      let tokensHtml = '';
      topTokens.forEach(item => {{
        const a = item.activation ?? 0.0;
        const color = getActivationColor(a, maxAct, minAct);
        const textColor = getTextColor(a, maxAct, minAct);
        const escaped = escapeHtml(item.token ?? "");
        tokensHtml += `<span class="similar-token" style="background: ${color}; color: ${textColor}">${escaped} <small>${formatActivation(a)}</small></span>`;
      }});

      similarHtml += `
        <div class="similar-item" onclick="navigateToNeuron(${layer}, ${neuron})">
          <div class="similar-item-header">Layer ${layer}, Neuron ${neuron}</div>
          <div class="similar-tokens">${tokensHtml}</div>
        </div>
      `;
      count++;
    }}
    if (count >= 8) break;
  }}

  if (!similarHtml) similarHtml = '<div class="empty-state">No other neurons found</div>';
  document.getElementById('similar-content').innerHTML = similarHtml;
}}

document.addEventListener('DOMContentLoaded', () => {{
  document.getElementById('layer-input').value = currentLayer;
  document.getElementById('neuron-input').value = currentNeuron;
  updateDisplay();
}});

document.addEventListener('keydown', (e) => {{
  if (e.key === 'Enter' && !e.shiftKey) {{
    findNeuron();
  }} else if (e.key === 'Backspace' && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {{
    e.preventDefault();
    goBack();
  }}
}});
</script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML visualization saved to: {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize cached-corpus activation dumps as HTML")
    ap.add_argument("--input_dir", type=str, required=True, help="Collector output dir containing metadata.json + layer_*_activations.json")
    ap.add_argument("--output_html", type=str, default="neuron_explorer.html", help="Path to write HTML")
    args = ap.parse_args()

    all_data, meta = load_collector_dir(args.input_dir)
    generate_html(all_data, meta, args.output_html)


if __name__ == "__main__":
    main()
