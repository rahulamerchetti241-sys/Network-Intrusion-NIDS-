"""
CICIDS2017 NIDS — Web Dashboard  (app.py)
==========================================
Run after train_nids.py.
    python app.py  →  http://localhost:5000
"""

import os, json, pickle, glob
from flask import Flask, jsonify, send_file, render_template_string
from flask_cors import CORS
import numpy as np

app  = Flask(__name__)
CORS(app)
BASE = os.path.dirname(os.path.abspath(__file__))
SAVE = os.path.join(BASE, "saved_models")

# ================= LOAD DATA =================
def _j(fname, default):
    p = os.path.join(SAVE, fname)
    return json.load(open(p)) if os.path.isfile(p) else default

METRICS = _j("metrics.json", [])
META    = _j("meta.json", {})

# ================= FIX DATA FORMAT =================
for d in METRICS:
    d["category"] = d.get("type", d.get("category", "Unknown"))
    d["f1_score"] = d.get("f1", d.get("f1_score", 0))
    d["dataset_type"] = d.get("dataset_type", "cleaned")

# ================= LOAD MODELS =================
BUNDLES = {}
for pkl_path in glob.glob(os.path.join(SAVE, "*.pkl")):
    key = os.path.basename(pkl_path).replace(".pkl", "")
    try:
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "model" in obj:
            BUNDLES[key] = obj
    except Exception:
        pass

print(f"[+] {len(METRICS)} results  |  {len(BUNDLES)} bundles")

# ================= API =================
@app.route("/api/metrics")
def api_metrics(): return jsonify(METRICS)

@app.route("/api/meta")
def api_meta(): return jsonify(META)

@app.route("/api/image/<path:key>")
def api_image(key):
    png_map = META.get("png_map", {})
    path    = png_map.get(key)
    if path and os.path.isfile(path):
        return send_file(path, mimetype="image/png")
    return jsonify({"error": "not found"}), 404

@app.route("/")
def index(): return render_template_string(HTML)

# ================= FRONTEND =================
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NIDS · CICIDS2017 Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Nunito:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
/* ── TOKENS ─────────────────────────────────────────────── */
:root {
  --bg:      #f7f5f2;
  --surface: #ffffff;
  --surface2: #faf9f7;
  --border:  #e8e4df;
  --border2: #d4cfc9;

  --ink:     #1e1b18;
  --ink2:    #4a4540;
  --ink3:    #8a8480;

  --sage:    #4a7c59;
  --sage-lt: #e8f0ea;
  --rose:    #c0504a;
  --rose-lt: #faeaea;
  --amber:   #c97a2a;
  --amber-lt:#fdf2e4;
  --indigo:  #4a5da0;
  --indigo-lt:#eaecf7;
  --teal:    #2a8080;

  --radius:  12px;
  --shadow:  0 2px 12px rgba(0,0,0,.06);
  --shadow-lg: 0 8px 32px rgba(0,0,0,.10);

  --font-head: 'Playfair Display', Georgia, serif;
  --font-body: 'Nunito', system-ui, sans-serif;
}

/* ── RESET ─────────────────────────────────────────────── */
*, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }
html { scroll-behavior:smooth; }
body { background: var(--bg); color: var(--ink); font-family: var(--font-body); font-size: 13.5px; line-height: 1.6; min-height: 100vh; }

/* ── SIDEBAR ────────────────────────────────────────────── */
.layout { display:flex; min-height:100vh; }
aside { width: 220px; flex-shrink: 0; background: var(--surface); border-right: 1px solid var(--border); display: flex; flex-direction: column; position: sticky; top: 0; height: 100vh; overflow-y: auto; }
.sidebar-logo { padding: 24px 20px 16px; border-bottom: 1px solid var(--border); }
.sidebar-logo h1 { font-family: var(--font-head); font-size: 20px; color: var(--ink); }
.sidebar-logo p { font-size: 11px; color: var(--ink3); }
nav { padding: 12px; flex: 1; }
.nav-section { font-size: 10px; font-weight: 700; letter-spacing: 1px; color: var(--ink3); text-transform: uppercase; padding: 0 12px; margin: 12px 0 6px; }
.nav-item { display: flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 13px; color: var(--ink2); transition: all .18s; border: none; background: transparent; width: 100%; text-align: left; }
.nav-item:hover { background: var(--surface2); color: var(--ink); }
.nav-item.active { background: var(--sage-lt); color: var(--sage); font-weight: 700; }

/* ── MAIN CONTENT ───────────────────────────────────────── */
.main-content { flex: 1; padding: 24px 32px; max-width: 1200px; margin: 0 auto; overflow-x: hidden; }
.page-header { margin-bottom: 24px; }
.page-header h2 { font-family: var(--font-head); font-size: 24px; color: var(--ink); }
.page-header p { color: var(--ink3); font-size: 13.5px; }

/* ── PANELS ─────────────────────────────────────────────── */
.panel { display:none; }
.panel.active { display:block; animation: fadeUp .3s ease both; }
@keyframes fadeUp { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }

/* ── CARDS ──────────────────────────────────────────────── */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 18px; box-shadow: var(--shadow); }
.card-title { font-size: 11.5px; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; color: var(--ink); margin-bottom: 12px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }

/* ── KPI STRIP ──────────────────────────────────────────── */
.kpi-strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }
.kpi-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; box-shadow: var(--shadow); border-top: 3px solid var(--sage); }
.kpi-card.blue { border-top-color: var(--indigo); }
.kpi-card.amber { border-top-color: var(--amber); }
.kpi-card.rose { border-top-color: var(--rose); }
.kpi-label { font-size: 10.5px; font-weight: 700; letter-spacing: .5px; text-transform: uppercase; color: var(--ink3); margin-bottom: 6px; }
.kpi-value { font-family: var(--font-head); font-size: 20px; font-weight: 700; color: var(--ink); line-height: 1.1; }
.kpi-sub { font-size: 11px; color: var(--ink3); margin-top: 4px; font-weight: 600; }

/* ── GRIDS ──────────────────────────────────────────────── */
.g2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px; }
.g3 { display:grid; grid-template-columns:repeat(3, 1fr); gap:16px; margin-bottom:16px; }
.ga { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:16px; margin-bottom:16px; }
@media(max-width:1000px) { .g3, .kpi-strip { grid-template-columns:1fr 1fr; } }
@media(max-width:768px) { .g2, .g3, .kpi-strip { grid-template-columns:1fr; } }

/* ── TAGS ───────────────────────────────────────────────── */
.tag { display: inline-flex; align-items: center; padding: 2px 8px; border-radius: 20px; font-size: 10px; font-weight: 700; white-space: nowrap; }
.tag-sup { background:var(--indigo-lt); color:var(--indigo); }
.tag-uns { background:var(--amber-lt); color:var(--amber); }
.tag-hyb { background:#e0f2f1; color:#00796b; }
.tag-cln { background:var(--sage-lt); color:var(--sage); }
.tag-raw { background:#ffebee; color:#c62828; }

/* ── TABLES ─────────────────────────────────────────────── */
.tbl-wrap { overflow-x:auto; border-radius:8px; border:1px solid var(--border); }
table { width:100%; border-collapse:collapse; font-size:12px; }
th { padding: 8px 12px; text-align: left; font-size: 10.5px; font-weight: 700; letter-spacing: .5px; text-transform: uppercase; color: var(--ink3); background: var(--surface2); border-bottom: 1px solid var(--border); }
td { padding: 10px 12px; border-bottom: 1px solid var(--border); color: var(--ink2); vertical-align: middle; }
tr:hover td { background: var(--surface2); }

/* ── MINI PIE GRID (Deep Pies) ──────────────────────────── */
.mini-pie-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 12px; margin-bottom: 16px; }
.mini-pie-item { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px; text-align: center; box-shadow: var(--shadow); transition: transform 0.2s; }
.mini-pie-item:hover { transform: translateY(-2px); box-shadow: var(--shadow-lg); border-color: var(--border2); }
.mini-pie-item canvas { max-width: 90px; max-height: 90px; margin: 0 auto 8px; display: block; }
.mini-pie-name { font-size: 12px; font-weight: 800; color: var(--ink); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.mini-pie-acc  { font-size: 13px; font-weight: 800; margin-top: 2px; }
.mini-pie-sub  { font-size: 9px; color: var(--ink3); margin-top: 2px; text-transform: uppercase; letter-spacing: 0.5px;}

/* ── UI ELEMENTS ────────────────────────────────────────── */
.filter-row { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px; }
.filter-pill { padding: 6px 14px; border-radius: 20px; font-size: 11.5px; font-weight: 700; cursor: pointer; border: 1px solid var(--border2); background: var(--surface); color: var(--ink3); transition: all .15s; font-family: var(--font-body); }
.filter-pill.active { background: var(--sage); border-color: var(--sage); color: #fff; }

.mini-bar { height:4px; background:var(--border); border-radius:2px; margin-top:4px; overflow:hidden; min-width:50px; }
.mini-fill { height:100%; border-radius:2px; }
.loader { display:none; text-align:center; padding:60px 20px; }
.loader.active { display:block; }
.spinner { width: 30px; height: 30px; border: 3px solid var(--border2); border-top-color: var(--sage); border-radius: 50%; animation: spin .8s linear infinite; margin: 0 auto 12px; }
@keyframes spin { to{transform:rotate(360deg)} }

/* ── VISUALS ────────────────────────────────────────────── */
.img-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:12px; }
.img-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; box-shadow: var(--shadow); cursor:pointer; }
.img-card img { width:100%; display:block; object-fit:cover; height: 120px; }
.img-card-label { padding: 8px; font-size: 10px; font-weight: 700; text-transform: uppercase; text-align:center; }
</style>
</head>
<body>

<div class="layout">
  <!-- ── SIDEBAR ── -->
  <aside>
    <div class="sidebar-logo">
      <h1>NIDS</h1>
      <p>CICIDS2017 Intelligence</p>
    </div>
    <nav>
      <div class="nav-section">Analysis</div>
      <button class="nav-item active" data-p="dashboard" onclick="go(this)">📊 Dashboard</button>
      <button class="nav-item" data-p="overview" onclick="go(this)">🔍 Overview Table</button>
      <button class="nav-item" data-p="models" onclick="go(this)">🤖 Model Directory</button>
      <div class="nav-section">Insights & Tools</div>
      <button class="nav-item" data-p="piecharts" onclick="go(this)">🥧 Pie Charts</button>
      <button class="nav-item" data-p="visuals" onclick="go(this)">🖼 Visualizations</button>
    </nav>
  </aside>

  <!-- ── MAIN ── -->
  <div class="main-content">

    <!-- ══ DASHBOARD ══ -->
    <div class="panel active" id="panel-dashboard">
      <div class="page-header">
        <h2>Executive Dashboard</h2>
        <p>A refined, compact overview of model performance and categorization.</p>
      </div>
      
      <div class="loader active" id="dash-loader"><div class="spinner"></div><p>Processing metrics...</p></div>
      
      <div id="dash-content" style="display:none">
        
        <!-- 1. KPI STRIP -->
        <div class="kpi-strip" id="kpi-strip"></div>
        
        <!-- 2. PIE CHARTS & RADAR (3 in a row) -->
        <div class="g3">
          <div class="card">
            <div class="card-title">Overall System Accuracy</div>
            <div style="max-width:140px; margin:0 auto">
              <canvas id="ch-overall-pie" height="140"></canvas>
            </div>
            <p style="text-align:center; font-size:10px; color:var(--ink3); margin-top:8px;">Avg across all data</p>
          </div>
          
          <div class="card">
            <div class="card-title">Algorithm Types</div>
            <div style="max-width:140px; margin:0 auto">
              <canvas id="ch-cat-pie" height="140"></canvas>
            </div>
            <p style="text-align:center; font-size:10px; color:var(--ink3); margin-top:8px;">Architecture spread</p>
          </div>
          
          <div class="card">
            <div class="card-title">Radar: First 5 Models</div>
            <div style="max-width:150px; margin:0 auto">
              <canvas id="ch-radar" height="140"></canvas>
            </div>
          </div>
        </div>

        <!-- 3. PIE CHARTS PREVIEW (Mini row) -->
        <div class="card gap" style="padding:16px;">
            <div class="card-title" style="margin-bottom:8px; border-bottom:none;">Top Models Breakdown (Deep Pies)</div>
            <div id="dash-mini-pies" class="mini-pie-grid" style="margin-bottom:0;"></div>
        </div>

        <!-- 4. BAR & SCATTER GRAPHS (Silently limited data) -->
        <div class="g2">
          <div class="card">
            <div class="card-title">Accuracy by Category (RAW data coming soon...)</div>
            <div style="height: 180px;"><canvas id="ch-cat-bar"></canvas></div>
          </div>
          <div class="card">
            <div class="card-title">Precision vs Recall</div>
            <div style="height: 180px;"><canvas id="ch-scatter"></canvas></div>
          </div>
        </div>

        <!-- 5. ROC & CONFUSION MATRIX -->
        <div class="g2">
          <div class="card">
            <div class="card-title">ROC Curves (Simulated Top 3)</div>
            <div style="height: 180px;"><canvas id="ch-roc"></canvas></div>
          </div>
          <div class="card">
            <div class="card-title">Simulated Confusion Matrix (Best Model)</div>
            <div id="conf-matrix" style="display:grid; grid-template-columns:1fr 1fr; gap:10px; text-align:center; height:180px; align-content:center;"></div>
          </div>
        </div>

      </div>
    </div>

    <!-- ══ OVERVIEW TABLE ══ -->
    <div class="panel" id="panel-overview">
      <div class="page-header"><h2>Leaderboard Overview</h2><p>Ranked models from best to worst.</p></div>
      <div class="card">
        <div class="tbl-wrap">
          <table>
            <thead>
              <tr><th>#</th><th>Model</th><th>Type</th><th>Variant</th><th>Data</th>
                  <th>Accuracy</th><th>F1-Score</th><th>Precision</th><th>Recall</th></tr>
            </thead>
            <tbody id="ov-tbody"></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ══ MODELS DIRECTORY ══ -->
    <div class="panel" id="panel-models">
      <div class="page-header"><h2>Model Directory</h2><p>Filter models by architecture category.</p></div>
      <div class="filter-row" id="cat-filter">
        <button class="filter-pill active" data-c="All" onclick="filterCat(this)">All Types</button>
        <button class="filter-pill" data-c="Supervised" onclick="filterCat(this)">Supervised</button>
        <button class="filter-pill" data-c="Unsupervised" onclick="filterCat(this)">Unsupervised</button>
        <button class="filter-pill" data-c="Hybrid" onclick="filterCat(this)">Hybrid (Pipelines)</button>
      </div>
      <div id="model-cards" class="ga"></div>
    </div>

    <!-- ══ PIE CHARTS (DEEP DIVE) ══ -->
    <div class="panel" id="panel-piecharts">
      <div class="page-header"><h2>Pie Charts</h2><p>Accuracy proportions perfectly scaled to ring percentages.</p></div>
      <div class="filter-row" id="pie-ds-filter">
        <button class="filter-pill active" data-ds="cleaned" onclick="setPieDs(this)">Cleaned Dataset</button>
        <button class="filter-pill" data-ds="raw" onclick="setPieDs(this)">Raw Dataset</button>
      </div>
      <div id="pie-indiv" class="mini-pie-grid"></div>
    </div>

    <!-- ══ VISUALIZATIONS ══ -->
    <div class="panel" id="panel-visuals">
      <div class="page-header"><h2>Visualizations</h2></div>
      <div id="img-grid" class="img-grid"></div>
    </div>

  </div>
</div>

<script>
// ── CHART.JS PLUGIN: Center Text ──
Chart.register({
  id: 'centerText',
  beforeDraw: function(chart) {
    if (chart.config.options.elements && chart.config.options.elements.center) {
      const ctx = chart.ctx;
      const cfg = chart.config.options.elements.center;
      ctx.save();
      ctx.font = "bold 14px 'Nunito', sans-serif";
      ctx.textBaseline = "middle";
      ctx.textAlign = "center";
      ctx.fillStyle = cfg.color || '#1e1b18';
      const cX = (chart.chartArea.left + chart.chartArea.right) / 2;
      const cY = (chart.chartArea.top + chart.chartArea.bottom) / 2;
      ctx.fillText(cfg.text, cX, cY);
      ctx.restore();
    }
  }
});

let DATA=[], META={};
let PIE_DS='cleaned';
const CH={};

// Standard Categorical Colors
const CCL = { 
  Supervised: '#4a5da0',   // Indigo
  Unsupervised: '#c97a2a', // Amber
  Hybrid: '#2a8080'        // Teal
};
const PAL = ['#4a5da0', '#c97a2a', '#2a8080', '#c0504a', '#7c3aed', '#4a7c59'];

function mk(id, cfg){
  if(CH[id]){ CH[id].destroy(); delete CH[id]; }
  const el = document.getElementById(id);
  if(!el) return null;
  return (CH[id] = new Chart(el, cfg));
}

// Ensure accuracy is always a fraction (0 - 1) for math formulas
function normalizeMetric(v) {
  if (v == null) return 0;
  return v > 1 ? v / 100 : v; 
}

function avg(arr, key){
  if(!arr.length) return 0;
  return arr.reduce((s, m) => s + normalizeMetric(m[key]), 0) / arr.length;
}

// ── SMART CATEGORIZATION ENGINE ──
function assignCategory(modelName) {
  const name = String(modelName).toUpperCase();
  if (name.includes('+')) return 'Hybrid';
  if (name.includes('SVM') || name.includes('ISOLATION') || name.includes('LOF') || name.includes('KMEANS') || name.includes('OUTLIER')) return 'Unsupervised';
  return 'Supervised';
}

// ── BOOT ──
async function boot(){
  try {
    const [dr,mr] = await Promise.all([ fetch('/api/metrics').then(r=>r.json()), fetch('/api/meta').then(r=>r.json()) ]);
    DATA=dr; META=mr;
  } catch(e) {
    console.error("Data load error:", e);
    if(DATA.length===0) return;
  }
  
  // Clean, Normalize, and Categorize Data
  DATA.forEach(d => {
    d.model = d.model || 'Unknown';
    d.category = assignCategory(d.model); 
    d.accuracy = normalizeMetric(d.accuracy);
    d.f1_score = normalizeMetric(d.f1_score);
    d.precision = normalizeMetric(d.precision);
    d.recall = normalizeMetric(d.recall);
  });

  document.getElementById('dash-loader').style.display='none';
  document.getElementById('dash-content').style.display='block';

  buildDashboard();
  buildOverview();
  buildModelCards();
  renderPies();
  buildVisuals();
}

// ── DASHBOARD ──
function buildDashboard(){
  // Sort best to worst globally for accurate selection
  const sortedAll = [...DATA].sort((a,b)=>b.accuracy-a.accuracy);
  const bestOverall = sortedAll[0] || {model:'None', accuracy:0, category:'Unknown'};
  
  const cl = sortedAll.filter(d => d.dataset_type === 'cleaned');
  const ra = sortedAll.filter(d => d.dataset_type === 'raw');
  const overallAvg = avg(DATA, 'accuracy');

  // KPI STRIP (Best Model prominent)
  document.getElementById('kpi-strip').innerHTML = `
    <div class="kpi-card blue">
      <div class="kpi-label">🏆 #1 Best Model</div>
      <div class="kpi-value" style="font-size:18px">${bestOverall.model}</div>
      <div class="kpi-sub">Acc: ${(bestOverall.accuracy*100).toFixed(2)}% | ${bestOverall.category}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Average Accuracy</div>
      <div class="kpi-value" style="color:var(--sage)">${(overallAvg*100).toFixed(1)}%</div>
      <div class="kpi-sub">System-wide</div>
    </div>
    <div class="kpi-card amber">
      <div class="kpi-label">Models Ranked</div>
      <div class="kpi-value">${DATA.length}</div>
      <div class="kpi-sub">Ready for analysis</div>
    </div>
    <div class="kpi-card rose">
      <div class="kpi-label">Top Category</div>
      <div class="kpi-value" style="font-size:18px">${bestOverall.category}</div>
      <div class="kpi-sub">Leads leaderboard</div>
    </div>`;

  // OVERALL ACCURACY PIE (Guaranteed Math: acc vs 1-acc)
  mk('ch-overall-pie', {
    type: 'doughnut',
    data: {
      labels: ['Accuracy', 'Error Rate'],
      datasets: [{ data: [overallAvg, 1 - overallAvg], backgroundColor: ['#4a7c59', '#e8e4df'], borderWidth: 0 }]
    },
    options: {
      responsive: true, cutout: '76%',
      elements: { center: { text: (overallAvg*100).toFixed(1)+'%', color: '#4a7c59' } },
      plugins: { legend: { display:false }, tooltip: { callbacks:{label:c=>` ${(c.parsed*100).toFixed(1)}%`} } }
    }
  });

  // CATEGORY PIE
  const cats = ['Supervised', 'Unsupervised', 'Hybrid'];
  const catCounts = cats.map(c => DATA.filter(d=>d.category===c).length);
  mk('ch-cat-pie', {
    type: 'doughnut',
    data: {
      labels: cats,
      datasets: [{ data: catCounts, backgroundColor: cats.map(c => CCL[c]), borderWidth: 0 }]
    },
    options: {
      responsive: true, cutout: '65%',
      plugins: { legend: { position:'right', labels:{boxWidth:8, font:{size:9}} } }
    }
  });

  // RADAR (Strictly first 5 models)
  const first5 = sortedAll.slice(0,5); 
  mk('ch-radar', {
    type: 'radar',
    data: {
      labels: ['Acc','Prec','Rec','F1'],
      datasets: first5.map((m,i)=>({
        label: m.model,
        data: [m.accuracy, m.precision, m.recall, m.f1_score],
        borderColor: PAL[i%PAL.length], backgroundColor: PAL[i%PAL.length]+'22', pointRadius:0, borderWidth: 1.5
      }))
    },
    options: {
      responsive: true,
      scales: { r: { min:0, max:1, ticks:{display:false} } },
      plugins: { legend: { display:false } } 
    }
  });

  // MINI DEEP PIES PREVIEW (First 6 to fit one row nicely)
  const top6 = cl.slice(0, 6);
  const miniGrid = document.getElementById('dash-mini-pies');
  miniGrid.innerHTML = top6.map((m, i) => {
    const col = CCL[m.category] || '#4a5da0';
    return `
      <div class="mini-pie-item">
        <canvas id="dash_pd_${i}" width="70" height="70"></canvas>
        <div class="mini-pie-name" title="${m.model}">${m.model}</div>
        <div class="mini-pie-acc" style="color:${col}">${(m.accuracy*100).toFixed(1)}%</div>
      </div>`;
  }).join('');
  
  // Render charts for the preview row
  top6.forEach((m, i) => {
    const col = CCL[m.category] || '#4a5da0';
    mk(`dash_pd_${i}`, {
      type: 'doughnut',
      data: { labels: ['Correct', 'Error'], datasets: [{ data: [m.accuracy, 1-m.accuracy], backgroundColor: [col, '#f0ede9'], borderWidth: 0 }] },
      options: { responsive: true, cutout: '75%', plugins: { legend: { display: false }, tooltip:{enabled:false} } }
    });
  });

  // CATEGORY BAR CHART
  mk('ch-cat-bar', {
    type: 'bar',
    data: {
      labels: cats,
      datasets: [
        { label: 'Cleaned Data', data: cats.map(c => avg(cl.filter(d=>d.category===c), 'accuracy')), backgroundColor: '#4a7c59' },
        { label: 'Raw Data', data: cats.map(c => avg(ra.filter(d=>d.category===c), 'accuracy')), backgroundColor: '#c0504a' }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { y: { min:0, max:1, ticks:{callback:v=>(v*100)+'%'} } },
      plugins: { legend: { position: 'top', labels:{boxWidth:10, font:{size:10}} } }
    }
  });

  // SCATTER (Silently limited to top 15 to avoid clutter)
  const scatterData = cl.slice(0, 15);
  mk('ch-scatter', {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'Models (Top 15)',
        data: scatterData.map(m => ({ x: m.precision, y: m.recall, model: m.model, cat: m.category })),
        backgroundColor: scatterData.map(m => CCL[m.category]),
        pointRadius: 4
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { x: { title:{display:true, text:'Precision', font:{size:10}}, min:0, max:1 }, y: { title:{display:true, text:'Recall', font:{size:10}}, min:0, max:1 } },
      plugins: { legend: {display:false}, tooltip: { callbacks: { label: c => `${c.raw.model}: P=${(c.raw.x*100).toFixed(0)}%, R=${(c.raw.y*100).toFixed(0)}%` } } }
    }
  });

  // ROC CURVES (Simulated Top 3)
  const top3 = sortedAll.slice(0,3);
  mk('ch-roc', {
    type: 'line',
    data: {
      labels: ['0', '0.2', '0.4', '0.6', '0.8', '1.0'],
      datasets: top3.map((m, i) => ({
        label: m.model,
        data: [0, m.recall*0.3, m.recall*0.5, m.recall*0.7, m.recall, 1],
        borderColor: PAL[i%PAL.length],
        backgroundColor: PAL[i%PAL.length]+'33',
        fill: false, tension: 0.3, borderWidth: 2
      }))
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { x: {title:{display:true,text:'False Positive Rate'}}, y: {title:{display:true,text:'True Positive Rate'}, min:0, max:1} },
      plugins: { legend: { position: 'top', labels:{boxWidth:10, font:{size:10}} } }
    }
  });

  // SIMULATED CONFUSION MATRIX
  const acc = bestOverall.accuracy || 0;
  const tp = (acc * 0.9).toFixed(2);
  const tn = (acc * 0.9).toFixed(2);
  const fn = ((1 - acc) / 2).toFixed(2);
  const fp = ((1 - acc) / 2).toFixed(2);
  
  document.getElementById('conf-matrix').innerHTML = `
    <div style="background:var(--sage-lt); padding:10px; border-radius:8px;">
        <div style="font-size:10px;color:var(--sage);font-weight:700;text-transform:uppercase;">True Positive</div>
        <div style="font-size:18px;font-weight:800;color:var(--ink)">${(tp*100).toFixed(0)}%</div>
    </div>
    <div style="background:var(--rose-lt); padding:10px; border-radius:8px;">
        <div style="font-size:10px;color:var(--rose);font-weight:700;text-transform:uppercase;">False Positive</div>
        <div style="font-size:18px;font-weight:800;color:var(--ink)">${(fp*100).toFixed(0)}%</div>
    </div>
    <div style="background:var(--rose-lt); padding:10px; border-radius:8px;">
        <div style="font-size:10px;color:var(--rose);font-weight:700;text-transform:uppercase;">False Negative</div>
        <div style="font-size:18px;font-weight:800;color:var(--ink)">${(fn*100).toFixed(0)}%</div>
    </div>
    <div style="background:var(--sage-lt); padding:10px; border-radius:8px;">
        <div style="font-size:10px;color:var(--sage);font-weight:700;text-transform:uppercase;">True Negative</div>
        <div style="font-size:18px;font-weight:800;color:var(--ink)">${(tn*100).toFixed(0)}%</div>
    </div>`;
}

// ── OVERVIEW TABLE ──
function buildOverview(){
  const sorted = [...DATA].sort((a,b)=>b.accuracy-a.accuracy);
  const tbody = document.getElementById('ov-tbody');
  const ctag = c => c==='Supervised'?'tag-sup':c==='Unsupervised'?'tag-uns':'tag-hyb';
  const pct = v => (v*100).toFixed(2)+'%';

  tbody.innerHTML = sorted.map((m,i)=>{
    const col = CCL[m.category]||'#4a5da0';
    return `<tr>
      <td style="font-weight:700">${i===0?'🥇':i===1?'🥈':i===2?'🥉':i+1}</td>
      <td style="font-weight:700;color:var(--ink)">${m.model}</td>
      <td><span class="tag ${ctag(m.category)}">${m.category}</span></td>
      <td><span class="tag tag-pca">${m.variant||'all'}</span></td>
      <td><span class="tag ${m.dataset_type==='cleaned'?'tag-cln':'tag-raw'}">${m.dataset_type}</span></td>
      <td>
        <div style="font-weight:700;color:${col}">${pct(m.accuracy)}</div>
        <div class="mini-bar"><div class="mini-fill" style="width:${m.accuracy*100}%;background:${col}"></div></div>
      </td>
      <td>${pct(m.f1_score)}</td><td>${pct(m.precision)}</td><td>${pct(m.recall)}</td>
    </tr>`;
  }).join('');
}

// ── MODELS DIRECTORY ──
function buildModelCards(){ renderMC('All'); }
function renderMC(cat){
  const uniqueModels = [...new Set(DATA.map(d=>d.model))];
  const html = uniqueModels.map(name => {
    const rows = DATA.filter(d=>d.model===name);
    const m = rows.find(d=>d.dataset_type==='cleaned') || rows[0];
    if(!m || (cat !== 'All' && m.category !== cat)) return '';
    const col = CCL[m.category] || '#4a5da0';
    return `
      <div class="card" style="border-top: 3px solid ${col}; padding: 14px;">
        <div style="font-size:13px;font-weight:700;color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${m.model}</div>
        <div style="font-size:9px;color:var(--ink3);text-transform:uppercase;margin-bottom:8px;">${m.category}</div>
        <div style="font-size:20px;font-weight:800;color:${col};line-height:1">${(m.accuracy*100).toFixed(1)}%</div>
      </div>`;
  }).join('');
  document.getElementById('model-cards').innerHTML = html || '<div style="grid-column:1/-1;text-align:center;padding:40px;color:#888;">No models match this category.</div>';
}
function filterCat(btn){
  document.querySelectorAll('#cat-filter .filter-pill').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active'); renderMC(btn.dataset.c);
}

// ── FULL DEEP PIES TAB ──
function setPieDs(btn){
  PIE_DS = btn.dataset.ds;
  document.querySelectorAll('#pie-ds-filter .filter-pill').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active'); renderPies();
}

function renderPies(){
  const sec = document.getElementById('pie-indiv');
  sec.innerHTML = '';
  // Sort best to worst so pies look organized
  const rows = DATA.filter(d => d.dataset_type === PIE_DS).sort((a,b)=>b.accuracy-a.accuracy);
  if(!rows.length) { sec.innerHTML='<p>No data available.</p>'; return; }

  // Draw DOM elements for the grid
  sec.innerHTML = rows.map((m, i) => {
    const col = CCL[m.category] || '#4a5da0';
    return `
      <div class="mini-pie-item">
        <canvas id="pd_${i}" width="90" height="90"></canvas>
        <div class="mini-pie-name" title="${m.model}">${m.model}</div>
        <div class="mini-pie-acc" style="color:${col}">${(m.accuracy*100).toFixed(1)}%</div>
        <div class="mini-pie-sub">${m.category}</div>
      </div>`;
  }).join('');

  // Attach Chart.js logic to perfectly match ring percentages
  rows.forEach((m, i) => {
    const col = CCL[m.category] || '#4a5da0';
    const acc = m.accuracy;
    const err = 1 - acc; // Strictly correct mathematical remainder
    
    mk(`pd_${i}`, {
      type: 'doughnut',
      data: { 
        labels: ['Correct', 'Error'], 
        datasets: [{ data: [acc, err], backgroundColor: [col, '#f0ede9'], borderWidth: 0 }] 
      },
      options: { 
        responsive: true, maintainAspectRatio: true, cutout: '76%', 
        plugins: { 
          legend: { display: false }, 
          tooltip: { callbacks: { label: c => ` ${(c.parsed * 100).toFixed(1)}%` } } 
        } 
      }
    });
  });
}

// ── VISUALIZATIONS ──
async function buildVisuals(){
  const png = META.png_map||{};
  const grid = document.getElementById('img-grid');
  if(!Object.keys(png).length){ grid.innerHTML='<p style="color:#888">No images found.</p>'; return; }
  grid.innerHTML = Object.entries(png).map(([key])=>`
    <div class="img-card"><img src="/api/image/${key}"><div class="img-card-label">${key.replace(/_/g,' ')}</div></div>
  `).join('');
}

// ── NAVIGATION ──
function go(btn){
  document.querySelectorAll('.nav-item').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('panel-'+btn.dataset.p).classList.add('active');
}

window.onload = boot;
</script>
</body>
</html>"""

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)