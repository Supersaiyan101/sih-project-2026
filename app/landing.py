"""BhoomiSetu landing page (served by FastAPI at GET /).

A polished static product page in pure Python — no Node, no external assets,
offline-safe. Uses the same brand palette + inline-SVG monogram as the dashboard.
The "Launch Dashboard" CTA links to the Streamlit app on :8501.
"""

NAVY = "#1F4E79"
STEEL = "#4A90A4"
GRN = "#2E7D32"
YEL = "#E8A33D"
RED = "#C62828"
BG = "#F5F7FA"
TEXT = "#1A1A1A"

MONOGRAM = f"""
<svg width="46" height="46" viewBox="0 0 42 42" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <circle cx="21" cy="21" r="20" fill="{NAVY}"/>
  <circle cx="21" cy="21" r="20" fill="none" stroke="{STEEL}" stroke-width="1.5"/>
  <text x="21" y="27" text-anchor="middle" font-family="Arial, Helvetica, sans-serif"
        font-size="16" font-weight="700" fill="#ffffff">BS</text>
</svg>"""

LANDING_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BhoomiSetu — Land Acquisition Delay Early-Warning</title>
<style>
  :root {{ --navy:{NAVY}; --steel:{STEEL}; --grn:{GRN}; --yel:{YEL}; --red:{RED}; --bg:{BG}; --text:{TEXT}; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Segoe UI','Helvetica Neue',Arial,sans-serif; color:var(--text); background:var(--bg); line-height:1.55; }}
  a {{ color:var(--steel); text-decoration:none; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:0 24px; }}

  .topnav {{ background:var(--navy); color:#fff; padding:12px 0; }}
  .topnav .wrap {{ display:flex; align-items:center; gap:14px; }}
  .brand {{ display:flex; align-items:center; gap:12px; font-weight:800; font-size:1.2rem; letter-spacing:.02em; }}
  .brand small {{ font-weight:400; opacity:.8; }}
  .states {{ margin-left:auto; font-size:.82rem; opacity:.9; }}
  .states b {{ background:rgba(255,255,255,.16); padding:2px 10px; border-radius:999px; margin-left:6px; }}

  .hero {{ background:linear-gradient(135deg,{NAVY} 0%, #2C6B9B 55%, {STEEL} 100%); color:#fff; padding:64px 0 56px; }}
  .hero h1 {{ font-size:2.5rem; line-height:1.15; font-weight:800; }}
  .hero .tag {{ font-size:1.2rem; margin-top:10px; opacity:.95; }}
  .hero .sub {{ font-size:.9rem; margin-top:6px; opacity:.78; }}
  .cta {{ display:inline-block; margin-top:26px; background:#fff; color:var(--navy);
         font-weight:800; padding:13px 26px; border-radius:8px; box-shadow:0 4px 14px rgba(0,0,0,.18); }}
  .cta:hover {{ transform:translateY(-1px); }}
  .chips {{ margin-top:34px; display:flex; gap:14px; flex-wrap:wrap; }}
  .chip {{ background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.25);
          border-radius:10px; padding:10px 16px; font-size:.88rem; }}

  section {{ padding:48px 0; }}
  h2 {{ color:var(--navy); font-size:1.5rem; margin-bottom:18px; }}
  .cardgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:18px; }}
  .card {{ background:#fff; border-radius:12px; padding:22px 24px; border-left:4px solid var(--steel);
          box-shadow:0 1px 4px rgba(0,0,0,.08); }}
  .card h3 {{ color:var(--navy); margin-bottom:6px; font-size:1.05rem; }}
  .card p {{ font-size:.92rem; color:#444; }}
  .card .tag2 {{ display:inline-block; font-size:.72rem; font-weight:700; color:#fff;
                 border-radius:999px; padding:2px 10px; margin-bottom:8px; }}
  .red {{ background:var(--red); }} .yel {{ background:var(--yel); }} .grn {{ background:var(--grn); }}

  footer {{ border-top:1px solid #e2e8f0; padding:26px 0 40px; color:#6b7280; font-size:.84rem; }}
  footer b {{ color:var(--navy); }}
</style>
</head>
<body>
  <nav class="topnav">
    <div class="wrap">
      <div class="brand">{MONOGRAM}<span>BhoomiSetu<small> &nbsp;·&nbsp; early warning for land acquisition</small></span></div>
      <div class="states">Live pilot <b>HP</b><b>PB</b><b>UK</b></div>
    </div>
  </nav>

  <section class="hero">
    <div class="wrap">
      <h1>Predicting delays before they cost the project.</h1>
      <div class="tag">An AI early-warning system for land-acquisition delays under the RFCTLARR Act, 2013.</div>
      <div class="sub">Bhoomi = land · Setu = bridge — we bridge land-acquisition data gaps before delays happen.</div>
      <a class="cta" href="http://localhost:8501" target="_blank">Launch Dashboard →</a>
      <div class="chips">
        <div class="chip">🎯 Parcel-level risk scores</div>
        <div class="chip">🧩 Per-stage delay probabilities</div>
        <div class="chip">🔍 Explainable AI (SHAP)</div>
        <div class="chip">🧭 Cold-start for new regions</div>
      </div>
    </div>
  </section>

  <section>
    <div class="wrap">
      <h2>The problem</h2>
      <p style="max-width:820px; font-size:1rem;">
        Land acquisition is the most time-sensitive phase of infrastructure delivery. Delays stem from
        administrative approvals, legal disputes, delayed compensation, incomplete documentation, ownership
        conflicts, and rehabilitation challenges — yet there is <b>no intelligent mechanism to flag a project
        before it falls behind</b>. Monitoring is reactive.
      </p>
      <h2 style="margin-top:44px;">What BhoomiSetu does</h2>
      <div class="cardgrid">
        <div class="card"><span class="tag2 red">Predict</span><h3>Forecast delays early</h3>
          <p>10 ML models score every parcel and project across the 5 RFCTLARR stages, predicting delay
          probability and expected days-overrun before they happen.</p></div>
        <div class="card"><span class="tag2 yel">Prioritize</span><h3>Risk scoring & ranking</h3>
          <p>Every project is RED / YELLOW / GREEN, rolled up from parcels to villages, districts, and states,
          so decision-makers know where to intervene first.</p></div>
        <div class="card"><span class="tag2 grn">Explain & act</span><h3>Explainable + actionable</h3>
          <p>SHAP shows exactly why a parcel is at risk, and rule-based recommendations suggest corrective
          actions to bring it back on track.</p></div>
        <div class="card"><span class="tag2 grn">Scale</span><h3>Cold-start by design</h3>
          <p>Models never see geography — a brand-new district or state is scored instantly by feature
          similarity. Leave-one-state-out validates this (avg drop ~2.5%).</p></div>
        <div class="card"><span class="tag2 yel">Monitor</span><h3>Lifecycle state tracking</h3>
          <p>Compensation and rehabilitation progress evolve through the lifecycle; the system re-scores live
          as states change and retrains as new data arrives.</p></div>
        <div class="card"><span class="tag2 red">Alert</span><h3>Alerts & notifications</h3>
          <p>High-risk projects auto-flag with simulated SMS/email/push, plus a FastAPI endpoint for
          integration with existing government systems.</p></div>
      </div>
    </div>
  </section>

  <footer>
    <div class="wrap">
      <b>BhoomiSetu</b> · Predictive Analytics for Early Detection of Land Acquisition Delays ·
      Built for SIH26017 · Himachal Pradesh · Punjab · Uttarakhand pilot.<br>
      Offline-capable · Pure Python · RFCTLARR-aligned.
    </div>
  </footer>
</body>
</html>
"""
