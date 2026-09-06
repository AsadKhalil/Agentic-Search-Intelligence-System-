#!/usr/bin/env python3
"""Render demo-output.json as a self-contained HTML console.

    make demo && make report        ->  demo-report.html

No network and no dependencies: the page carries its own CSS, and every chart is
drawn with CSS from the numbers in the JSON, so it opens from disk.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import webbrowser
from pathlib import Path
from typing import Any

# Node timings span 0.03 ms to 8_200 ms. A linear bar would render every fast node as
# nothing, so the axis is log10 with decade ticks that say so.
LOG_MIN, LOG_MAX = -2.0, 4.0        # 0.01 ms .. 10 s
DECADES = [(-2, "0.01ms"), (-1, "0.1ms"), (0, "1ms"), (1, "10ms"),
           (2, "100ms"), (3, "1s"), (4, "10s")]

STATUS_TONE = {"completed": "ok", "partial": "warn", "failed": "crit"}
FALLBACK_NODES = {"fallback_plan", "fallback_serp", "fallback_analysis"}
LLM_NODES = {"plan_queries", "analyze"}


def e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def log_pct(ms: float) -> float:
    if ms <= 0:
        return 0.0
    pos = (math.log10(ms) - LOG_MIN) / (LOG_MAX - LOG_MIN)
    return max(1.2, min(100.0, pos * 100))


def ms(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.2f} s"
    if value >= 1:
        return f"{value:.0f} ms"
    return f"{value:.2f} ms"


# ---------------------------------------------------------------- components

def scenario_card(s: dict, index: int) -> str:
    tone = STATUS_TONE.get(s["status"], "warn")
    t = s["totals"]
    injected = s["injected_failures"]
    inject_text = ("no failures injected" if not injected else
                   " · ".join(f"{k} ×{'all' if v > 90 else v}" for k, v in injected.items()))
    return f"""
    <article class="card" data-tone="{tone}">
      <header class="card-head">
        <span class="run-no">Run {index}</span>
        <span class="pill pill-{tone}">{e(s['status'])}</span>
      </header>
      <h3>{e(s['label'].split('. ', 1)[-1])}</h3>
      <p class="inject">{e(inject_text)}</p>
      <dl class="figures">
        <div><dt>API calls</dt><dd>{t['api_calls']}</dd></div>
        <div><dt>Retries</dt><dd>{t['retries']}</dd></div>
        <div><dt>Errors</dt><dd>{len(s['errors'])}</dd></div>
        <div><dt>Wall time</dt><dd>{ms(t['duration_ms'])}</dd></div>
      </dl>
      <p class="path">{' <span class="arrow">&rsaquo;</span> '.join(
          f'<span class="node {"fb" if n in FALLBACK_NODES else ""}">{e(n)}</span>'
          for n in s['path'])}</p>
    </article>"""


def comparison(scenarios: list[dict]) -> str:
    peak = max(max(s["totals"]["api_calls"], s["totals"]["retries"]) for s in scenarios) or 1
    rows = []
    for i, s in enumerate(scenarios, 1):
        t = s["totals"]
        rows.append(f"""
        <tr>
          <th scope="row"><span class="run-no">Run {i}</span> {e(s['label'].split('. ', 1)[-1])}</th>
          <td class="barcell">
            <span class="bar bar-calls" style="width:{t['api_calls'] / peak * 100:.1f}%"></span>
            <span class="barval">{t['api_calls']}</span>
          </td>
          <td class="barcell">
            <span class="bar bar-retry" style="width:{t['retries'] / peak * 100:.1f}%"></span>
            <span class="barval">{t['retries']}</span>
          </td>
          <td class="num">{t['llm_tokens'] or '&mdash;'}</td>
          <td class="num">{ms(t['duration_ms'])}</td>
        </tr>""")
    return f"""
    <table class="grid">
      <caption>Every run reached <code>report</code>. What changed was the cost of getting there.</caption>
      <thead><tr><th scope="col">Run</th><th scope="col">Provider calls</th>
        <th scope="col">Retries</th><th scope="col">LLM tokens</th>
        <th scope="col">Wall time</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def node_timeline(s: dict) -> str:
    ticks = "".join(
        f'<span class="tick" style="left:{(d - LOG_MIN) / (LOG_MAX - LOG_MIN) * 100:.2f}%">'
        f'<i></i><b>{label}</b></span>'
        for d, label in DECADES)
    rows = []
    for ev in s["node_events"]:
        klass = "fb" if ev["node"] in FALLBACK_NODES else ("llm" if ev["node"] in LLM_NODES else "")
        note = []
        if ev["retries"]:
            note.append(f"{ev['retries']} retries")
        if ev["api_calls"]:
            note.append(f"{ev['api_calls']} calls")
        if ev["tokens"]:
            note.append(f"{ev['tokens']} tok")
        rows.append(f"""
        <div class="tl-row">
          <span class="tl-name {klass}">{e(ev['node'])}{'' if ev['ok'] else ' <i class="fail">failed</i>'}</span>
          <span class="tl-track">
            <span class="tl-bar {klass}" style="width:{log_pct(ev['duration_ms']):.2f}%"></span>
          </span>
          <span class="tl-ms">{ms(ev['duration_ms'])}</span>
          <span class="tl-note">{e(' · '.join(note))}</span>
        </div>""")
    return f"""
      <div class="timeline">
        <div class="axis">{ticks}</div>
        {''.join(rows)}
        <p class="axis-note">Logarithmic scale &mdash; node durations span four orders of
        magnitude, so a linear axis would render every deterministic node as nothing.</p>
      </div>"""


def error_block(s: dict, max_attempts: int) -> str:
    if not s["errors"]:
        return ('<p class="clean">No errors recorded. Every planned call returned a '
                'usable payload.</p>')
    rows = []
    for err in s["errors"]:
        blocks = "".join(
            f'<i class="att {"spent" if n < err["attempts"] else "unused"}"></i>'
            for n in range(max_attempts))
        rows.append(f"""
        <tr>
          <td><code>{e(err['tool'] or err['node'])}</code></td>
          <td class="qk">{''.join(f'<code class="q">{e(q)}</code>' for q in err['query_keys']) or '<span class="dim">not attributable</span>'}</td>
          <td><span class="pill pill-warn">{e(err['classification'] or err['kind'])}</span>
              <code class="code">{e(err['status_code'] or '&mdash;')}</code></td>
          <td class="attempts">{blocks}<span class="barval">{err['attempts']}/{max_attempts}</span></td>
          <td class="msg">{e(err['message'])}</td>
        </tr>""")
    return f"""
      <table class="grid errors">
        <caption>Simulated failures. <code>50000 Internal Error</code> is classified
        <b>retryable</b>, so each call spent its full budget before the graph moved on.</caption>
        <thead><tr><th scope="col">Tool</th><th scope="col">Queries it cost</th>
          <th scope="col">Class</th><th scope="col">Attempts</th>
          <th scope="col">Provider message</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>"""


def visibility_bar(v: dict) -> str:
    total = sum(v.values()) or 1
    seg = []
    for key, label, tone in (("visible", "visible", "ok"),
                             ("not_visible", "not visible", "crit"),
                             ("unknown", "unknown", "dim")):
        if v[key]:
            seg.append(f'<span class="seg seg-{tone}" style="width:{v[key] / total * 100:.1f}%">'
                       f'{v[key]} {label}</span>')
    return f'<div class="stack">{"".join(seg)}</div>'


def query_table(report: dict) -> str:
    if not report["queries"]:
        return '<p class="clean">No queries reached the table.</p>'
    ticks = "".join(f'<span style="left:{p}%"><i></i><b>{p / 100:.2f}</b></span>'
                    for p in (0, 25, 50, 75, 100))
    rows = []
    for q in report["queries"]:
        if q["domain_visible"] is True:
            vis = f'<span class="pill pill-ok">rank {q["visibility_position"] or "?"}</span>'
        elif q["domain_visible"] is False:
            vis = '<span class="pill pill-crit">absent</span>'
        else:
            vis = '<span class="pill pill-dim">unknown</span>'
        stale = "" if q["retrieval_status"] == "ok" else f' <i class="fail">{e(q["retrieval_status"])}</i>'
        rows.append(f"""
        <tr>
          <th scope="row"><code class="q">{e(q['query_text'])}</code>{stale}</th>
          <td>{vis}</td>
          <td class="num">{q['search_volume'] if q['search_volume'] is not None else '&mdash;'}</td>
          <td class="num">{q['competitive_difficulty'] if q['competitive_difficulty'] is not None else '&mdash;'}</td>
          <td>{'<span class="chip">AI Overview</span>' if q['ai_overview_mentioned'] else ''}
              {'<span class="chip">ChatGPT</span>' if q['chatgpt_mentioned'] else ''}</td>
          <td class="scorecell">
            <span class="score" style="width:{q['opportunity_score'] * 100:.1f}%"></span>
            <span class="barval">{q['opportunity_score']:.3f}</span>
          </td>
        </tr>""")
    return f"""
      <table class="grid queries">
        <thead><tr><th scope="col">Query</th><th scope="col">Organic</th>
          <th scope="col">Volume</th><th scope="col">Difficulty</th>
          <th scope="col">Also cited in</th>
          <th scope="col" class="scorehead">Opportunity<span class="axis-x">{ticks}</span></th>
        </tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>"""


def findings(report: dict) -> str:
    if not report["insights"] and not report["recommendations"]:
        return ""
    ins = "".join(
        f'<li><code class="q">{e(i["query_key"])}</code><p>{e(i["rationale"])}</p></li>'
        for i in report["insights"])
    recs = "".join(f"""
      <li>
        <header><span class="pill pill-{'crit' if r['priority'] == 'high' else 'warn'}">{e(r['priority'])}</span>
          <b>{e(r['title'])}</b></header>
        <p class="kind">{e(r['content_type'])} &middot;
           {' '.join(f'<code class="q">{e(k)}</code>' for k in r['target_keywords'])}</p>
        <p>{e(r['rationale'])}</p>
      </li>""" for r in report["recommendations"])
    by = report["analysis_generated_by"]
    tag = ("written by the model" if by == "llm"
           else "templated &mdash; the model was unavailable, so only the scores are reported")
    blocks = ""
    if ins:
        blocks += f'<div><h4>Insights <span class="dim">({tag})</span></h4><ol class="findings">{ins}</ol></div>'
    if recs:
        blocks += f'<div><h4>Recommended actions</h4><ol class="findings recs">{recs}</ol></div>'
    else:
        blocks += ('<div><h4>Recommended actions</h4><p class="clean">None. With no query '
                   'measured there is nothing to recommend &mdash; the deterministic fallback '
                   'declines to invent one.</p></div>')
    return f'<div class="findings-grid">{blocks}</div>'


# ---------------------------------------------------------------- page

def render(data: dict) -> str:
    cfg = data["config"]
    scenarios = data["scenarios"]
    max_attempts = cfg["retry"]["max_attempts"]
    generated = data["generated_at"][:19].replace("T", " ") + " UTC"
    totals = {k: sum(s["totals"][k] for s in scenarios)
              for k in ("api_calls", "retries", "llm_tokens")}

    sections = []
    for i, s in enumerate(scenarios, 1):
        report = s["report"]
        tone = STATUS_TONE.get(s["status"], "warn")
        sections.append(f"""
      <section class="run" id="run-{i}">
        <header class="run-head">
          <h2><span class="run-no">Run {i}</span> {e(s['label'].split('. ', 1)[-1])}</h2>
          <div class="run-meta">
            <span class="pill pill-{tone}">{e(s['status'])}</span>
            {'<span class="pill pill-warn">degraded</span>' if s['degraded'] else ''}
            <code class="cid" title="correlation id">{e(s['correlation_id'][:16])}&hellip;</code>
          </div>
        </header>
        <p class="lede">{e(report['summary'])}</p>

        <h3>Where the time went</h3>
        {node_timeline(s)}

        <h3>Failures</h3>
        {error_block(s, max_attempts)}

        <h3>Search visibility</h3>
        {visibility_bar(report['visibility'])}
        {query_table(report)}

        {findings(report)}
      </section>""")

    return f"""<title>Pipeline Run Console</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root {{
  --ground:#f5f8f8; --surface:#ffffff; --sunk:#eef3f3; --line:#d5e0e0;
  --ink:#10191c; --ink-2:#3d5056; --muted:#66787e;
  --accent:#0d7d76; --accent-soft:#d4e8e6;
  --ok:#2f7d4f; --ok-soft:#d8ecdf;
  --warn:#a86a12; --warn-soft:#f7e7cd;
  --crit:#b23a2f; --crit-soft:#f7dcd9;
  --radius:6px;
  --sans:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}}
@media (prefers-color-scheme:dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#0d1416; --surface:#141d20; --sunk:#101a1c; --line:#26363a;
    --ink:#e4eded; --ink-2:#a9bfc3; --muted:#7d9298;
    --accent:#3fb3a8; --accent-soft:#12312f;
    --ok:#5cbd80; --ok-soft:#12301f;
    --warn:#d79b3f; --warn-soft:#33260f;
    --crit:#e0705f; --crit-soft:#361a16;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#0d1416; --surface:#141d20; --sunk:#101a1c; --line:#26363a;
  --ink:#e4eded; --ink-2:#a9bfc3; --muted:#7d9298;
  --accent:#3fb3a8; --accent-soft:#12312f;
  --ok:#5cbd80; --ok-soft:#12301f;
  --warn:#d79b3f; --warn-soft:#33260f;
  --crit:#e0705f; --crit-soft:#361a16;
}}
* {{ box-sizing:border-box; }}
body {{ background:var(--ground); color:var(--ink); font-family:var(--sans);
  line-height:1.55; margin:0; padding:0 20px 72px; -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1080px; margin:0 auto; }}
h1,h2,h3,h4 {{ text-wrap:balance; margin:0; font-weight:600; letter-spacing:-0.012em; }}
h1 {{ font-size:1.95rem; letter-spacing:-0.02em; }}
h2 {{ font-size:1.3rem; }}
h3 {{ font-size:0.78rem; text-transform:uppercase; letter-spacing:0.09em;
  color:var(--muted); font-weight:600; margin:34px 0 12px; }}
h4 {{ font-size:0.95rem; margin-bottom:10px; }}
code, .num, .barval, dd {{ font-family:var(--mono); font-variant-numeric:tabular-nums; }}
p {{ margin:0 0 10px; }}
.dim {{ color:var(--muted); }}

/* header */
header.page {{ padding:44px 0 26px; border-bottom:2px solid var(--ink); margin-bottom:26px; }}
.eyebrow {{ font-family:var(--mono); font-size:0.72rem; letter-spacing:0.16em;
  text-transform:uppercase; color:var(--accent); margin-bottom:10px; }}
.sub {{ color:var(--ink-2); max-width:62ch; margin-top:8px; }}
.cfg {{ display:flex; flex-wrap:wrap; gap:7px; margin-top:18px; }}
.cfg code {{ background:var(--sunk); border:1px solid var(--line); border-radius:var(--radius);
  padding:4px 9px; font-size:0.76rem; color:var(--ink-2); }}
.cfg code b {{ color:var(--ink); font-weight:600; }}

/* cards */
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(268px,1fr)); gap:14px; }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:var(--radius);
  padding:16px 17px; display:flex; flex-direction:column; gap:9px; border-top:3px solid var(--muted); }}
.card[data-tone="ok"] {{ border-top-color:var(--ok); }}
.card[data-tone="warn"] {{ border-top-color:var(--warn); }}
.card[data-tone="crit"] {{ border-top-color:var(--crit); }}
.card h3 {{ margin:0; font-size:1.02rem; text-transform:none; letter-spacing:-0.01em;
  color:var(--ink); font-weight:600; }}
.card-head {{ display:flex; justify-content:space-between; align-items:center; }}
.run-no {{ font-family:var(--mono); font-size:0.7rem; letter-spacing:0.08em;
  text-transform:uppercase; color:var(--muted); }}
.inject {{ font-family:var(--mono); font-size:0.76rem; color:var(--muted); margin:0; }}
.figures {{ display:grid; grid-template-columns:1fr 1fr; gap:8px 14px; margin:4px 0 0; }}
.figures div {{ display:flex; justify-content:space-between; align-items:baseline;
  border-bottom:1px dotted var(--line); padding-bottom:3px; }}
.figures dt {{ font-size:0.76rem; color:var(--muted); }}
.figures dd {{ margin:0; font-size:1.02rem; font-weight:600; }}
.path {{ font-family:var(--mono); font-size:0.71rem; color:var(--ink-2); margin:2px 0 0;
  line-height:1.9; }}
.path .node {{ background:var(--sunk); border-radius:3px; padding:2px 5px; }}
.path .node.fb {{ background:var(--warn-soft); color:var(--warn); }}
.path .arrow {{ color:var(--muted); }}

/* pills + chips */
.pill {{ display:inline-block; font-family:var(--mono); font-size:0.68rem; font-weight:600;
  letter-spacing:0.05em; text-transform:uppercase; padding:2px 8px; border-radius:20px; }}
.pill-ok {{ background:var(--ok-soft); color:var(--ok); }}
.pill-warn {{ background:var(--warn-soft); color:var(--warn); }}
.pill-crit {{ background:var(--crit-soft); color:var(--crit); }}
.pill-dim {{ background:var(--sunk); color:var(--muted); }}
.chip {{ display:inline-block; font-size:0.68rem; font-family:var(--mono);
  border:1px solid var(--accent); color:var(--accent); border-radius:3px; padding:1px 5px;
  margin-right:4px; }}
code.q {{ background:var(--accent-soft); color:var(--accent); border-radius:3px;
  padding:1px 5px; font-size:0.79rem; }}
code.code, code.cid {{ background:var(--sunk); border-radius:3px; padding:1px 5px;
  font-size:0.76rem; color:var(--ink-2); }}
.fail {{ color:var(--crit); font-style:normal; font-family:var(--mono); font-size:0.7rem; }}

/* tables */
.grid {{ width:100%; border-collapse:collapse; font-size:0.86rem; }}
.grid caption {{ caption-side:top; text-align:left; color:var(--muted); font-size:0.82rem;
  padding-bottom:9px; max-width:70ch; }}
.grid th[scope=col] {{ text-align:left; font-size:0.7rem; text-transform:uppercase;
  letter-spacing:0.07em; color:var(--muted); font-weight:600; padding:0 10px 7px 0;
  border-bottom:1px solid var(--line); }}
.grid td, .grid th[scope=row] {{ padding:9px 10px 9px 0; border-bottom:1px solid var(--line);
  text-align:left; font-weight:400; vertical-align:middle; }}
.grid tbody tr:last-child td, .grid tbody tr:last-child th {{ border-bottom:none; }}
.num {{ text-align:right; }}
.msg {{ font-family:var(--mono); font-size:0.74rem; color:var(--ink-2); }}
.qk code {{ display:inline-block; margin:1px 3px 1px 0; }}
.scroll {{ overflow-x:auto; }}

/* bars */
.barcell {{ position:relative; min-width:120px; }}
.bar {{ display:inline-block; height:11px; border-radius:2px; vertical-align:middle; }}
.bar-calls {{ background:var(--accent); }}
.bar-retry {{ background:var(--warn); }}
.barval {{ font-size:0.76rem; color:var(--ink-2); margin-left:7px; }}
.scorecell {{ min-width:170px; }}
.score {{ display:inline-block; height:11px; background:var(--accent); border-radius:2px;
  vertical-align:middle; }}
.scorehead {{ position:relative; min-width:170px; }}
.axis-x {{ display:block; position:relative; height:12px; margin-top:3px; }}
.axis-x span {{ position:absolute; transform:translateX(-50%); }}
.axis-x i {{ display:block; width:1px; height:3px; background:var(--line); margin:0 auto; }}
.axis-x b {{ font-family:var(--mono); font-size:0.6rem; font-weight:400; color:var(--muted); }}

/* timeline */
.timeline {{ background:var(--surface); border:1px solid var(--line);
  border-radius:var(--radius); padding:16px 18px 12px; }}
.axis {{ position:relative; height:16px; margin:0 0 10px 152px; }}
.axis .tick {{ position:absolute; transform:translateX(-50%); }}
.axis .tick i {{ display:block; width:1px; height:4px; background:var(--line); margin:0 auto; }}
.axis .tick b {{ font-family:var(--mono); font-size:0.6rem; font-weight:400; color:var(--muted); }}
.tl-row {{ display:grid; grid-template-columns:150px 1fr 68px 110px; gap:8px;
  align-items:center; padding:3px 0; }}
.tl-name {{ font-family:var(--mono); font-size:0.76rem; color:var(--ink-2); }}
.tl-name.llm {{ color:var(--accent); }}
.tl-name.fb {{ color:var(--warn); }}
.tl-track {{ background:var(--sunk); border-radius:2px; height:13px; position:relative; }}
.tl-bar {{ position:absolute; inset:0 auto 0 0; background:var(--ink-2); border-radius:2px; }}
.tl-bar.llm {{ background:var(--accent); }}
.tl-bar.fb {{ background:var(--warn); }}
.tl-ms {{ font-family:var(--mono); font-size:0.74rem; text-align:right;
  font-variant-numeric:tabular-nums; }}
.tl-note {{ font-family:var(--mono); font-size:0.68rem; color:var(--muted); }}
.axis-note {{ font-size:0.74rem; color:var(--muted); margin:12px 0 0; max-width:68ch; }}

/* attempts */
.attempts {{ white-space:nowrap; }}
.att {{ display:inline-block; width:9px; height:14px; border-radius:2px; margin-right:2px; }}
.att.spent {{ background:var(--crit); }}
.att.unused {{ background:var(--sunk); border:1px solid var(--line); }}

/* stacked visibility */
.stack {{ display:flex; height:30px; border-radius:var(--radius); overflow:hidden;
  border:1px solid var(--line); margin-bottom:16px; }}
.seg {{ display:flex; align-items:center; justify-content:center; font-family:var(--mono);
  font-size:0.72rem; color:#fff; white-space:nowrap; }}
.seg-ok {{ background:var(--ok); }}
.seg-crit {{ background:var(--crit); }}
.seg-dim {{ background:var(--muted); }}

/* findings */
.findings-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
  gap:22px; margin-top:30px; }}
.findings {{ list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:11px; }}
.findings li {{ background:var(--surface); border:1px solid var(--line);
  border-radius:var(--radius); padding:12px 14px; }}
.findings li p {{ font-size:0.86rem; color:var(--ink-2); margin:6px 0 0; }}
.recs header {{ display:flex; gap:8px; align-items:baseline; }}
.recs .kind {{ font-family:var(--mono); font-size:0.72rem; color:var(--muted); margin:5px 0 0; }}
.clean {{ color:var(--muted); font-size:0.86rem; background:var(--sunk);
  border:1px dashed var(--line); border-radius:var(--radius); padding:11px 13px; }}

/* runs */
.run {{ border-top:1px solid var(--line); padding-top:30px; margin-top:44px; }}
.run-head {{ display:flex; justify-content:space-between; align-items:center;
  flex-wrap:wrap; gap:10px; }}
.run-head h2 {{ display:flex; gap:10px; align-items:baseline; }}
.run-meta {{ display:flex; gap:7px; align-items:center; }}
.lede {{ color:var(--ink-2); max-width:78ch; margin-top:10px; font-size:0.93rem; }}
footer {{ margin-top:56px; padding-top:18px; border-top:1px solid var(--line);
  color:var(--muted); font-size:0.8rem; }}
footer code {{ background:var(--sunk); border-radius:3px; padding:1px 5px; }}
@media (max-width:720px) {{
  .tl-row {{ grid-template-columns:112px 1fr 62px; }}
  .tl-note {{ display:none; }}
  .axis {{ margin-left:114px; }}
}}
</style>

<div class="wrap">
  <header class="page">
    <p class="eyebrow">Agentic Search Intelligence &mdash; pipeline run console</p>
    <h1>Three runs, one question, escalating failure</h1>
    <p class="sub">The same question is put to the pipeline three times against the fixture
    transport. Run&nbsp;1 is healthy. Runs&nbsp;2 and&nbsp;3 <b>inject provider failures on
    purpose</b> &mdash; the <code>50000 Internal Error</code> responses below are simulated,
    not a fault. What the console shows is what the graph does about them.</p>
    <div class="cfg">
      <code>question <b>{e(data['question'])}</b></code>
      <code>brand <b>{e(data['profile']['name'])} &middot; {e(data['profile']['domain'])}</b></code>
      <code>planner <b>{e(cfg['llm_mode'])} / {e(cfg['llm_model'])}</b></code>
      <code>transport <b>{e(cfg['dataforseo_transport'])}</b></code>
      <code>retry <b>{max_attempts} attempts, {cfg['retry']['base_delay_seconds']}s base,
        {cfg['retry']['max_delay_seconds']}s cap, full jitter</b></code>
      <code>generated <b>{e(generated)}</b></code>
    </div>
  </header>

  <h3>At a glance</h3>
  <div class="cards">{''.join(scenario_card(s, i) for i, s in enumerate(scenarios, 1))}</div>

  <h3>Cost of each run</h3>
  <div class="scroll">{comparison(scenarios)}</div>

  {''.join(sections)}

  <footer>
    Across all three runs: <b>{totals['api_calls']}</b> provider calls,
    <b>{totals['retries']}</b> retries, <b>{totals['llm_tokens']}</b> LLM tokens.
    Regenerate with <code>make demo &amp;&amp; make report</code>. Full data in
    <code>demo-output.json</code>; the correlated log stream in
    <code>demo-logs.ndjson</code>.
  </footer>
</div>"""


def build(input_path: str | Path, out_path: str | Path, open_after: bool = False) -> Path:
    """Render input_path to out_path, optionally opening it in the default browser."""
    source, out = Path(input_path), Path(out_path)
    if not source.exists():
        raise SystemExit(f"{source} not found - run `make demo` first")
    out.write_text(render(json.loads(source.read_text())))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    if open_after:
        webbrowser.open(out.resolve().as_uri())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="demo-output.json")
    p.add_argument("--out", default="demo-report.html")
    p.add_argument("--open", action="store_true", dest="open_after",
                   help="open the report in the default browser when it is written")
    args = p.parse_args()
    build(args.input, args.out, args.open_after)


if __name__ == "__main__":
    main()
