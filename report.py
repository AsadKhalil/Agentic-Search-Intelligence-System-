#!/usr/bin/env python3
"""Render demo-output.json as a self-contained HTML report.

    make demo && make report        ->  demo-report.html

Two views. "What we found" answers the question in plain language and needs no knowledge
of the system. "How it ran" carries the engineering detail: node timings, retry budgets,
provider status codes.

Genuinely self-contained: no scripts, no fonts, no images, nothing fetched. Every chart is
CSS sized from a number in the JSON, so the page renders identically from disk, from a
USB stick, and offline.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import webbrowser
from pathlib import Path
from typing import Any

# Node timings span 0.03 ms to 8_200 ms. A linear bar renders every fast node as nothing,
# so the engineering axis is log10 with decade ticks that say so.
LOG_MIN, LOG_MAX = -2.0, 4.0        # 0.01 ms .. 10 s
DECADES = [(-2, "0.01ms"), (-1, "0.1ms"), (0, "1ms"), (1, "10ms"),
           (2, "100ms"), (3, "1s"), (4, "10s")]

STATUS_TONE = {"completed": "ok", "partial": "warn", "failed": "crit"}
FALLBACK_NODES = {"fallback_plan", "fallback_serp", "fallback_analysis"}
LLM_NODES = {"plan_queries", "analyze"}

# The plain view never prints a node name. Each step gets the sentence a reader who has
# never seen the code would use for it.
PLAIN_STEP = {
    "plan_queries": "Work out what to look up",
    "fallback_plan": "Work out what to look up (backup method)",
    "retrieve": "Collect the data",
    "fallback_serp": "Collect the data (second attempt)",
    "normalize": "Tidy the data",
    "analyze": "Interpret it",
    "fallback_analysis": "Interpret it (without the writer)",
    "report": "Write the report",
}

TOOL_LABEL = {
    "google_serp": "Google search results",
    "keyword_metrics": "Search volume and difficulty",
    "chatgpt_response": "ChatGPT's answer",
}


# ---------------------------------------------------------------- formatting

def e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def log_pct(ms_value: float) -> float:
    if ms_value <= 0:
        return 0.0
    pos = (math.log10(ms_value) - LOG_MIN) / (LOG_MAX - LOG_MIN)
    return max(1.2, min(100.0, pos * 100))


def ms(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.1f} seconds"
    if value >= 1:
        return f"{value:.0f} ms"
    return f"{value:.2f} ms"


def ms_short(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.2f} s"
    if value >= 1:
        return f"{value:.0f} ms"
    return f"{value:.2f} ms"


def num(value: Any) -> str:
    return f"{value:,}" if isinstance(value, (int, float)) else "&mdash;"


def plural(n: int, one: str, many: str | None = None) -> str:
    return one if n == 1 else (many or one + "s")


def opportunity_word(score: float) -> str:
    return "High" if score >= 0.6 else "Medium" if score >= 0.35 else "Low"


def difficulty_word(value: float | None) -> str:
    if value is None:
        return "not measured"
    return "easy" if value < 30 else "moderate" if value < 60 else "hard"


def visibility_cell(query: dict) -> str:
    if query["retrieval_status"] == "failed":
        return '<span class="pill pill-dim">not measured</span>'
    if query["domain_visible"] is True:
        pos = query["visibility_position"]
        where = f"position {pos}" if pos else "on page one"
        return f'<span class="pill pill-ok">Yes &mdash; {e(where)}</span>'
    if query["domain_visible"] is False:
        return '<span class="pill pill-crit">No</span>'
    return '<span class="pill pill-dim">not checked</span>'


# ---------------------------------------------------------------- plain view

def kpis(data: dict, primary: dict) -> str:
    report = primary["report"]
    queries = report["queries"]
    measured = [q for q in queries if q["retrieval_status"] != "failed"]
    checked = [q for q in queries if q["domain_visible"] is not None]
    seen = [q for q in checked if q["domain_visible"] is True]
    best = max(measured, key=lambda q: q["opportunity_score"], default=None)
    brand = data["profile"]["name"]

    tiles = [
        ("Searches checked", str(len(queries)), "distinct search phrases"),
        (f"{e(brand)} appears in", f"{len(seen)} of {len(checked)}"
            if checked else "&mdash;", "of the ones ranked far enough to tell"),
        ("Biggest gap", e(best["query_text"]) if best else "&mdash;",
         f"opportunity {opportunity_word(best['opportunity_score']).lower()}"
         if best else "nothing measurable"),
        ("Actions suggested", str(len(report["recommendations"])),
         "pages to write or refresh"),
    ]
    return '<div class="kpis">' + "".join(
        f'<div class="kpi"><dt>{label}</dt><dd>{value}</dd><p>{note}</p></div>'
        for label, value, note in tiles) + "</div>"


def visibility_bar(v: dict, brand: str) -> str:
    total = sum(v.values()) or 1
    segments = [
        ("visible", f"{brand} shows up", "ok"),
        ("not_visible", "does not show up", "crit"),
        ("unknown", "not measured", "dim"),
    ]
    bars = "".join(
        f'<span class="seg seg-{tone}" style="width:{v[key] / total * 100:.1f}%">{v[key]}</span>'
        for key, _label, tone in segments if v[key])
    legend = "".join(
        f'<span class="key"><i class="sw sw-{tone}"></i>{v[key]} {label}</span>'
        for key, label, tone in segments if v[key])
    return f'<div class="stack">{bars}</div><p class="legend">{legend}</p>'


def plain_query_table(report: dict, brand: str) -> str:
    if not report["queries"]:
        return '<p class="note">No searches were recorded for this run.</p>'
    rows = []
    for q in report["queries"]:
        score = q["opportunity_score"]
        failed = q["retrieval_status"] == "failed"
        cited = []
        if q["ai_overview_mentioned"]:
            cited.append('<span class="chip">Google AI Overview</span>')
        if q["chatgpt_mentioned"]:
            cited.append('<span class="chip">ChatGPT</span>')
        difficulty = (f'{q["competitive_difficulty"]:.0f} <span class="dim">'
                      f'({difficulty_word(q["competitive_difficulty"])})</span>'
                      if q["competitive_difficulty"] is not None else "&mdash;")
        rows.append(f"""
        <tr>
          <th scope="row">{e(q['query_text'])}</th>
          <td>{visibility_cell(q)}</td>
          <td class="num">{num(q['search_volume'])}</td>
          <td class="num">{difficulty}</td>
          <td>{''.join(cited) or '<span class="dim">&mdash;</span>'}</td>
          <td class="oppcell">
            <span class="oppbar"><i style="width:{score * 100:.1f}%"></i></span>
            <span class="opplabel{' faded' if failed else ''}">
              {opportunity_word(score) if not failed else 'unknown'}</span>
          </td>
        </tr>""")
    return f"""
      <div class="scroll"><table class="grid">
        <thead><tr>
          <th scope="col">Search phrase</th>
          <th scope="col">Does {e(brand)} show up?</th>
          <th scope="col" class="num">Searches / month</th>
          <th scope="col" class="num">Difficulty (0&ndash;100)</th>
          <th scope="col">Also cited by</th>
          <th scope="col">Opportunity</th>
        </tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table></div>
      <p class="note">Opportunity weighs how many people search for the phrase, how hard
      it is to rank for, and how visible {e(brand)} already is. Higher means more to gain
      from working on it.</p>"""


def actions(report: dict, brand: str) -> str:
    recs = report["recommendations"]
    if not recs:
        return ('<p class="note">No actions. Nothing was measured in this run, and the '
                'system will not invent a recommendation it has no evidence for.</p>')
    cards = "".join(f"""
      <li data-priority="{e(r['priority'])}">
        <header><span class="pill pill-{'crit' if r['priority'] == 'high' else 'warn'}">{e(r['priority'])} priority</span></header>
        <h4>{e(r['title'])}</h4>
        <p class="kind">{e(r['content_type'])}</p>
        <p>{e(r['rationale'])}</p>
        <p class="targets">Targets: {' '.join(f'<code>{e(k)}</code>' for k in r['target_keywords'])}</p>
      </li>""" for r in recs)
    return f'<ol class="actions">{cards}</ol>'


def insights_list(report: dict) -> str:
    if not report["insights"]:
        return '<p class="note">Nothing was measured, so there is nothing to read.</p>'
    return '<ul class="insights">' + "".join(
        f'<li><b>{e(i["query_text"])}</b><p>{e(i["rationale"])}</p></li>'
        for i in report["insights"]) + "</ul>"


def plain_result(s: dict, primary: dict) -> tuple[str, str, str]:
    """Tone, verdict, and one plain paragraph on what this run actually delivered."""
    totals, errors = s["totals"], s["errors"]
    if not errors and not totals["retries"]:
        return ("ok", "Full answer",
                f"All {totals['api_calls']} lookups answered first time.")
    if not errors:
        same = (" The numbers came out identical to the healthy run."
                if s["report"]["queries"] == primary["report"]["queries"] else "")
        return ("ok", "Full answer",
                f"{totals['retries']} {plural(totals['retries'], 'lookup')} failed and "
                f"{plural(totals['retries'], 'was', 'were')} retried automatically until "
                f"{plural(totals['retries'], 'it', 'they')} succeeded.{same} The whole "
                f"detour cost {ms(totals['duration_ms'])}.")
    unknown = s["report"]["visibility"]["unknown"]
    return ("crit", "Partial answer",
            f"Every lookup failed. After {totals['retries']} retries the system stopped "
            f"trying, wrote the report anyway, and marked all {unknown} searches "
            f"&ldquo;not measured&rdquo; rather than filling in a guess.")


def cause_of(s: dict) -> str:
    injected = s["injected_failures"] or {}
    if not injected:
        return "Nothing was broken"
    if all(v > 90 for v in injected.values()):
        return f"All {len(injected)} data sources switched off"
    name = ", ".join(TOOL_LABEL.get(k, k) for k in injected)
    return f"{name} made to fail {max(injected.values())} times in a row"


def reliability(scenarios: list[dict], primary: dict) -> str:
    cards = []
    for i, s in enumerate(scenarios, 1):
        tone, verdict, body = plain_result(s, primary)
        steps = " <span class='arrow'>&rsaquo;</span> ".join(
            f'<span class="step{" swap" if n in FALLBACK_NODES else ""}">'
            f'{e(PLAIN_STEP.get(n, n))}</span>' for n in s["path"])
        cards.append(f"""
        <article class="rel" data-tone="{tone}">
          <header><span class="run-no">Test {i}</span>
            <span class="pill pill-{tone}">{verdict}</span></header>
          <h4>{e(cause_of(s))}</h4>
          <p>{body}</p>
          <p class="steps">{steps}</p>
        </article>""")
    return f'<div class="rel-grid">{"".join(cards)}</div>'


# ---------------------------------------------------------------- technical view

def comparison(scenarios: list[dict]) -> str:
    peak = max(max(s["totals"]["api_calls"], s["totals"]["retries"])
               for s in scenarios) or 1
    rows = []
    for i, s in enumerate(scenarios, 1):
        t = s["totals"]
        rows.append(f"""
        <tr>
          <th scope="row"><span class="run-no">Run {i}</span>
            {e(s['label'].split('. ', 1)[-1])}</th>
          <td class="barcell">
            <span class="bar bar-calls" style="width:{t['api_calls'] / peak * 110:.0f}px"></span>
            <span class="barval">{t['api_calls']}</span></td>
          <td class="barcell">
            <span class="bar bar-retry" style="width:{t['retries'] / peak * 110:.0f}px"></span>
            <span class="barval">{t['retries']}</span></td>
          <td class="num">{t['llm_tokens'] or '&mdash;'}</td>
          <td class="num">{ms_short(t['duration_ms'])}</td>
        </tr>""")
    return f"""
      <div class="scroll"><table class="grid">
        <caption>All three runs reached <code>report</code>. What changed was the cost of
        getting there.</caption>
        <thead><tr><th scope="col">Run</th><th scope="col">Provider calls</th>
          <th scope="col">Retries</th><th scope="col">LLM tokens</th>
          <th scope="col" class="num">Wall time</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table></div>"""


def planned_calls(s: dict) -> str:
    skip = {"location_code", "language_code", "device", "max_output_tokens"}
    rows = []
    for call in s["planned_calls"]:
        args = []
        for key, value in call["args"].items():
            if key in skip:
                continue
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            text = str(value)
            args.append(f'{key}=<b>{e(text[:58] + "…" if len(text) > 58 else text)}</b>')
        rows.append(f'<li><code class="tool">{e(call["tool"])}</code>'
                    f'<span class="args">{" · ".join(args)}</span></li>')
    return f'<ul class="calls">{"".join(rows)}</ul>'


def node_timeline(s: dict) -> str:
    ticks = "".join(
        f'<span class="tick" style="left:{(d - LOG_MIN) / (LOG_MAX - LOG_MIN) * 100:.2f}%">'
        f'<i></i><b>{label}</b></span>' for d, label in DECADES)
    rows = []
    for ev in s["node_events"]:
        klass = ("fb" if ev["node"] in FALLBACK_NODES
                 else "llm" if ev["node"] in LLM_NODES else "")
        note = []
        if ev["retries"]:
            note.append(f"{ev['retries']} retries")
        if ev["api_calls"]:
            note.append(f"{ev['api_calls']} calls")
        if ev["tokens"]:
            note.append(f"{ev['tokens']} tok")
        fail = "" if ev["ok"] else ' <i class="fail">failed</i>'
        rows.append(f"""
        <div class="tl-row">
          <span class="tl-name {klass}">{e(ev['node'])}{fail}</span>
          <span class="tl-track">
            <span class="tl-bar {klass}" style="width:{log_pct(ev['duration_ms']):.2f}%"></span>
          </span>
          <span class="tl-ms">{ms_short(ev['duration_ms'])}</span>
          <span class="tl-note">{e(' · '.join(note))}</span>
        </div>""")
    return f"""
      <div class="timeline">
        <div class="axis">{ticks}</div>
        {''.join(rows)}
        <p class="note">Logarithmic scale &mdash; node durations span four orders of
        magnitude, so a linear axis would render every deterministic node as nothing.</p>
      </div>"""


def error_block(s: dict, max_attempts: int) -> str:
    if not s["errors"]:
        return ('<p class="note">No errors recorded. Every planned call returned a usable '
                'payload.</p>')
    rows = []
    for err in s["errors"]:
        blocks = "".join(
            f'<i class="att {"spent" if n < err["attempts"] else "unused"}"></i>'
            for n in range(max_attempts))
        keys = "".join(f'<code class="q">{e(q)}</code>' for q in err["query_keys"])
        rows.append(f"""
        <tr>
          <td><code class="tool">{e(err['tool'] or err['node'])}</code></td>
          <td class="qk">{keys or '<span class="dim">not attributable</span>'}</td>
          <td class="cls"><span class="pill pill-warn">{e(err['classification'] or err['kind'])}</span>
            <code>{e(err['status_code'] or '&mdash;')}</code></td>
          <td class="attempts">{blocks}<span class="barval">{err['attempts']}/{max_attempts}</span></td>
          <td class="msg">{e(err['message'])}</td>
        </tr>""")
    return f"""
      <div class="scroll"><table class="grid">
        <caption>Injected failures. <code>50000 Internal Error</code> classifies as
        <b>retryable</b>, so each call spent its full attempt budget before the graph moved
        on.</caption>
        <thead><tr><th scope="col">Tool</th><th scope="col">Queries it cost</th>
          <th scope="col">Class</th><th scope="col">Attempts</th>
          <th scope="col">Provider message</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table></div>"""


def tech_run(s: dict, index: int, max_attempts: int) -> str:
    tone = STATUS_TONE.get(s["status"], "warn")
    degraded = '<span class="pill pill-warn">degraded</span>' if s["degraded"] else ""
    return f"""
      <section class="run">
        <header class="run-head">
          <h3><span class="run-no">Run {index}</span>
            {e(s['label'].split('. ', 1)[-1])}</h3>
          <div class="run-meta">
            <span class="pill pill-{tone}">{e(s['status'])}</span>{degraded}
            <code title="correlation id">{e(s['correlation_id'][:16])}&hellip;</code>
          </div>
        </header>
        <h4>Plan</h4>
        {planned_calls(s)}
        <h4>Where the time went</h4>
        {node_timeline(s)}
        <h4>Failures</h4>
        {error_block(s, max_attempts)}
      </section>"""


# ---------------------------------------------------------------- page

CSS = """
:root {
  --ground:#f4f7f7; --surface:#ffffff; --sunk:#eaf0f0; --line:#d3dfdf;
  --ink:#101a1d; --ink-2:#3c5157; --muted:#66797f;
  --accent:#0d7d76; --accent-soft:#d6e9e7;
  --ok:#2c7a4c; --ok-soft:#d8ecdf;
  --warn:#a4670f; --warn-soft:#f7e7cd;
  --crit:#b03a2e; --crit-soft:#f7dcd9;
  --radius:7px;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
}
@media (prefers-color-scheme:dark) {
  :root:not([data-theme="light"]) {
    --ground:#0d1416; --surface:#141d20; --sunk:#101a1c; --line:#26363a;
    --ink:#e4eded; --ink-2:#a9bfc3; --muted:#7d9298;
    --accent:#3fb3a8; --accent-soft:#12312f;
    --ok:#5cbd80; --ok-soft:#12301f;
    --warn:#d79b3f; --warn-soft:#33260f;
    --crit:#e0705f; --crit-soft:#361a16;
  }
}
:root[data-theme="dark"] {
  --ground:#0d1416; --surface:#141d20; --sunk:#101a1c; --line:#26363a;
  --ink:#e4eded; --ink-2:#a9bfc3; --muted:#7d9298;
  --accent:#3fb3a8; --accent-soft:#12312f;
  --ok:#5cbd80; --ok-soft:#12301f;
  --warn:#d79b3f; --warn-soft:#33260f;
  --crit:#e0705f; --crit-soft:#361a16;
}
* { box-sizing:border-box; }
body { background:var(--ground); color:var(--ink); font-family:var(--sans);
  font-size:15px; line-height:1.6; margin:0; padding:0 20px 72px;
  -webkit-font-smoothing:antialiased; }
.wrap { max-width:1060px; margin:0 auto; }
h1,h2,h3,h4 { text-wrap:balance; margin:0; font-weight:650; letter-spacing:-0.015em; }
h1 { font-size:2.1rem; letter-spacing:-0.028em; line-height:1.15; }
h2 { font-size:1.35rem; margin-bottom:4px; }
h3 { font-size:1.05rem; }
h4 { font-size:0.95rem; }
p { margin:0 0 12px; }
code { font-family:var(--mono); font-size:0.84em; }
.num, .barval, .kpi dd { font-variant-numeric:tabular-nums; }
.dim, .muted { color:var(--muted); }
a { color:var(--accent); }

/* page header */
header.page { padding:46px 0 24px; }
.eyebrow { font-family:var(--mono); font-size:0.7rem; letter-spacing:0.17em;
  text-transform:uppercase; color:var(--accent); margin-bottom:12px; }
.lede { color:var(--ink-2); max-width:64ch; font-size:1.02rem; margin-top:10px; }
.askline { margin-top:20px; padding:14px 16px; background:var(--surface);
  border:1px solid var(--line); border-left:3px solid var(--accent);
  border-radius:0 var(--radius) var(--radius) 0; max-width:64ch; }
.askline b { display:block; font-size:0.7rem; letter-spacing:0.12em; font-weight:650;
  text-transform:uppercase; color:var(--muted); margin-bottom:3px; }
.askline q { font-size:1.05rem; quotes:'\\201C' '\\201D'; }

/* tabs: radios so the page needs no script and keeps keyboard behaviour */
.tabin { position:absolute; top:0; left:0; width:1px; height:1px; opacity:0; }
.tabbar { display:flex; gap:4px; border-bottom:1px solid var(--line); margin-bottom:30px; }
.tabbar label { padding:10px 16px; cursor:pointer; font-weight:600; font-size:0.94rem;
  color:var(--muted); border-bottom:2px solid transparent; margin-bottom:-1px;
  border-radius:var(--radius) var(--radius) 0 0; }
.tabbar label:hover { color:var(--ink); background:var(--sunk); }
.panel { display:none; }
#v-found:checked ~ #p-found, #v-tech:checked ~ #p-tech { display:block; }
#v-found:checked ~ .tabbar label[for="v-found"],
#v-tech:checked ~ .tabbar label[for="v-tech"] {
  color:var(--ink); border-bottom-color:var(--accent); background:none; }
#v-found:focus-visible ~ .tabbar label[for="v-found"],
#v-tech:focus-visible ~ .tabbar label[for="v-tech"] {
  outline:2px solid var(--accent); outline-offset:-2px; }

/* section headings */
.sec { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em;
  color:var(--muted); font-weight:650; margin:40px 0 14px; }
.sec:first-child { margin-top:0; }

/* KPI tiles */
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); border-radius:var(--radius);
  overflow:hidden; }
.kpi { background:var(--surface); padding:16px 18px; }
.kpi dt { font-size:0.74rem; text-transform:uppercase; letter-spacing:0.07em;
  color:var(--muted); font-weight:650; }
.kpi dd { margin:6px 0 2px; font-size:1.3rem; font-weight:650; letter-spacing:-0.02em;
  line-height:1.25; }
.kpi p { margin:0; font-size:0.8rem; color:var(--muted); }

/* stacked visibility */
.stack { display:flex; height:34px; border-radius:var(--radius); overflow:hidden;
  border:1px solid var(--line); }
.seg { display:flex; align-items:center; justify-content:center; color:#fff;
  font-weight:650; font-size:0.85rem; font-variant-numeric:tabular-nums; }
.seg-ok { background:var(--ok); } .seg-crit { background:var(--crit); }
.seg-dim { background:var(--muted); }
.legend { display:flex; flex-wrap:wrap; gap:16px; margin:9px 0 0; font-size:0.84rem;
  color:var(--ink-2); }
.key { display:flex; align-items:center; gap:6px; }
.sw { width:11px; height:11px; border-radius:2px; display:inline-block; }
.sw-ok { background:var(--ok); } .sw-crit { background:var(--crit); }
.sw-dim { background:var(--muted); }

/* tables */
.grid { width:100%; border-collapse:collapse; font-size:0.9rem; }
.grid caption { caption-side:top; text-align:left; color:var(--muted); font-size:0.85rem;
  padding-bottom:10px; max-width:72ch; }
.grid th[scope=col] { text-align:left; font-size:0.71rem; text-transform:uppercase;
  letter-spacing:0.07em; color:var(--muted); font-weight:650; padding:0 12px 8px 0;
  border-bottom:1px solid var(--line); white-space:nowrap; }
.grid td, .grid th[scope=row] { padding:11px 12px 11px 0;
  border-bottom:1px solid var(--line); text-align:left; vertical-align:middle; }
.grid th[scope=row] { font-weight:600; }
.grid tbody tr:last-child td, .grid tbody tr:last-child th { border-bottom:none; }
.grid td.num, .grid th.num { text-align:right; padding-right:20px; }
.scroll { overflow-x:auto; }
.msg { font-family:var(--mono); font-size:0.76rem; color:var(--ink-2); }
.qk code { display:inline-block; margin:1px 3px 1px 0; }

/* opportunity bar */
.oppcell { min-width:150px; }
.oppbar { display:inline-block; width:78px; height:9px; background:var(--sunk);
  border-radius:5px; overflow:hidden; vertical-align:middle; }
.oppbar i { display:block; height:100%; background:var(--accent); border-radius:5px; }
.opplabel { margin-left:9px; font-size:0.85rem; font-weight:600; }
.opplabel.faded { color:var(--muted); font-weight:400; }

/* pills, chips */
.pill { display:inline-block; font-size:0.72rem; font-weight:650; letter-spacing:0.02em;
  padding:3px 9px; border-radius:20px; white-space:nowrap; }
.pill-ok { background:var(--ok-soft); color:var(--ok); }
.pill-warn { background:var(--warn-soft); color:var(--warn); }
.pill-crit { background:var(--crit-soft); color:var(--crit); }
.pill-dim { background:var(--sunk); color:var(--muted); }
.chip { display:inline-block; font-size:0.72rem; border:1px solid var(--accent);
  color:var(--accent); border-radius:4px; padding:1px 6px; margin:1px 4px 1px 0; }
code.q, code.tool { background:var(--accent-soft); color:var(--accent); border-radius:3px;
  padding:1px 6px; }
.fail { color:var(--crit); font-style:normal; font-family:var(--mono); font-size:0.72rem; }
.note { font-size:0.85rem; color:var(--muted); max-width:72ch; margin:12px 0 0; }

/* insights */
.insights { list-style:none; margin:0; padding:0; display:flex; flex-direction:column;
  gap:2px; }
.insights li { padding:11px 0 11px 15px; border-left:2px solid var(--line); }
.insights li:hover { border-left-color:var(--accent); }
.insights b { font-weight:650; }
.insights p { margin:2px 0 0; font-size:0.89rem; color:var(--ink-2); }

/* actions */
.actions { list-style:none; margin:0; padding:0; display:grid; gap:14px;
  grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); }
.actions li { background:var(--surface); border:1px solid var(--line);
  border-radius:var(--radius); padding:16px 18px; border-top:3px solid var(--warn); }
.actions li[data-priority="high"] { border-top-color:var(--crit); }
.actions h4 { margin:10px 0 2px; }
.actions .kind { font-size:0.78rem; text-transform:uppercase; letter-spacing:0.06em;
  color:var(--accent); font-weight:650; margin:0 0 8px; }
.actions p { font-size:0.88rem; color:var(--ink-2); }
.actions .targets { font-size:0.8rem; color:var(--muted); margin-bottom:0; }

/* reliability cards */
.rel-grid { display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); }
.rel { background:var(--surface); border:1px solid var(--line); border-radius:var(--radius);
  padding:16px 18px; border-top:3px solid var(--ok); }
.rel[data-tone="crit"] { border-top-color:var(--crit); }
.rel header { display:flex; justify-content:space-between; align-items:center; gap:8px;
  margin-bottom:10px; }
.rel h4 { margin-bottom:6px; }
.rel p { font-size:0.88rem; color:var(--ink-2); }
.run-no { font-family:var(--mono); font-size:0.7rem; letter-spacing:0.08em;
  text-transform:uppercase; color:var(--muted); }
.steps { font-size:0.75rem !important; line-height:2; margin-bottom:0 !important; }
.step { background:var(--sunk); border-radius:3px; padding:2px 6px; }
.step.swap { background:var(--warn-soft); color:var(--warn); font-weight:600; }
.arrow { color:var(--muted); }

/* technical view */
.cfg { display:flex; flex-wrap:wrap; gap:7px; margin-bottom:6px; }
.cfg code { background:var(--surface); border:1px solid var(--line);
  border-radius:var(--radius); padding:5px 10px; font-size:0.76rem; color:var(--muted); }
.cfg code b { color:var(--ink); font-weight:650; }
.calls { list-style:none; margin:0 0 4px; padding:0; display:flex; flex-direction:column;
  gap:5px; }
.calls li { display:flex; flex-wrap:wrap; gap:9px; align-items:baseline;
  background:var(--surface); border:1px solid var(--line); border-radius:var(--radius);
  padding:8px 12px; }
.args { font-family:var(--mono); font-size:0.74rem; color:var(--muted); }
.args b { color:var(--ink-2); font-weight:500; }
.barcell { display:flex; align-items:center; gap:9px; min-width:150px; }
.bar { flex:none; height:11px; border-radius:2px; }
.bar-calls { background:var(--accent); } .bar-retry { background:var(--warn); }
.barval { font-size:0.78rem; color:var(--ink-2); }
.timeline { background:var(--surface); border:1px solid var(--line);
  border-radius:var(--radius); padding:16px 18px 14px; }
.axis { position:relative; height:16px; margin:0 0 10px 152px; }
.axis .tick { position:absolute; transform:translateX(-50%); }
.axis .tick i { display:block; width:1px; height:4px; background:var(--line); margin:0 auto; }
.axis .tick b { font-family:var(--mono); font-size:0.6rem; font-weight:400; color:var(--muted); }
.tl-row { display:grid; grid-template-columns:150px 1fr 70px 108px; gap:8px;
  align-items:center; padding:3px 0; }
.tl-name { font-family:var(--mono); font-size:0.76rem; color:var(--ink-2); }
.tl-name.llm { color:var(--accent); } .tl-name.fb { color:var(--warn); }
.tl-track { background:var(--sunk); border-radius:2px; height:13px; position:relative; }
.tl-bar { position:absolute; inset:0 auto 0 0; background:var(--ink-2); border-radius:2px; }
.tl-bar.llm { background:var(--accent); } .tl-bar.fb { background:var(--warn); }
.tl-ms { font-family:var(--mono); font-size:0.74rem; text-align:right;
  font-variant-numeric:tabular-nums; }
.tl-note { font-family:var(--mono); font-size:0.68rem; color:var(--muted); }
.cls, .attempts { white-space:nowrap; }
.att { display:inline-block; width:9px; height:14px; border-radius:2px; margin-right:2px; }
.att.spent { background:var(--crit); }
.att.unused { background:var(--sunk); border:1px solid var(--line); }
.run { border-top:1px solid var(--line); padding-top:26px; margin-top:36px; }
.run-head { display:flex; justify-content:space-between; align-items:center;
  flex-wrap:wrap; gap:10px; margin-bottom:20px; }
.run-head h3 { display:flex; gap:10px; align-items:baseline; }
.run-meta { display:flex; gap:7px; align-items:center; }
.run h4 { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em;
  color:var(--muted); margin:26px 0 10px; }

footer { margin-top:56px; padding-top:18px; border-top:1px solid var(--line);
  color:var(--muted); font-size:0.82rem; }
footer code { background:var(--sunk); border-radius:3px; padding:2px 6px; }
@media (max-width:720px) {
  h1 { font-size:1.65rem; }
  .tl-row { grid-template-columns:110px 1fr 62px; }
  .tl-note { display:none; }
  .axis { margin-left:112px; }
}
@media (prefers-reduced-motion:reduce) { * { transition:none !important; } }
"""


def body(data: dict) -> str:
    cfg = data["config"]
    scenarios = data["scenarios"]
    brand = data["profile"]["name"]
    domain = data["profile"]["domain"]
    max_attempts = cfg["retry"]["max_attempts"]
    generated = data["generated_at"][:19].replace("T", " ") + " UTC"
    primary = next((s for s in scenarios if s["status"] == "completed"), scenarios[0])
    report = primary["report"]
    totals = {k: sum(s["totals"][k] for s in scenarios)
              for k in ("api_calls", "retries", "llm_tokens")}
    if report["analysis_generated_by"] != "llm":
        written_by = ("produced from the scores alone &mdash; the writing step was "
                      "unavailable, so nothing was invented")
    elif cfg["llm_mode"] == "scripted":
        written_by = ("written by the built-in analyst; this offline demo calls no "
                      "external model")
    else:
        written_by = f"written by {e(cfg['llm_model'])}"

    return f"""
<div class="wrap">
  <header class="page">
    <p class="eyebrow">Search visibility report &middot; {e(brand)}</p>
    <h1>Where {e(brand)} shows up, and where it doesn&rsquo;t</h1>
    <p class="lede">{e(domain)} was checked against
    {report['queries_analysed']} search {plural(report['queries_analysed'], 'phrase')}.
    The same check was then run twice more with parts of the system deliberately broken,
    to see what the answer looks like when a data source stops responding.</p>
    <div class="askline"><b>The question</b><q>{e(data['question'])}</q></div>
  </header>

  <div class="tabs">
    <input class="tabin" type="radio" name="view" id="v-found" checked>
    <input class="tabin" type="radio" name="view" id="v-tech">
    <div class="tabbar">
      <label for="v-found">What we found</label>
      <label for="v-tech">How it ran</label>
    </div>

    <div class="panel" id="p-found">
      <h2 class="sec">The short answer</h2>
      {kpis(data, primary)}

      <h2 class="sec">Visibility across the {report['queries_analysed']} searches</h2>
      {visibility_bar(report['visibility'], brand)}

      <h2 class="sec">Search by search</h2>
      {plain_query_table(report, brand)}

      <h2 class="sec">What this means, search by search</h2>
      {insights_list(report)}

      <h2 class="sec">What to do about it</h2>
      <p class="note" style="margin:0 0 14px">These are {written_by}.</p>
      {actions(report, brand)}

      <h2 class="sec">Does it still work when things break?</h2>
      <p class="note" style="margin:0 0 14px">Three tests, same question. The failures in
      tests 2 and 3 were caused on purpose &mdash; they are not faults in the system, they
      are the point of the exercise.</p>
      {reliability(scenarios, primary)}
    </div>

    <div class="panel" id="p-tech">
      <h2 class="sec">Run configuration</h2>
      <div class="cfg">
        <code>planner <b>{e(cfg['llm_mode'])} / {e(cfg['llm_model'])}</b></code>
        <code>transport <b>{e(cfg['dataforseo_transport'])}</b></code>
        <code>retry <b>{max_attempts} attempts, {cfg['retry']['base_delay_seconds']}s base,
          {cfg['retry']['max_delay_seconds']}s cap, full jitter</b></code>
        <code>timeouts <b>{cfg['timeouts']['http_seconds']}s http,
          {cfg['timeouts']['chatgpt_seconds']}s chatgpt</b></code>
        <code>generated <b>{e(generated)}</b></code>
      </div>
      <p class="note">Both dependencies are pinned: fixture transport and the deterministic
      planner. The three runs therefore plan identically and differ only in which failures
      are injected, which is what makes the comparison below mean anything.</p>

      <h2 class="sec">Cost of each run</h2>
      {comparison(scenarios)}

      {''.join(tech_run(s, i, max_attempts) for i, s in enumerate(scenarios, 1))}
    </div>
  </div>

  <footer>
    Across all three runs: <b>{totals['api_calls']}</b> provider calls,
    <b>{totals['retries']}</b> retries, <b>{totals['llm_tokens']}</b> LLM tokens.
    Regenerate with <code>make demo</code>. Full data in <code>demo-output.json</code>;
    the correlated log stream in <code>demo-logs.ndjson</code>.
  </footer>
</div>"""


def render(data: dict, standalone: bool = True) -> str:
    head = f"<title>Search Visibility Report</title>\n<style>{CSS}</style>"
    page = body(data)
    if not standalone:
        return f"{head}\n{page}"
    return (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'{head}\n</head>\n<body>{page}\n</body>\n</html>\n')


def build(input_path: str | Path, out_path: str | Path, open_after: bool = False,
          standalone: bool = True) -> Path:
    """Render input_path to out_path, optionally opening it in the default browser."""
    source, out = Path(input_path), Path(out_path)
    if not source.exists():
        raise SystemExit(f"{source} not found - run `make demo` first")
    out.write_text(render(json.loads(source.read_text(encoding="utf-8")), standalone),
                   encoding="utf-8")
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
    p.add_argument("--fragment", action="store_true",
                   help="emit title+style+body only, for embedding in a host page")
    args = p.parse_args()
    build(args.input, args.out, args.open_after, standalone=not args.fragment)


if __name__ == "__main__":
    main()
