#!/usr/bin/env python3
"""End-to-end API walkthrough with changeable inputs.

    make run                       # in another terminal
    ./walkthrough.py               # defaults
    ./walkthrough.py --name Linear --domain linear.app \
        --industry "issue tracking" --question "Where do we rank for bug tracking?"
    ./walkthrough.py --profile <uuid> --question "..."     # reuse a profile
    ./walkthrough.py --recheck --curl                      # + recheck, show curl

Standard library only, so `python3 walkthrough.py` works without the venv.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
import urllib.error
import urllib.request

BOLD, DIM, RED, GRN, YLW, OFF = "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[0m"
STATUS_COLOUR = {"completed": GRN, "partial": YLW, "failed": RED}

SHOW_CURL = False


def step(text: str) -> None:
    print(f"\n{BOLD}==> {text}{OFF}")


def die(message: str) -> "NoReturn":  # type: ignore[valid-type]
    print(f"{RED}error:{OFF} {message}", file=sys.stderr)
    raise SystemExit(1)


def call(method: str, url: str, body: dict | None = None) -> tuple[int, dict]:
    """One HTTP call, printing the curl equivalent when --curl is on."""
    if SHOW_CURL:
        parts = ["curl", "-s"]
        if method != "GET":
            parts += ["-X", method]
        if body is not None:
            parts += ["-H", "content-type: application/json", "-d", json.dumps(body)]
        parts.append(url)
        print(f"    {DIM}$ {' '.join(shlex.quote(p) for p in parts)}{OFF}")

    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:                       # 4xx / 5xx carry a body
        raw = exc.read()
        try:
            return exc.code, json.loads(raw or b"{}")
        except ValueError:
            return exc.code, {"detail": raw.decode(errors="replace")}
    except urllib.error.URLError as exc:
        die(f"cannot reach {url} — is the server running? ({exc.reason})")


def main() -> None:
    global SHOW_CURL
    p = argparse.ArgumentParser(description="Walk the Agentic Search Intelligence API.")
    p.add_argument("--host", default="http://localhost:8000")
    p.add_argument("--name", default="Acme")
    p.add_argument("--domain", default="acme.io")
    p.add_argument("--industry", default="project management")
    p.add_argument("--competitors", default="asana.com,monday.com",
                   help="comma-separated")
    p.add_argument("--question", default="Are we visible for agile planning tools?")
    p.add_argument("--min-score", type=float, default=0.5)
    p.add_argument("--profile", help="reuse an existing profile instead of creating one")
    p.add_argument("--recheck", action="store_true",
                   help="also re-measure the top query and show it updated in place")
    p.add_argument("--curl", action="store_true", help="print the equivalent curl commands")
    args = p.parse_args()

    SHOW_CURL = args.curl
    host = args.host.rstrip("/")
    api = f"{host}/api/v1"

    # -- 0. is the server up, and in which mode? ---------------------------
    step(f"Checking {host}")
    _, health = call("GET", f"{host}/health")
    print(f"    status          : {health.get('status')}")
    print(f"    llm_mode        : {health.get('llm_mode')}")
    print(f"    mock_dataforseo : {health.get('mock_dataforseo')}")
    if health.get("mock_dataforseo") is False:
        print(f"\n    {YLW}note:{OFF} MOCK_DATAFORSEO=false — every call hits the paid API.")
        print("          An unverified DataForSEO account returns 40104 and the run")
        print("          will come back 'partial'. Set MOCK_DATAFORSEO=true for fixtures.")

    # -- 1. profile --------------------------------------------------------
    if args.profile:
        step(f"Reusing profile {args.profile}")
        code, profile = call("GET", f"{api}/profiles/{args.profile}")
        if code != 200:
            die(f"profile {args.profile} not found (HTTP {code})")
        profile_uuid = args.profile
    else:
        step("Creating profile")
        code, profile = call("POST", f"{api}/profiles", {
            "name": args.name,
            "domain": args.domain,
            "industry": args.industry,
            "competitors": [c.strip() for c in args.competitors.split(",") if c.strip()],
        })
        if code != 201:
            print(json.dumps(profile, indent=2))
            die(f"create failed (HTTP {code})")
        profile_uuid = profile["uuid"]
    print(f"    {profile_uuid}  {profile['name']} ({profile['domain']})")

    # -- 2. run ------------------------------------------------------------
    step(f'Running: "{args.question}"')
    code, run = call("POST", f"{api}/profiles/{profile_uuid}/run",
                     {"question": args.question})
    if code != 200:
        print(json.dumps(run, indent=2))
        die(f"run failed (HTTP {code})")

    metrics, colour = run["metrics"], STATUS_COLOUR.get(run["status"], "")
    print(f"    status     : {colour}{run['status']}{OFF}   degraded: {run['degraded']}")
    print(f"    calls      : {run['planned_call_count']} planned   "
          f"records: {run['extracted_record_count']}   tokens: {run['tokens_used']}")
    print(f"    path       : {' -> '.join(metrics['node_sequence'])}")
    print(f"    api calls  : {metrics['total_api_calls']}   retries: {metrics['total_retries']}")
    print(f"    correlation: {DIM}{run['correlation_id']}{OFF}")
    if run["errors"]:
        print(f"\n    {RED}errors ({len(run['errors'])}){OFF}")
        for e in run["errors"]:
            tool = f" {e['tool']}" if e.get("tool") else ""
            print(f"      - [{e['node']}/{e['kind']}]{tool} {e['message'][:160]}")
    print(f"\n    {run['report']['summary']}")

    # -- 3. queries --------------------------------------------------------
    step(f"Queries with opportunity_score >= {args.min_score}")
    _, page = call("GET", f"{api}/profiles/{profile_uuid}/queries"
                          f"?min_score={args.min_score}")
    if not page["items"]:
        print("    (none above the threshold)")
    else:
        print(f"    {'score':>6}  {'visible':<8} {'pos':>4} {'volume':>8} {'diff':>5}  "
              f"{'status':<8} query")
        print("    " + "-" * 86)
        for q in page["items"]:
            visible = {True: "yes", False: "NO", None: "unknown"}[q["domain_visible"]]
            print(f"    {q['opportunity_score']:>6}  {visible:<8} "
                  f"{str(q['visibility_position'] or '-'):>4} "
                  f"{str(q['estimated_search_volume'] or '-'):>8} "
                  f"{str(q['competitive_difficulty'] or '-'):>5}  "
                  f"{q['retrieval_status']:<8} {q['query_text']}")
        print(f"\n    {page['total']} shown (page {page['page']}, "
              f"{page['per_page']} per page)")

    # -- 4. recommendations ------------------------------------------------
    step("Recommendations")
    _, recs = call("GET", f"{api}/profiles/{profile_uuid}/recommendations")
    if not recs["items"]:
        print("    (none)")
    else:
        for i, r in enumerate(recs["items"], 1):
            print(f"    {i}. [{r['priority']:^6}] {r['title']}")
            print(f"       type     : {r['content_type']}")
            print(f"       keywords : {', '.join(r['target_keywords'])}")
            print(f"       why      : {r['rationale']}\n")
        print(f"    {recs['total']} total")

    # -- 5. recheck --------------------------------------------------------
    if args.recheck:
        step("Rechecking the top query in place")
        _, top = call("GET", f"{api}/profiles/{profile_uuid}/queries?per_page=1")
        if not top["items"]:
            print("    (no queries to recheck)")
        else:
            target = top["items"][0]
            print(f"    target: {target['query_text']}")
            before = target["opportunity_score"]
            code, recheck = call("POST", f"{api}/queries/{target['uuid']}/recheck")
            if code != 200:
                print(json.dumps(recheck, indent=2))
                die(f"recheck failed (HTTP {code})")
            print(f"    kind: {recheck['kind']}   status: {recheck['status']}")
            _, after_page = call("GET", f"{api}/profiles/{profile_uuid}/queries")
            after = next((q for q in after_page["items"]
                          if q["uuid"] == target["uuid"]), None)
            if after:
                print(f"    score {before} -> {after['opportunity_score']}   "
                      f"(same row uuid: updated in place, not duplicated)")
            print(f"    {after_page['total']} queries total, unchanged by the recheck")

    print(f"\n{GRN}done.{OFF} reuse this profile:\n"
          f"  ./walkthrough.py --profile {profile_uuid} --question \"...\"")


if __name__ == "__main__":
    main()
