"""Score whether models call tools correctly, on Ollama's native and OpenAI-compatible APIs.

Standalone on purpose: depends only on httpx (plus PyYAML for the cases file). Scoring is
deterministic (no model judge): each case is replayed at temperature 0 and checked for a native
tool call, the right tool(s), exact arguments, no leaked thinking, and no empty or truncated
arguments. Server errors and timeouts are counted separately from wrong answers. Results are
appended to a markdown file and failed attempts to a JSONL log, never overwritten.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import itertools
import json
import platform
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

DEFAULT_URL = "http://localhost:11434"
CASES_PATH = Path(__file__).resolve().parent / "toolcall_cases.yaml"
CATEGORIES = ["single", "hard", "none", "parallel", "chain", "recovery"]
NUM_CTX = 8192
SCHEMA_KEYS = {"type", "description", "enum", "items", "properties", "required"}
TOOL_KEYS = ("name", "description", "parameters")
EXTRA = "extra optional argument"
THINK_MODES = {"on": [True], "off": [False], "both": [True, False]}
ABORT_AFTER = 5  # Consecutive errors/timeouts that abort a model/think group.
_TYPES = dict(string=str, integer=int, number=(int, float), boolean=bool, array=list, object=dict)


@dataclass
class Call:  # One tool call from a reply; args is the parsed value (None if the JSON was bad).
    name: str
    args: Any
    raw: str | None = None
    id: str = ""
    bad_json: bool = False


@dataclass
class Reply:  # The parts of a chat reply that scoring needs.
    content: str = ""
    calls: list[Call] = field(default_factory=list)
    truncated: bool = False


def check_schema(value: Any, schema: dict[str, Any], path: str = "args") -> list[str]:
    """Validate a value against the small JSON-schema subset the tools use."""
    kind = schema.get("type")
    if kind:
        ok = isinstance(value, _TYPES[kind]) and (kind == "boolean" or not isinstance(value, bool))
        if kind == "integer" and isinstance(value, float):
            ok = value.is_integer()
        if not ok:
            return [f"{path}: expected {kind}"]
    errs: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in enum")
    if kind == "object":
        props = schema.get("properties", {})
        errs += [
            f"{path}.{k}: missing required" for k in schema.get("required", []) if k not in value
        ]
        errs += [f"{path}.{k}: unknown argument" for k in value if k not in props]
        for key in (k for k in props if k in value):
            errs += check_schema(value[key], props[key], f"{path}.{key}")
    if kind == "array" and "items" in schema:
        for i, item in enumerate(value):
            errs += check_schema(item, schema["items"], f"{path}[{i}]")
    return errs


def _schema_problems(schema: Any, path: str) -> list[str]:
    """Reject schema keys outside the supported subset (a YAML slip adds stray keys)."""
    if not isinstance(schema, dict):
        return [f"{path}: schema must be a mapping"]
    errs = [f"{path}: unknown schema key {k!r}" for k in schema if k not in SCHEMA_KEYS]
    subs = [(f"{path}.{k}", v) for k, v in (schema.get("properties") or {}).items()]
    subs += [(f"{path}[]", schema.get("items"))] * ("items" in schema)
    return errs + [e for sub_path, sub in subs for e in _schema_problems(sub, sub_path)]


def validate_cases(data: dict[str, Any]) -> list[str]:
    """Return a list of problems in a loaded cases file (empty means valid)."""
    errs: list[str] = []
    if not isinstance(data.get("version"), int) or isinstance(data["version"], bool):
        errs.append("version: must be an integer")
    tools: dict[str, Any] = {}
    for i, tool in enumerate(data.get("tools", [])):
        label = tool.get("name") or f"tool #{i + 1}"
        errs += [f"{label}: missing {k}" for k in TOOL_KEYS if k not in tool]
        params = tool.get("parameters")
        errs += _schema_problems(params, label) if params is not None else []
        if tool.get("name") and isinstance(params, dict):
            tools[tool["name"]] = params
    seen: set[str] = set()
    for case in data.get("cases", []):
        cid = case.get("id", "?")
        errs += [f"{cid}: duplicate id"] * (cid in seen)
        seen.add(cid)
        cat = case.get("category")
        errs += [f"{cid}: bad category"] * (cat not in CATEGORIES)
        turns = [case] + list(case.get("then", []))
        errs += [f"{cid}: needs a 'then' turn"] * (cat in ("chain", "recovery") and len(turns) < 2)
        errs += [f"{cid}: none case must expect []"] * (cat == "none" and case.get("expect") != [])
        errs += [f"{cid}: follow-up turn without result" for t in turns[1:] if "result" not in t]
        for call in (c for t in turns for c in t.get("expect", [])):
            if call.get("tool") not in tools:
                errs.append(f"{cid}: unknown tool {call.get('tool')}")
            elif "args" in call:
                errs += [f"{cid}: {e}" for e in check_schema(call["args"], tools[call["tool"]])]
    return errs


def load_cases(path: Path) -> dict[str, Any]:
    """Load and validate the cases file; PyYAML is imported here so --help never needs it."""
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML is required to read the cases file: pip install pyyaml") from None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if problems := validate_cases(data):
        raise SystemExit("Invalid cases file:\n" + "\n".join(problems))
    return data


def _eq(a: Any, b: Any) -> bool:
    """JSON equality where 3 == 3.0 but True != 1."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, str) and isinstance(b, str):
        return a.strip() == b.strip()  # Reason: stray edge whitespace is not a wrong answer.
    return type(a) is type(b) and a == b


def args_match(actual: Any, exp: dict, allow_extra: bool, schema: dict | None = None) -> str | None:
    """Compare a call's arguments with an expected spec: None if they match, else the reason.

    The reason is "wrong arguments", "unknown argument" (key outside the schema or failing it)
    or EXTRA (only schema-valid optional extras: a strict failure, a lenient pass)."""
    if "args" not in exp:
        return None
    if not isinstance(actual, dict):
        return "wrong arguments"
    for key, want in exp["args"].items():
        mode, got = exp.get("match", {}).get(key), actual.get(key)
        if key not in actual:
            return "wrong arguments"
        if mode == "contains":
            ok = isinstance(got, str) and str(want).lower() in got.lower()
        else:
            ok = mode == "any" or _eq(got, want)
        if not ok:
            return "wrong arguments"
    extras = set(actual) - set(exp["args"])
    if not extras or allow_extra:
        return None
    props = (schema or {}).get("properties", {})
    if any(k not in props or check_schema(actual[k], props[k], k) for k in extras):
        return "unknown argument"
    return EXTRA


def _looks_like_call(content: str) -> bool:
    """True if message content is itself a tool call (JSON or a <tool_call> block)."""
    if "<tool_call>" in content:
        return True
    text = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M).strip()
    try:
        data = json.loads(text)
    except ValueError:
        return bool(re.search(r'"name"\s*:.*"arguments"\s*:', content, re.S))
    items = data if isinstance(data, list) else [data]
    return any(
        isinstance(i, dict) and {"name", "tool", "function", "tool_calls"} & i.keys() for i in items
    )


def score_turn(
    reply: Reply, expect: list[dict], allow_extra: bool = False, schemas: dict | None = None
) -> str | None:
    """Failure reason for one reply, or None if correct; schemas maps tool name to its schema."""
    if reply.truncated:
        return "truncated"
    texts = [reply.content] + [c.raw or json.dumps(c.args) for c in reply.calls]
    if any("<think>" in t or "</think>" in t for t in texts):
        return "thinking leaked"
    if not reply.calls:
        if _looks_like_call(reply.content):
            return "tool call in text"
        return "no tool call" if expect else None
    if not expect:
        return "unexpected tool call"
    if any(c.bad_json for c in reply.calls):
        return "invalid arguments JSON"
    remaining = list(expect)
    extra = False
    wanted = {e["tool"] for e in expect}
    for call in reply.calls:
        cands = [e for e in remaining if e["tool"] == call.name]
        if not cands:
            return "wrong number of calls" if call.name in wanted else "wrong tool"
        if not call.args and any(e.get("args", True) for e in cands):
            return "empty arguments"
        ok_extra = allow_extra or any(e.get("allow_extra") for e in cands)
        sch = (schemas or {}).get(call.name)
        results = [(e, args_match(call.args, e, ok_extra, sch)) for e in cands]
        hit = next((r for r in results if r[1] is None), None) or next(
            (r for r in results if r[1] == EXTRA), None
        )
        if hit is None:
            return results[0][1]
        extra = extra or hit[1] == EXTRA
        remaining.remove(hit[0])
    return "wrong number of calls" if remaining else (EXTRA if extra else None)


def _call(name: str, raw: Any, call_id: str) -> Call:
    """Build a Call; OpenAI sends arguments as a JSON string that may not parse."""
    if isinstance(raw, str):
        try:
            return Call(name, json.loads(raw) if raw.strip() else {}, raw, call_id)
        except ValueError:
            return Call(name, None, raw, call_id, bad_json=True)
    return Call(name, raw, None, call_id)


def parse_reply(backend: str, body: dict[str, Any]) -> Reply:
    """Normalise an Ollama or OpenAI-compatible response body."""
    if backend == "ollama":
        if body.get("error"):
            raise RuntimeError(str(body["error"]))
        msg, truncated = body["message"], body.get("done_reason") == "length"
    else:
        choice = body["choices"][0]
        msg, truncated = choice["message"], choice.get("finish_reason") == "length"
    calls = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc["function"]
        calls.append(_call(fn["name"], fn.get("arguments"), tc.get("id") or f"call_{i}"))
    return Reply(msg.get("content") or "", calls, truncated)


async def chat(
    client: httpx.AsyncClient,
    backend: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    think: bool | None,
    timeout: float,
) -> Reply:
    """Send one non-streaming chat request; raises on HTTP errors and timeouts."""
    payload: dict[str, Any] = dict(model=model, messages=messages, tools=tools)
    if backend == "ollama":
        path = "/api/chat"
        payload.update(stream=False, options={"temperature": 0, "num_ctx": NUM_CTX})
        if think is not None:  # Reason: models without a think switch reject the field.
            payload["think"] = think
    else:
        path = "/v1/chat/completions"
        payload["temperature"] = 0
    resp = await client.post(path, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {' '.join(resp.text.split())[:100]}")
    return parse_reply(backend, resp.json())


def _echo(backend: str, reply: Reply, result: Any) -> list[dict]:
    """Echo the model's calls and the canned tool result back into the history."""
    ol = backend == "ollama"
    calls, results = [], []
    for c in reply.calls:
        fn = {"name": c.name, "arguments": c.args if ol else c.raw or json.dumps(c.args)}
        calls.append({"function": fn} if ol else {"id": c.id, "type": "function", "function": fn})
        ref = {"tool_name": c.name} if ol else {"tool_call_id": c.id}
        results.append({"role": "tool", "content": json.dumps(result), **ref})
    content = reply.content if ol else reply.content or None
    return [{"role": "assistant", "content": content, "tool_calls": calls}] + results


async def run_case(
    client: httpx.AsyncClient,
    backend: str,
    model: str,
    think: bool | None,
    case: dict[str, Any],
    data: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """Replay one case (all its turns) and return passed/reason/error/latency."""
    tools = [{"type": "function", "function": t} for t in data["tools"]]
    messages = [
        dict(role="system", content=data["system"]),
        dict(role="user", content=case["prompt"]),
    ]
    turns = [case] + list(case.get("then", []))
    row: dict[str, Any] = dict(id=case["id"], category=case["category"], passed=False)
    row.update(lenient=False, reason=None, error=None, step=None)
    schemas = {t["name"]: t["parameters"] for t in data["tools"]}
    start = time.perf_counter()
    i, turn = 0, case
    try:
        for i, turn in enumerate(turns):
            reply = await chat(client, backend, model, messages, tools, think, timeout)
            reason = score_turn(reply, turn["expect"], case.get("allow_extra", False), schemas)
            if reason and (reason != EXTRA or row["step"] is None):
                row.update(step=i + 1, expected=turn["expect"], reply=reply)
            why = reason if i == 0 else f"step {i + 1}: {reason}"
            if reason == EXTRA:  # Strict failure, but the chain carries on for the lenient score.
                row["reason"] = row["reason"] or why
            elif reason:
                row["reason"] = why
                break
            if i + 1 < len(turns):
                messages += _echo(backend, reply, turns[i + 1]["result"])
        else:
            row["lenient"] = True
            row["passed"] = row["reason"] is None
    except Exception as exc:
        text = " ".join(f"{type(exc).__name__}: {exc}".split())[:120]
        row["error"] = (
            f"timeout after {timeout:g}s" if isinstance(exc, httpx.TimeoutException) else text
        )
        row.update(step=i + 1, expected=turn["expect"], reply=None)
    row["latency"] = time.perf_counter() - start
    return row


async def evaluate(
    client: httpx.AsyncClient,
    backend: str,
    models: list[str],
    modes: list[bool | None],
    runs: int,
    data: dict[str, Any],
    timeout: float,
    log_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Run every model x think mode x run x case; a dead model becomes a FAILED row."""
    groups: list[dict[str, Any]] = []
    for model in models:
        for think in modes:
            label = "n/a" if think is None else ("on" if think else "off")
            group: dict[str, Any] = {"model": model, "backend": backend, "think": label, "rows": []}
            streak = 0
            for run_no, case in itertools.product(range(1, runs + 1), data["cases"]):
                row = await run_case(client, backend, model, think, case, data, timeout)
                group["rows"].append(row)
                if log_path and not row["passed"]:
                    log_failure(log_path, group, run_no, row)
                streak = streak + 1 if row["error"] else 0
                n = len(group["rows"])
                if row["error"] and n == 1:  # Fail fast: a missing model must not cost every case.
                    group["note"] = f"FAILED: {row['error']}"
                elif streak >= ABORT_AFTER:  # Reason: the server probably died mid-run.
                    group["note"] = f"ABORTED after {n} attempts: {row['error']}"
                if "note" in group:
                    break
            print(f"{model} [{backend}, think={label}]: {group.get('note', 'done')}", flush=True)
            groups.append(group)
        if backend == "ollama":  # Evict so the next model gets a clean load.
            with contextlib.suppress(Exception):
                body = {"model": model, "keep_alive": 0}
                await client.post("/api/generate", json=body, timeout=30)
    return groups


def log_failure(path: Path, group: dict[str, Any], run_no: int, row: dict[str, Any]) -> None:
    """Append one JSON line for a failed or errored attempt (never overwrites the file)."""
    rec = dict(timestamp=datetime.now().isoformat(timespec="seconds"), model=group["model"])
    rec.update(backend=group["backend"], think=group["think"], run=run_no, case=row["id"])
    rec.update(step=row["step"], reason=row["reason"] or row["error"])
    reply = row["reply"] or Reply()
    actual = [{"name": c.name, "args": c.raw if c.bad_json else c.args} for c in reply.calls]
    rec.update(expected=row["expected"], actual=actual, content=reply.content[:500])
    append_results(path, json.dumps(rec) + "\n")


def _tally(pairs: Any) -> str:
    """Group (case id, text) pairs into 'id (text) xN' items."""
    return ", ".join(f"{c} ({w}) x{n}" for (c, w), n in sorted(Counter(pairs).items()))


def format_report(groups: list[dict], url: str, runs: int, version: int | None = None) -> str:
    """Render one timestamped markdown section: pass-rate table, failed and errored case ids."""
    host = f"{platform.node()} ({platform.system()} {platform.release()})"
    ver = f" | cases v{version}" if version is not None else ""
    lines = [
        f"## Tool-calling eval {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{ver}",
        "",
        f"Host: {host} | Server: {url} | {runs} run(s) per case | pass rates count server "
        "errors and timeouts as failures (also shown in Errors). Strict fails a call that adds "
        "a schema-valid optional argument; lenient passes it",
        "",
        f"| Model | Backend | Think | {' | '.join(CATEGORIES)} | Overall (strict) "
        "| Overall (lenient) | Median latency (s) | Errors |",
        "|---|---|---|" + "---|" * (len(CATEGORIES) + 4),
    ]
    failures, errored = [], []
    for g in groups:
        head = f"| {g['model']} | {g['backend']} | {g['think']} |"
        tag = f"- {g['model']} [{g['backend']}, think={g['think']}]: "
        if errs := _tally((r["id"], r["error"][:60]) for r in g["rows"] if r["error"]):
            errored.append(tag + errs)
        if "note" in g:
            lines.append(f"{head} {g['note'].replace('|', '/')} |" + " |" * (len(CATEGORIES) + 3))
            continue
        rows = g["rows"]
        scored = [r for r in rows if not r["error"]]

        def pct(sub: list[dict[str, Any]], key: str) -> str:
            ok = sum(1 for r in sub if r["passed"] or (key == "lenient" and r.get("lenient")))
            return f"{100 * ok / len(sub):.0f}%" if sub else "-"

        cells = [pct([r for r in rows if r["category"] == c], "passed") for c in CATEGORIES]
        med = f"{statistics.median(r['latency'] for r in scored):.2f}" if scored else "-"
        lines.append(
            f"{head} {' | '.join(cells)} | {pct(rows, 'passed')} | {pct(rows, 'lenient')} "
            f"| {med} | {len(rows) - len(scored)} |"
        )
        if bad := _tally((r["id"], r["reason"]) for r in scored if not r["passed"]):
            failures.append(tag + bad)
    if failures:
        lines += ["", "Failed cases:", ""] + failures
    if errored:
        lines += ["", "Errored cases (server error or timeout):", ""] + errored
    return "\n".join(lines) + "\n\n"


def append_results(path: Path, text: str) -> None:
    """Append to the results file, creating parent directories; never truncates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (kept free of I/O so --help works anywhere)."""
    p = argparse.ArgumentParser(
        description="Score tool-calling correctness of local models (--think: Ollama only)."
    )
    p.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    p.add_argument("--models", required=True, help="Comma-separated model names")
    p.add_argument("--url", default=DEFAULT_URL, help="Server base URL")
    p.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Repeats of every case; at temperature 0 repeats mostly reproduce the same answers",
    )
    p.add_argument("--think", choices=["on", "off", "both"], default="both", help="Ollama only")
    p.add_argument("--timeout", type=float, default=300, help="Per-request timeout in seconds")
    p.add_argument("--cases", default=str(CASES_PATH), help="Cases YAML file")
    p.add_argument("--out", default="docs/benchmarks/toolcall-results.md", help="File to append to")
    p.add_argument("--log-failures", help="Failures JSONL (default: --out + -failures.jsonl)")
    return p


async def main_async(args: argparse.Namespace) -> int:
    """Run the evaluation and append the report."""
    data = load_cases(Path(args.cases))
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    modes: list = [None] if args.backend == "openai" else THINK_MODES[args.think]
    out = Path(args.out)
    log_path = Path(args.log_failures or out.with_name(f"{out.stem}-failures.jsonl"))
    async with httpx.AsyncClient(base_url=args.url) as client:
        groups = await evaluate(
            client, args.backend, models, modes, args.runs, data, args.timeout, log_path
        )
    append_results(out, format_report(groups, args.url, args.runs, data["version"]))
    print(f"Appended {len(groups)} row(s) to {args.out}; failures logged to {log_path}")
    return 0


def main() -> int:
    """CLI entry point."""
    return asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
