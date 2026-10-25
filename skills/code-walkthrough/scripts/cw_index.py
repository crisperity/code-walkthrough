"""Index logic for the /walkthrough page: scan the store, group rows, parse what the omnibox
receives. Pure functions; `now` and `run` are passed in."""

import json
import os
import re
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cw_repo
import cw_store

TICKET_RE = re.compile(r"^[A-Z][A-Z0-9]+-\d+")
KEY_SUFFIX_RE = re.compile(r"-[0-9a-f]{6}$")
GH_PR_RE = re.compile(r"^https?://github\.com/([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+)/pull/(\d+)(?:[/?#].*)?$")
RANGE_SPLIT_RE = re.compile(r"(?<!\.)\.{2,3}(?!\.)")
REF_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._/^~@-]*$")
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
BUCKETS = ("Today", "Yesterday", "Earlier this week", "Last week", "Older")
META_FIELDS = ("key", "id", "repo", "base", "head", "base_ref", "head_ref", "pr", "gh_repo", "target", "slug",
               "title", "status", "error", "remedy", "created_at", "updated_at", "sibling_of", "batches", "route")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_META_CACHE, _THREADS_CACHE = {}, {}
_CACHE_LOCK = threading.Lock()


def _ts(s):
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return _EPOCH
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _pr(e):
    try:
        return int(e.get("pr"))
    except (TypeError, ValueError):
        return None


def _read_threads(d):
    raw = cw_store.read_json(d / "threads.json", default=None)
    threads = raw.get("threads") if isinstance(raw, dict) else raw
    return [t for t in threads if isinstance(t, dict)] if isinstance(threads, list) else []


def _stamp(path):
    try:
        st = path.stat()
        return st.st_mtime_ns, st.st_size, st.st_ino
    except OSError:
        return None


def _meta_entry(d):
    ns = _stamp(d / "meta.json")
    hit = _META_CACHE.get(d)
    if hit and hit[0] == ns:
        return hit[1]
    meta = cw_store.read_meta(d)
    e = None
    if isinstance(meta, dict):
        e = {f: meta.get(f) for f in META_FIELDS}
        e["key"], e["id"] = meta.get("key") or d.parent.name, meta.get("id") or d.name
    _META_CACHE[d] = (ns, e)
    return e


def _threads_entry(d):
    ns = _stamp(d / "threads.json")
    hit = _THREADS_CACHE.get(d)
    if hit and hit[0] == ns:
        return hit[1]
    threads = _read_threads(d) if ns is not None else []
    newest = sorted(threads, key=lambda t: str(t.get("first_comment_at") or ""), reverse=True)[:3]
    val = (len(threads), [{"path": t.get("path"), "line": t.get("line"), "body": str(t.get("body") or "")[:300],
                           "resolved_by": t.get("resolved_by")} for t in newest])
    _THREADS_CACHE[d] = (ns, val)
    return val


def scan(root=None, threads=True):
    """One entry per <root>/<key>/<id>/meta.json: meta fields plus thread_count,
    threads_preview (newest 3 by first_comment_at) and managed (checkout is a clone under
    cw_repo.cache_root()). threads=False skips threads.json (thread_count None, no preview).
    Parsed files are memoised per directory on (mtime_ns, size, inode)."""
    root = Path(root) if root else cw_store.store_root()
    managed_root = cw_repo.cache_root()
    out, seen = [], set()
    # ponytail: one lock around the whole scan; per-directory locking if concurrent scans ever contend
    with _CACHE_LOCK:
        for meta_path in sorted(root.glob("*/*/meta.json")):
            d = meta_path.parent
            seen.add(d)
            base = _meta_entry(d)
            if base is None:
                continue
            e = dict(base)
            e["thread_count"], e["threads_preview"] = _threads_entry(d) if threads else (None, [])
            e["managed"] = bool(e["repo"]) and Path(e["repo"]).is_relative_to(managed_root)
            out.append(e)
        for cache in (_META_CACHE, _THREADS_CACHE):
            for d in [d for d in cache if d not in seen]:
                del cache[d]
    return out


def label(e):
    ref = e.get("sibling_of")
    if isinstance(ref, dict) and ref.get("title"):
        return str(ref["title"])
    m = TICKET_RE.match(e.get("title") or "")
    if m:
        return m.group(0)
    if _pr(e) is not None:
        return f"#{_pr(e)}"
    target = str(e.get("target") or e.get("slug") or e.get("id") or "")
    return target[:7] if HEX40_RE.fullmatch(target) else target[:40]


def _repo_name(e):
    return e["gh_repo"].split("/")[-1] if e.get("gh_repo") else Path(e.get("repo") or "").name


def _dkey(e):
    return (e["gh_repo"], _pr(e)) if e.get("gh_repo") and _pr(e) is not None else None


def _dedupe(entries):
    """One entry per (gh_repo, pr), newest updated_at wins. Returns (kept, {id(kept): [other key basenames]})."""
    best = {}
    for e in entries:
        k = _dkey(e) or ("_", id(e))
        cur = best.get(k)
        if cur is None or _ts(e.get("updated_at")) > _ts(cur.get("updated_at")):
            best[k] = e
    also = {}
    for e in entries:
        win = best[_dkey(e) or ("_", id(e))]
        if win is not e:
            names = also.setdefault(id(win), [])
            name = KEY_SUFFIX_RE.sub("", e["key"])
            if name != KEY_SUFFIX_RE.sub("", win["key"]) and name not in names:
                names.append(name)
    return list(best.values()), also


def _linked(parent, child):
    """ponytail: heuristic, not the PR's real baseRefName. Requires parent.pr < child.pr so a dev->main
    integration PR (head_ref "dev") is not the parent of feature PRs based on dev; a fast-forward merge
    of the parent into the child's base can still fake a link."""
    pp, cp = _pr(parent), _pr(child)
    if pp is None or cp is None or pp >= cp:
        return False
    return bool(child.get("base_ref") and child["base_ref"] == parent.get("head_ref")
                or child.get("base") and child["base"] == parent.get("head"))


def _chains(kept):
    """Linear stacks of 2+ PRs, base first. A child's base_ref (or base SHA) equals its parent's
    head_ref (or head SHA), same gh_repo, parent PR number lower (see _linked). Each PR has at most
    one parent and one child."""
    cand = sorted((e for e in kept if e.get("gh_repo") and not e.get("sibling_of") and _pr(e) is not None), key=_pr)
    by_head = {}
    for p in cand:
        for k in (p.get("head_ref"), p.get("head")):
            if k:
                by_head.setdefault((p["gh_repo"], k), []).append(p)
    nxt, has_parent = {}, set()
    for c in cand:
        parents = sorted((p for k in (c.get("base_ref"), c.get("base")) if k
                          for p in by_head.get((c["gh_repo"], k), ())), key=_pr)
        p = next((p for p in parents if id(p) not in nxt and _linked(p, c)), None)
        if p is not None:
            nxt[id(p)] = c
            has_parent.add(id(c))
    chains = []
    for e in cand:
        if id(e) in nxt and id(e) not in has_parent:
            chain = [e]
            while id(chain[-1]) in nxt:
                chain.append(nxt[id(chain[-1])])
            chains.append(chain)
    return chains


def _bucket(dt, now):
    today = now.date()
    d = dt.astimezone(now.tzinfo).date()
    monday = today - timedelta(days=today.weekday())
    if d >= today:
        return "Today"
    if d == today - timedelta(days=1):
        return "Yesterday"
    if d >= monday:
        return "Earlier this week"
    if d >= monday - timedelta(days=7):
        return "Last week"
    return "Older"


def _status(e):
    raw = e.get("status")
    return "building" if raw == "building" else "failed" if raw in ("failed", "interrupted") else "done"


def _row(e, also, now):
    status = _status(e)
    created_bucket = _bucket(_ts(e.get("created_at")), now)
    return {
        "key": e["key"], "id": e["id"], "label": label(e), "title": e.get("title"), "status": status,
        "error": (e.get("error") or ("interrupted" if e.get("status") == "interrupted" else None)) if status == "failed" else None,
        "remedy": e.get("remedy") if status == "failed" else None,
        "repo": e.get("repo"), "repo_name": _repo_name(e), "gh_repo": e.get("gh_repo"), "pr": _pr(e),
        "target": e.get("target"), "base_ref": e.get("base_ref"), "head_ref": e.get("head_ref"),
        "base": e.get("base"), "head": e.get("head"), "created_at": e.get("created_at"), "created_bucket": created_bucket,
        "updated_at": e.get("updated_at"), "batches": e.get("batches"),
        "thread_count": e.get("thread_count", 0), "threads_preview": e.get("threads_preview", []),
        "managed": bool(e.get("managed")), "also": also, "parent": None, "stack": None, "bucket": None,
    }


def rows(entries, now):
    """Ordered list for the index. `now` is an aware datetime; buckets use its timezone. Units
    (a stack, or a lone walkthrough) sort by their newest member and stay contiguous; sibling
    walkthroughs follow their parent with row["parent"]={key,id}."""
    kept, also = _dedupe(entries)
    by_ref = {(e["key"], e["id"]): e for e in kept}
    win = {_dkey(e): e for e in kept if _dkey(e)}
    for e in entries:
        if _dkey(e):
            by_ref.setdefault((e["key"], e["id"]), win[_dkey(e)])
    kids = {}
    for e in kept:
        ref = e.get("sibling_of")
        p = by_ref.get((ref.get("key"), ref.get("id"))) if isinstance(ref, dict) else None
        if p is not None and p is not e and not p.get("sibling_of"):
            kids.setdefault(id(p), []).append(e)
    is_kid = {id(k) for ks in kids.values() for k in ks}
    chains = _chains(kept)
    in_chain = {id(e) for c in chains for e in c}
    units = list(chains) + [[e] for e in kept if id(e) not in in_chain and id(e) not in is_kid]

    def newest(unit):
        return max(_ts(x.get("updated_at")) for m in unit for x in [m, *kids.get(id(m), [])])

    out = []
    for unit in sorted(units, key=newest, reverse=True):
        bucket = _bucket(newest(unit), now)
        stack = None
        if len(unit) > 1:
            stack = {"id": f"{unit[0]['key']}/{unit[0]['id']}", "n": len(unit),
                     "chain": [{"key": m["key"], "id": m["id"], "label": label(m), "pr": _pr(m)} for m in unit]}
        for i, m in enumerate(unit):
            r = _row(m, also.get(id(m), []), now)
            r["bucket"] = bucket
            r["stack"] = dict(stack, pos=i) if stack else None
            out.append(r)
            for k in sorted(kids.get(id(m), []), key=lambda x: _ts(x.get("updated_at")), reverse=True):
                kr = _row(k, also.get(id(k), []), now)
                kr["bucket"], kr["parent"] = bucket, {"key": m["key"], "id": m["id"]}
                out.append(kr)
    return out


def counts(rows_):
    c = {"all": len(rows_), "building": 0, "done": 0, "failed": 0}
    for r in rows_:
        c[r["status"]] += 1
    return c


def _page_url(key, wid):
    return f"/walkthrough/{key}/{wid}"


def page_nav(entries, key, wid, url=_page_url):
    """Header data for one walkthrough page: breadcrumb plus neighbours in its stack, or None
    when (key, wid) is unknown. `url(key, id)` builds the links."""
    kept, _ = _dedupe(entries)
    me = next((e for e in kept if (e["key"], e["id"]) == (key, wid)), None) or next(
        (e for e in entries if (e["key"], e["id"]) == (key, wid)), None)
    if me is None:
        return None
    chain = next((c for c in _chains(kept) if any(x is me for x in c)), [me])
    i = next(j for j, x in enumerate(chain) if x is me)

    def link(e):
        return {"key": e["key"], "id": e["id"], "label": label(e), "url": url(e["key"], e["id"])}

    return {"index": "/walkthrough", "repo": _repo_name(me), "label": label(me),
            "prev": link(chain[i - 1]) if i else None,
            "next": link(chain[i + 1]) if i + 1 < len(chain) else None, "pos": i, "n": len(chain)}


def parse_target(text):
    """What the omnibox received: {kind:"pr",gh_repo,pr} for a GitHub /pull/N URL,
    {kind:"range",base,head} for a..b or a...b, {kind:"commit",sha} for 7-40 hex, else None."""
    text = (text or "").strip()
    m = GH_PR_RE.match(text)
    if m:
        return {"kind": "pr", "gh_repo": f"{m.group(1)}/{m.group(2)}", "pr": int(m.group(3))}
    parts = RANGE_SPLIT_RE.split(text)
    if len(parts) == 2 and all(REF_RE.match(p) and ".." not in p and not p.endswith(".") for p in parts):
        return {"kind": "range", "base": parts[0], "head": parts[1]}
    if COMMIT_RE.fullmatch(text):
        return {"kind": "commit", "sha": text.lower()}
    return None


def _strip_origin(ref):
    return (ref or "").removeprefix("origin/")


def matches(parsed, entries):
    """Existing walkthroughs for a parsed target, newest first."""
    if not parsed:
        return []
    if parsed["kind"] == "pr":
        hit = [e for e in entries if e.get("gh_repo") == parsed["gh_repo"] and _pr(e) == parsed["pr"]]
    elif parsed["kind"] == "range":
        want = (_strip_origin(parsed["base"]), _strip_origin(parsed["head"]))
        hit = [e for e in entries if (_strip_origin(e.get("base_ref")), _strip_origin(e.get("head_ref"))) == want]
    else:
        sha = parsed["sha"]
        hit = [e for e in entries if str(e.get("head") or "").startswith(sha)]
    hit.sort(key=lambda e: _ts(e.get("updated_at")), reverse=True)
    return [{"key": e["key"], "id": e["id"], "label": label(e), "title": e.get("title"), "status": _status(e),
             "repo_name": _repo_name(e), "updated_at": e.get("updated_at")} for e in hit]


def repo_choices(parsed, entries, run=subprocess.run, timeout=10):
    """Known checkouts a new walkthrough for `parsed` could start in, most recently used first.
    PR: checkouts of that gh_repo. Commit: checkouts where `git cat-file -e` finds it. Range: every known checkout."""
    if not parsed:
        return []
    seen, out = set(), []
    for e in sorted(entries, key=lambda e: _ts(e.get("updated_at")), reverse=True):
        repo = e.get("repo")
        if not repo or repo in seen or (parsed["kind"] == "pr" and e.get("gh_repo") != parsed["gh_repo"]):
            continue
        seen.add(repo)
        if parsed["kind"] == "commit":
            try:
                ok = run(["git", "-C", repo, "cat-file", "-e", f"{parsed['sha']}^{{commit}}"],
                         capture_output=True, text=True, timeout=timeout).returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                ok = False
            if not ok:
                continue
        out.append({"repo": repo, "name": _repo_name(e), "gh_repo": e.get("gh_repo"), "managed": bool(e.get("managed"))})
    return out


def _slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.")[:40] or "walkthrough"


def _git_fetch(repo, specs, run):
    try:
        return run(["git", "-C", repo, "fetch", "origin", *specs], capture_output=True, text=True,
                   timeout=cw_repo.CLONE_TIMEOUT_S).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _pr_info(parsed, repo, gh):
    cwd = repo
    if not os.path.isabs(repo):
        cwd = str(cw_repo.cache_root())
        os.makedirs(cwd, exist_ok=True)
    out = gh({"repo": cwd}, ["pr", "view", str(parsed["pr"]), "--repo", parsed["gh_repo"], "--json",
                             "baseRefName,headRefName,headRefOid,title,isCrossRepository"])
    try:
        info = json.loads(out.stdout) if out.returncode == 0 else None
    except ValueError:
        info = None
    if not isinstance(info, dict) or not info.get("baseRefName") or not info.get("headRefOid"):
        detail = (out.stderr or "").strip()[:200]
        raise cw_store.CWError(f"gh pr view failed for {parsed['gh_repo']}#{parsed['pr']}" + (f": {detail}" if detail else ""),
                               remedy="check gh auth status and that the PR exists")
    return info


def start_args(parsed, repo, *, fresh=False, clone=False, run=subprocess.run, gh=None):
    """Arguments for walkthrough_start from a parse_target result. `repo` is an absolute checkout
    path, or owner/name (PR only: the daemon resolves or clones it). Fetches what the refs need
    first; raises CWError when a PR cannot be read or fetched."""
    if not parsed:
        raise cw_store.CWError("not a PR URL, commit or range", remedy="paste a GitHub PR URL, a commit sha or base...head")
    if not repo:
        raise cw_store.CWError("no repo", remedy="pick a repo")
    local = os.path.isabs(repo)
    kind = parsed["kind"]
    if kind != "pr" and not local:
        raise cw_store.CWError("a range or commit needs a local checkout", remedy="pick a repo")
    if kind == "pr":
        import cw_run
        pr, gh_repo = parsed["pr"], parsed["gh_repo"]
        info = _pr_info(parsed, repo, gh or cw_run._gh)
        base, head = f"origin/{info['baseRefName']}", info["headRefOid"]
        if local:
            ok = False
            if not info.get("isCrossRepository") and info.get("headRefName"):
                branch = info["headRefName"]
                ok = _git_fetch(repo, [f"+refs/heads/{info['baseRefName']}:refs/remotes/{base}",
                                       f"+refs/heads/{branch}:refs/remotes/origin/{branch}"], run)
                head = f"origin/{branch}" if ok else head
            if not ok and not _git_fetch(repo, [f"+refs/heads/{info['baseRefName']}:refs/remotes/{base}",
                                                f"+refs/pull/{pr}/head:refs/cw/pull/{pr}"], run):
                raise cw_store.CWError(f"git fetch failed in {repo}", remedy="check gh auth status and network, then retry")
        args = {"repo": repo, "base": base, "head": head, "pr": pr, "target": f"https://github.com/{gh_repo}/pull/{pr}",
                "slug": f"pr-{pr}", "title": info.get("title") or None}
    elif kind == "commit":
        args = {"repo": repo, "base": f"{parsed['sha']}^", "head": parsed["sha"], "target": parsed["sha"],
                "slug": parsed["sha"][:12]}
    else:
        base, head = parsed["base"], parsed["head"]
        # ponytail: best-effort; a missing origin ref surfaces as prepare_walkthrough's "not in the local object store"
        for ref in (base, head):
            if ref.startswith("origin/"):
                _git_fetch(repo, [f"+refs/heads/{ref[7:]}:refs/remotes/{ref}"], run)
        args = {"repo": repo, "base": base, "head": head, "target": f"{base}...{head}", "slug": _slug(f"{base}...{head}")}
    if clone:
        args["clone"] = True
    if fresh:
        args["fresh"] = True
    return {k: v for k, v in args.items() if v is not None}
