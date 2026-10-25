"""cw_index: grouping, stacks, buckets, target parsing. Fixtures copy the fields of real store entries."""

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import cw_index

UTC = timezone.utc
NOW = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)  # a Thursday; that week's Monday is Oct 19
OZ = "ozone-project/ad-manager-workflows-orchestrator"
AO_KEY = "ad-manager-workflows-orchestrator-174d66"


def E(key, wid, **kw):
    e = {"key": key, "id": wid, "repo": "/w/" + key, "base": None, "head": None, "base_ref": None,
         "head_ref": None, "pr": None, "gh_repo": None, "target": None, "slug": None, "title": None,
         "status": "done", "error": None, "remedy": None, "created_at": "2026-10-23T06:00:00Z",
         "updated_at": "2026-10-22T10:00:00Z", "sibling_of": None, "thread_count": 0, "threads_preview": [],
         "managed": False}
    e.update(kw)
    return e


def ao(wid, pr, title, base, head, base_ref, head_ref, updated):
    return E(AO_KEY, wid, pr=pr, gh_repo=OZ, title=title, base=base, head=head, base_ref=base_ref,
             head_ref=head_ref, updated_at=updated)


def ao_stack():
    return [
        ao("pr-61-ce85", 61, "AO-151: block runs", "4ca74ec0125529421eb68c5e4a7c4bce07640163",
           "a5d491fc72ae77df3e376a5b6e820b84ec995033", "origin/AO-150", "origin/AO-151", "2026-10-22T07:17:49Z"),
        ao("pr-59-4a99", 59, "AO-149: pause runs", "ff905ca57d1d469b6ca20fdfc72cd2b244b1794b",
           "3fb1d6c2f0d617b6d5dea1943d6ca9be4bc19ad9", "origin/dev", "origin/AO-149", "2026-10-22T07:06:01Z"),
        ao("pr-60-fe96", 60, "AO-150: run status", "3fb1d6c2f0d617b6d5dea1943d6ca9be4bc19ad9",
           "4ca74ec0125529421eb68c5e4a7c4bce07640163", "origin/AO-149",
           "4ca74ec0125529421eb68c5e4a7c4bce07640163", "2026-10-22T07:10:56Z"),
    ]


def test_stacks_link_by_ref_and_by_sha():
    other = E("axon-e4aef0", "pr-236-ae93", pr=236, gh_repo="ozone-project/axon", updated_at="2026-10-22T11:00:00Z")
    a = ao("pr-63-7572", 63, "SWAR-81: actor columns", "9847927c", "7cc1b0ad3aaa1aa5c7efc03e679953668e1b1f87",
           "origin/SWAR-81-accept-and-enforce", "origin/SWAR-81-run-actor-and-input", "2026-10-21T13:24:05Z")
    b = ao("pr-64-4e89", 64, "SWAR-81: shared agent record tables", "7cc1b0ad3aaa1aa5c7efc03e679953668e1b1f87",
           "7f767a804579932f808c057df3d09b383ad09883", "origin/SWAR-81-run-actor-and-input",
           "7f767a804579932f808c057df3d09b383ad09883", "2026-10-21T13:50:27Z")
    lone = ao("pr-42-cf44", 42, "feat(SWAR-67): x", "cc531fec", "707f374c", "cc531fe", "707f374", "2026-10-21T18:14:11Z")
    out = cw_index.rows(ao_stack() + [other, b, lone, a], NOW)
    assert [r["id"] for r in out] == ["pr-236-ae93", "pr-59-4a99", "pr-60-fe96", "pr-61-ce85",
                                      "pr-42-cf44", "pr-63-7572", "pr-64-4e89"]
    assert out[0]["stack"] is None and out[4]["stack"] is None
    stack = [r["stack"] for r in out[1:4]]
    assert [s["pos"] for s in stack] == [0, 1, 2] and {s["n"] for s in stack} == {3}
    assert [c["label"] for c in stack[0]["chain"]] == ["AO-149", "AO-150", "AO-151"]
    assert len({s["id"] for s in stack}) == 1
    assert out[5]["stack"]["n"] == 2 and out[6]["stack"]["pos"] == 1 and out[5]["label"] == "SWAR-81"


def test_same_pr_under_two_keys_is_one_row_with_also():
    mk = lambda key, wid, upd: E(key, wid, pr=2562, gh_repo="ozone-project/prebid-server", base_ref="origin/main",
                                 head_ref="origin/TD-273", title="improve: add echo error handler", updated_at=upd)
    old = mk("TD-273-efaa91", "pr-2562-5b2a", "2026-10-13T05:51:47Z")
    new = mk("prebid-server-20fda5", "pr-2562-5b2a", "2026-10-13T06:05:08Z")
    for order in ([old, new], [new, old]):
        out = cw_index.rows(order, NOW)
        assert len(out) == 1 and out[0]["key"] == "prebid-server-20fda5" and out[0]["also"] == ["TD-273"]
        assert out[0]["label"] == "#2562"


def test_cmp_siblings_follow_their_parent_and_never_stack():
    key = "cw-prebid-server-097296"
    parent = E(key, "pr-5007-b381", pr=5007, gh_repo="prebid/prebid-server", title="Code walkthrough: PR 5007",
               base="8cf6dd22", head="b8c98124", updated_at="2026-10-11T20:48:45Z")
    ref = {"key": key, "id": "pr-5007-b381"}
    s1 = E(key, "cmp-2c08813a", sibling_of=ref, base="8cf6dd22", head="fac0e4dc", updated_at="2026-10-11T20:39:20Z")
    s2 = E(key, "cmp-6115ba6a", sibling_of=ref, base="8cf6dd22", head="c538470d", updated_at="2026-10-10T06:56:28Z")
    out = cw_index.rows([s2, s1, parent], NOW)
    assert [r["id"] for r in out] == ["pr-5007-b381", "cmp-2c08813a", "cmp-6115ba6a"]
    assert out[0]["parent"] is None and out[1]["parent"] == ref == out[2]["parent"]
    assert {r["bucket"] for r in out} == {"Older"} and all(r["stack"] is None for r in out)


def test_orphan_sibling_stands_alone():
    out = cw_index.rows([E("k-aaaaaa", "cmp-2c08813a", sibling_of={"key": "gone-aaaaaa", "id": "pr-1-aaaa"})], NOW)
    assert len(out) == 1 and out[0]["parent"] is None


def test_failed_interrupted_and_counts():
    failed = E("crisperit-skills-410ae7", "cmp-21131838", status="failed", target="7c81d71c58844d5aa1bca7b653daf6714686fd51",
               error="small route failed the fragment gate", remedy="re-run /code-walkthrough on the same target")
    cut = E("k-aaaaaa", "cmp-aaaaaaaa", status="interrupted", target="a..b")
    live = E("k-bbbbbb", "cmp-bbbbbbbb", status="building")
    ok = E("k-cccccc", "cmp-cccccccc", error="stale", remedy="stale")
    out = {r["id"]: r for r in cw_index.rows([failed, cut, live, ok], NOW)}
    assert out["cmp-21131838"]["status"] == "failed" and out["cmp-21131838"]["error"].startswith("small route")
    assert out["cmp-21131838"]["remedy"].startswith("re-run")
    assert out["cmp-aaaaaaaa"]["status"] == "failed" and out["cmp-aaaaaaaa"]["error"] == "interrupted"
    assert out["cmp-cccccccc"]["error"] is None and out["cmp-cccccccc"]["remedy"] is None
    assert cw_index.counts(list(out.values())) == {"all": 4, "building": 1, "done": 1, "failed": 2}


def test_labels():
    sha = "7c81d71c58844d5aa1bca7b653daf6714686fd51"
    assert cw_index.label(E("k", "i", title="AO-149: pause", pr=59)) == "AO-149"
    assert cw_index.label(E("k", "i", pr=22, target="#22")) == "#22"
    assert cw_index.label(E("k", "i", target=sha)) == "7c81d71"
    assert cw_index.label(E("k", "i", target="263b199..7c81d71")) == "263b199..7c81d71"
    assert cw_index.label(E("k", "i", title="Make the side panels configurable", pr=5702)) == "#5702"


@pytest.mark.parametrize("stamp,want", [
    ("2026-10-22T00:00:00Z", "Today"),
    ("2026-10-21T23:59:59Z", "Yesterday"),
    ("2026-10-21T00:00:00Z", "Yesterday"),
    ("2026-10-20T23:59:59Z", "Earlier this week"),
    ("2026-10-19T00:00:00Z", "Earlier this week"),
    ("2026-10-18T23:59:59Z", "Last week"),
    ("2026-10-12T00:00:00Z", "Last week"),
    ("2026-10-11T23:59:59Z", "Older"),
    ("garbage", "Older"),
])
def test_bucket_edges(stamp, want):
    assert cw_index.rows([E("k-aaaaaa", "cmp-aaaaaaaa", updated_at=stamp)], NOW)[0]["bucket"] == want


def test_buckets_use_the_timezone_of_now():
    plus2 = timezone(timedelta(hours=2))
    now = datetime(2026, 10, 22, 12, 0, tzinfo=plus2)
    e = E("k-aaaaaa", "cmp-aaaaaaaa", updated_at="2026-10-21T23:30:00Z")  # 01:30 on the 22nd locally
    assert cw_index.rows([e], now)[0]["bucket"] == "Today"
    assert cw_index.rows([e], NOW)[0]["bucket"] == "Yesterday"


def test_units_sort_newest_first_and_stack_takes_its_newest_bucket():
    old = E("k-aaaaaa", "cmp-aaaaaaaa", updated_at="2026-10-01T00:00:00Z")
    mid = E("k-bbbbbb", "cmp-bbbbbbbb", updated_at="2026-10-21T12:00:00Z")
    out = cw_index.rows([old, mid] + ao_stack(), NOW)
    assert [r["bucket"] for r in out] == ["Today"] * 3 + ["Yesterday", "Older"]


def test_scan_reads_meta_threads_and_managed(tmp_path, monkeypatch):
    monkeypatch.setenv("CODE_WALKTHROUGH_HOME", str(tmp_path))
    root = tmp_path / "w"
    managed = tmp_path / "repos" / "github.com" / "o" / "r"

    def put(key, wid, meta, threads=None):
        d = root / key / wid
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({"key": key, "id": wid, **meta}))
        if threads is not None:
            (d / "threads.json").write_text(json.dumps(threads))

    th = [{"path": f"f{i}.go", "line": i, "first_comment_at": f"2026-10-1{i}T00:00:00Z", "body": "x" * 400,
           "resolved_by": "bob" if i == 3 else None} for i in range(5)]
    put("r-aaaaaa", "pr-1-aaaa", {"repo": str(managed), "pr": 1, "title": "t"}, {"threads": th})
    put("l-bbbbbb", "pr-2-bbbb", {"repo": "/home/me/l"}, [th[0]])
    put("n-cccccc", "pr-3-cccc", {})
    (root / "junk" / "x").mkdir(parents=True)
    got = {e["id"]: e for e in cw_index.scan()}
    assert set(got) == {"pr-1-aaaa", "pr-2-bbbb", "pr-3-cccc"}
    a = got["pr-1-aaaa"]
    assert a["thread_count"] == 5 and a["managed"] is True and a["title"] == "t"
    assert [t["path"] for t in a["threads_preview"]] == ["f4.go", "f3.go", "f2.go"]
    assert len(a["threads_preview"][0]["body"]) == 300 and a["threads_preview"][1]["resolved_by"] == "bob"
    assert got["pr-2-bbbb"]["thread_count"] == 1 and got["pr-2-bbbb"]["managed"] is False
    assert got["pr-3-cccc"]["thread_count"] == 0 and got["pr-3-cccc"]["threads_preview"] == []
    assert cw_index.scan(tmp_path / "nowhere") == []


def test_page_nav():
    es = ao_stack()
    nav = cw_index.page_nav(es, AO_KEY, "pr-60-fe96")
    assert nav["repo"] == "ad-manager-workflows-orchestrator" and nav["label"] == "AO-150"
    assert nav["prev"] == {"key": AO_KEY, "id": "pr-59-4a99", "label": "AO-149", "url": f"/walkthrough/{AO_KEY}/pr-59-4a99"}
    assert nav["next"]["label"] == "AO-151" and (nav["pos"], nav["n"]) == (1, 3)
    first = cw_index.page_nav(es, AO_KEY, "pr-59-4a99", url=lambda k, w: f"/x/{k}/{w}")
    assert first["prev"] is None and first["next"]["url"] == f"/x/{AO_KEY}/pr-60-fe96"
    assert cw_index.page_nav(es, AO_KEY, "pr-99-0000") is None
    solo = cw_index.page_nav([E("k-aaaaaa", "cmp-aaaaaaaa", repo="/home/me/proj")], "k-aaaaaa", "cmp-aaaaaaaa")
    assert solo["prev"] is None and solo["next"] is None and solo["repo"] == "proj" and (solo["pos"], solo["n"]) == (0, 1)


@pytest.mark.parametrize("text,want", [
    ("https://github.com/ozone-project/prebid-server/pull/2562",
     {"kind": "pr", "gh_repo": "ozone-project/prebid-server", "pr": 2562}),
    ("  https://github.com/o/r/pull/7/files#diff-1  ", {"kind": "pr", "gh_repo": "o/r", "pr": 7}),
    ("https://github.com/o/r/pull/7?x=1", {"kind": "pr", "gh_repo": "o/r", "pr": 7}),
    ("main...feature/x", {"kind": "range", "base": "main", "head": "feature/x"}),
    ("v1.2.3..v1.3.0", {"kind": "range", "base": "v1.2.3", "head": "v1.3.0"}),
    ("origin/main..origin/AO-149", {"kind": "range", "base": "origin/main", "head": "origin/AO-149"}),
    ("7c81d71", {"kind": "commit", "sha": "7c81d71"}),
    ("7C81D71C58844D5AA1BCA7B653DAF6714686FD51", {"kind": "commit", "sha": "7c81d71c58844d5aa1bca7b653daf6714686fd51"}),
    ("a....b", None),
    ("a..", None),
    ("..b", None),
    ("a..b..c", None),
    ("sovrn", None),
    ("7c81d7", None),
    ("7c81d71c58844d5aa1bca7b653daf6714686fd51a", None),
    ("https://gitlab.com/o/r/pull/7", None),
    ("https://github.com/o/r/issues/7", None),
    ("https://github.com/o/r/pull/x", None),
    ("", None),
    (None, None),
])
def test_parse_target(text, want):
    assert cw_index.parse_target(text) == want


def test_matches():
    es = ao_stack() + [E("k-aaaaaa", "cmp-aaaaaaaa", head="7c81d71c58844d5aa1bca7b653daf6714686fd51", repo="/r/a")]
    pr = cw_index.matches({"kind": "pr", "gh_repo": OZ, "pr": 60}, es)
    assert [m["id"] for m in pr] == ["pr-60-fe96"] and pr[0]["label"] == "AO-150" and pr[0]["status"] == "done"
    assert cw_index.matches({"kind": "pr", "gh_repo": OZ, "pr": 99}, es) == []
    assert cw_index.matches({"kind": "pr", "gh_repo": "x/y", "pr": 60}, es) == []
    rng = cw_index.matches({"kind": "range", "base": "origin/dev", "head": "AO-149"}, es)
    assert [m["id"] for m in rng] == ["pr-59-4a99"]
    assert [m["id"] for m in cw_index.matches({"kind": "commit", "sha": "7c81d71"}, es)] == ["cmp-aaaaaaaa"]
    assert cw_index.matches(None, es) == []


def test_repo_choices():
    es = [E("a-aaaaaa", "pr-1-aaaa", pr=1, gh_repo="o/r", repo="/r/old", updated_at="2026-10-01T00:00:00Z"),
          E("b-bbbbbb", "pr-2-bbbb", pr=2, gh_repo="o/r", repo="/r/new", managed=True, updated_at="2026-10-02T00:00:00Z"),
          E("b-bbbbbb", "pr-3-bbbb", pr=3, gh_repo="o/r", repo="/r/new", updated_at="2026-10-03T00:00:00Z"),
          E("c-cccccc", "pr-4-cccc", pr=4, gh_repo="x/y", repo="/r/other", updated_at="2026-10-04T00:00:00Z")]
    pr = cw_index.repo_choices({"kind": "pr", "gh_repo": "o/r", "pr": 9}, es)
    assert [c["repo"] for c in pr] == ["/r/new", "/r/old"] and pr[0]["name"] == "r"
    assert cw_index.repo_choices({"kind": "pr", "gh_repo": "q/z", "pr": 9}, es) == []

    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(returncode=0 if argv[2] == "/r/old" else 1)

    got = cw_index.repo_choices({"kind": "commit", "sha": "abc1234"}, es, run=run)
    assert [c["repo"] for c in got] == ["/r/old"]
    assert calls[0][:2] == ["git", "-C"] and calls[0][3:] == ["cat-file", "-e", "abc1234^{commit}"]
    assert len(calls) == 3
    assert len(cw_index.repo_choices({"kind": "range", "base": "a", "head": "b"}, es)) == 3
    assert cw_index.repo_choices(None, es) == []


def _gh_ok(info):
    return lambda meta, argv, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(info), stderr="")


def _fetch_run(calls, ok=lambda argv: True):
    def run(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(returncode=0 if ok(argv) else 1, stdout="", stderr="boom")
    return run


PR = {"kind": "pr", "gh_repo": "o/r", "pr": 7}
INFO = {"baseRefName": "dev", "headRefName": "feat", "headRefOid": "a" * 40, "title": "AB-1 t", "isCrossRepository": False}


def test_start_args_pr_same_repo_fetches_branches():
    calls = []
    args = cw_index.start_args(PR, "/w/r", run=_fetch_run(calls), gh=_gh_ok(INFO))
    assert args == {"repo": "/w/r", "base": "origin/dev", "head": "origin/feat", "pr": 7,
                    "target": "https://github.com/o/r/pull/7", "slug": "pr-7", "title": "AB-1 t"}
    assert calls == [["git", "-C", "/w/r", "fetch", "origin", "+refs/heads/dev:refs/remotes/origin/dev",
                      "+refs/heads/feat:refs/remotes/origin/feat"]]


def test_start_args_pr_fork_and_deleted_branch_use_pull_ref_and_oid():
    for info, ok in ((dict(INFO, isCrossRepository=True), lambda a: True),
                     (INFO, lambda a: not any("refs/heads/feat" in x for x in a))):
        calls = []
        args = cw_index.start_args(PR, "/w/r", run=_fetch_run(calls, ok), gh=_gh_ok(info))
        assert args["head"] == "a" * 40 and args["base"] == "origin/dev"
        assert any("+refs/pull/7/head:refs/cw/pull/7" in x for x in calls[-1])


def test_start_args_pr_managed_name_does_no_git_and_passes_clone_fresh():
    calls = []
    args = cw_index.start_args(PR, "o/r", fresh=True, clone=True, run=_fetch_run(calls), gh=_gh_ok(INFO))
    assert calls == [] and args["repo"] == "o/r" and args["head"] == "a" * 40
    assert args["clone"] is True and args["fresh"] is True


def test_start_args_pr_errors_are_cwerrors():
    bad = lambda meta, argv, **kw: SimpleNamespace(returncode=1, stdout="", stderr="no such pr")
    with pytest.raises(cw_index.cw_store.CWError) as e:
        cw_index.start_args(PR, "/w/r", gh=bad)
    assert "no such pr" in str(e.value) and e.value.remedy
    with pytest.raises(cw_index.cw_store.CWError):
        cw_index.start_args(PR, "/w/r", run=_fetch_run([], lambda a: False), gh=_gh_ok(dict(INFO, isCrossRepository=True)))
    for parsed, repo in ((None, "/w/r"), ({"kind": "commit", "sha": "abc1234"}, "o/r"), (PR, None)):
        with pytest.raises(cw_index.cw_store.CWError):
            cw_index.start_args(parsed, repo)


def test_start_args_commit_and_range():
    assert cw_index.start_args({"kind": "commit", "sha": "abc1234"}, "/w/r") == {
        "repo": "/w/r", "base": "abc1234^", "head": "abc1234", "target": "abc1234", "slug": "abc1234"}
    calls = []
    args = cw_index.start_args({"kind": "range", "base": "origin/dev", "head": "feat/x"}, "/w/r", run=_fetch_run(calls, lambda a: False))
    assert args == {"repo": "/w/r", "base": "origin/dev", "head": "feat/x", "target": "origin/dev...feat/x",
                    "slug": "origin-dev...feat-x"}
    assert len(calls) == 1 and calls[0][-1] == "+refs/heads/dev:refs/remotes/origin/dev"


def test_missing_or_garbage_updated_at_west_of_utc_does_not_overflow():
    now = datetime(2026, 10, 22, 12, 0, tzinfo=timezone(timedelta(hours=-5)))
    out = cw_index.rows([E("k-aaaaaa", "cmp-aaaaaaaa", updated_at=None), E("k-bbbbbb", "cmp-bbbbbbbb", updated_at="garbage")], now)
    assert {r["bucket"] for r in out} == {"Older"}


def test_child_with_two_same_head_parents_appears_once():
    mk = lambda pr, base_ref, head_ref: E("k-aaaaaa", f"pr-{pr}-aaaa", pr=pr, gh_repo="o/r", base_ref=base_ref, head_ref=head_ref)
    out = cw_index.rows([mk(1, "main", "feat"), mk(2, "main", "feat"), mk(3, "feat", "child")], NOW)
    assert sorted(r["id"] for r in out) == ["pr-1-aaaa", "pr-2-aaaa", "pr-3-aaaa"]
    assert len({r["stack"]["id"] for r in out if r["stack"]}) <= 1


def test_integration_branch_pr_does_not_stack_feature_prs():
    mk = lambda pr, base_ref, head_ref: E("k-aaaaaa", f"pr-{pr}-aaaa", pr=pr, gh_repo="o/r", base_ref=base_ref, head_ref=head_ref)
    out = cw_index.rows([mk(90, "main", "dev"), mk(10, "dev", "f1"), mk(11, "dev", "f2")], NOW)
    assert len(out) == 3 and all(r["stack"] is None for r in out)


def test_scan_memoises_on_mtime_and_threads_false_skips_threads(tmp_path, monkeypatch):
    monkeypatch.setenv("CODE_WALKTHROUGH_HOME", str(tmp_path))
    d = tmp_path / "w" / "k-aaaaaa" / "pr-1-aaaa"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"key": "k-aaaaaa", "id": "pr-1-aaaa", "title": "one"}))
    (d / "threads.json").write_text(json.dumps([{"path": "a", "body": "b"}]))
    assert cw_index.scan()[0]["thread_count"] == 1
    calls = []
    real = json.loads
    monkeypatch.setattr(cw_index.cw_store.json, "loads", lambda *a, **k: calls.append(1) or real(*a, **k))
    assert cw_index.scan()[0]["title"] == "one" and calls == []
    nt = cw_index.scan(threads=False)[0]
    assert nt["thread_count"] is None and nt["threads_preview"] == [] and calls == []
    (d / "meta.json").write_text(json.dumps({"key": "k-aaaaaa", "id": "pr-1-aaaa", "title": "two"}))
    st = (d / "meta.json").stat()
    os.utime(d / "meta.json", ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000))
    assert cw_index.scan()[0]["title"] == "two" and len(calls) == 1
    (d / "meta.json").unlink()
    assert cw_index.scan() == [] and d not in cw_index._META_CACHE


def test_sibling_label_is_parent_title_and_created_bucket():
    ref = {"key": "k-aaaaaa", "id": "pr-1-aaaa", "title": "Compare A vs B"}
    sib = E("k-aaaaaa", "cmp-aaaaaaaa", pr=5007, sibling_of=ref, created_at="2026-10-22T08:00:00Z")
    assert cw_index.label(sib) == "Compare A vs B"
    assert cw_index.label(E("k", "i", pr=5, sibling_of={"key": "k", "id": "x", "title": ""})) == "#5"
    old = E("k-bbbbbb", "cmp-bbbbbbbb", created_at="2026-09-01T00:00:00Z")
    out = {r["id"]: r for r in cw_index.rows([sib, old], NOW)}
    assert out["cmp-aaaaaaaa"]["created_bucket"] == "Today" and out["cmp-bbbbbbbb"]["created_bucket"] == "Older"


def test_sibling_of_deduped_parent_follows_surviving_row():
    mk = lambda key, wid, upd: E(key, wid, pr=7, gh_repo="o/r", updated_at=upd)
    loser, winner = mk("a-aaaaaa", "pr-7-aaaa", "2026-10-01T00:00:00Z"), mk("b-bbbbbb", "pr-7-bbbb", "2026-10-02T00:00:00Z")
    sib = E("a-aaaaaa", "cmp-aaaaaaaa", sibling_of={"key": "a-aaaaaa", "id": "pr-7-aaaa"})
    out = cw_index.rows([loser, winner, sib], NOW)
    assert [r["id"] for r in out] == ["pr-7-bbbb", "cmp-aaaaaaaa"] and out[1]["parent"] == {"key": "b-bbbbbb", "id": "pr-7-bbbb"}


def test_concurrent_scans_over_different_roots(tmp_path):
    for root in ("a", "b"):
        for i in range(20):
            d = tmp_path / root / f"k-{i:06d}" / "pr-1-aaaa"
            d.mkdir(parents=True)
            (d / "meta.json").write_text(json.dumps({"title": root}))
    errors = []

    def work(root):
        try:
            for _ in range(20):
                assert len(cw_index.scan(tmp_path / root)) == 20
        except Exception as ex:
            errors.append(ex)

    ts = [threading.Thread(target=work, args=(r,)) for r in "abababab"]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert errors == []
