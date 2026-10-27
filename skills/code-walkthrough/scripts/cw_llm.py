#!/usr/bin/env python3
"""HTTP client for an OpenAI-compatible chat completions endpoint: retries, auth, the
tool-calling loop and usage/cost accounting. The one place cw_run.py and cw_mcp.py talk to a
model backend.

Stdlib only, no network except the configured profile's base_url.
"""

import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import cw_store  # noqa: E402

BACKOFF_S = [1.0, 4.0]
RETRY_AFTER_MAX_S = 60
ATTEMPTS = 3

_RETRY_STATUSES = {429, 500, 502, 503, 504}
_CONTEXT_MARKERS = ("context_length_exceeded", "context length", "maximum context")
_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

CLAUDE_TOOLS = {"read_file": "Read", "grep": "Grep", "list_dir": "Glob"}
_CLAUDE_ENV_DROP = (
    "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_SESSION_ATTENDED", "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_CODE_MESSAGING_TOKEN", "CLAUDE_PID", "CLAUDE_AGENT_SDK_VERSION",
)
_CLAUDE_NOT_ON_PATH_REMEDY = (
    "install Claude Code and log in, or point the role at an openai profile, then cw_mcp.py stop"
)
_CLAUDE_AUTH_REMEDY = "run `claude` in a terminal and log in (`claude auth login`)"
_CLAUDE_RATE_REMEDY = "wait for the reset the message names, or point the role at another profile"
_CLAUDE_AUTH_MARKERS = ("not logged in", "unauthorized", "authentication required")
_CLAUDE_RATE_MARKERS = ("usage limit", "rate limit", "rate_limit")
_CLAUDE_CONTEXT_MARKERS = ("prompt is too long", "context length")

_semaphore_lock = threading.Lock()
_semaphore = threading.BoundedSemaphore(4)


def configure(max_concurrency):
    """Replace the module semaphore. Call before any chat()."""
    global _semaphore
    with _semaphore_lock:
        _semaphore = threading.BoundedSemaphore(max_concurrency)


def _current_semaphore():
    with _semaphore_lock:
        return _semaphore


class Done:
    def __init__(self, value):
        self.value = value


class LLMError(cw_store.CWError):
    def __init__(self, message, remedy=None, *, kind="error", status=None):
        super().__init__(message, remedy)
        self.kind = kind
        self.status = status


def _expand_env(value):
    def _sub(match):
        name = match.group(1)
        if name not in os.environ:
            raise LLMError(
                f"{name} is not set", kind="config",
                remedy="export VAR in the shell that starts your agent, then cw_mcp.py stop",
            )
        return os.environ[name]

    return _ENV_VAR_RE.sub(_sub, value)


def _retry_sleep(headers, attempt):
    retry_after = headers.get("Retry-After")
    if retry_after is not None:
        try:
            seconds = int(retry_after)
        except ValueError:
            seconds = None
        if seconds is not None:
            return min(seconds, RETRY_AFTER_MAX_S)
    return BACKOFF_S[attempt]


def _finish(status, raw, profile, body):
    base_url = profile["base_url"].rstrip("/")
    text_body = raw.decode("utf-8", "replace")

    if status in (401, 403):
        raise LLMError(
            f"HTTP {status} from {base_url}", kind="auth", status=status,
            remedy=f"set {profile.get('api_key_env')} in the shell that starts your agent, then cw_mcp.py stop",
        )
    if status == 404:
        raise LLMError(
            f"HTTP 404 from {base_url}", kind="notfound", status=404,
            remedy=f"check model for profile {profile.get('name')}",
        )
    if status == 400:
        lowered = text_body.lower()
        if any(marker in lowered for marker in _CONTEXT_MARKERS):
            raise LLMError(
                f"context length exceeded for {profile.get('model')}", kind="context", status=400,
                remedy=f"this batch is too big for {profile.get('model')}: point the role at a larger-context model",
            )
        raise LLMError(text_body[:300], kind="bad_request", status=400)
    if status != 200:
        raise LLMError(f"HTTP {status} from {base_url}", kind="protocol", status=status)

    try:
        parsed = json.loads(text_body)
        choices = parsed["choices"]
        message_raw = choices[0]["message"]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise LLMError("malformed reply: no choices", kind="protocol") from e

    message = {
        "role": message_raw.get("role"),
        "content": message_raw.get("content"),
        "tool_calls": message_raw.get("tool_calls"),
    }

    usage_raw = parsed.get("usage")
    if usage_raw:
        usage = {
            "prompt_tokens": usage_raw.get("prompt_tokens", 0),
            "completion_tokens": usage_raw.get("completion_tokens", 0),
            "estimated": False,
        }
    else:
        # ponytail: chars/4 estimate, not a real tokenizer; upgrade if accuracy on
        # usage-omitting providers matters.
        usage = {
            "prompt_tokens": len(json.dumps(body)) // 4,
            "completion_tokens": len(json.dumps(message)) // 4,
            "estimated": True,
        }
    return message, usage


def chat(profile, messages, tools=None, *, timeout=300, stream=False, on_chunk=None):
    if stream:
        raise NotImplementedError("streaming is phase 7")

    base_url = profile["base_url"].rstrip("/")
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}

    api_key_env = profile.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise LLMError(
                f"{api_key_env} is not set", kind="auth",
                remedy=f"set {api_key_env} in the shell that starts your agent, then cw_mcp.py stop",
            )
        headers["Authorization"] = f"Bearer {api_key}"

    for key, value in (profile.get("headers") or {}).items():
        headers[key] = _expand_env(value)

    body = {"model": profile["model"], "messages": messages}
    if tools is not None:
        body["tools"] = tools
    payload = json.dumps(body).encode()

    for attempt in range(ATTEMPTS):
        sem = _current_semaphore()
        sem.acquire()
        status = None
        raw = b""
        resp_headers = {}
        try:
            request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    status = response.status
                    raw = response.read()
                    resp_headers = dict(response.headers)
            except urllib.error.HTTPError as e:
                status = e.code
                raw = e.read()
                resp_headers = dict(e.headers or {})
        except socket.timeout as e:
            raise LLMError(
                str(e), kind="timeout",
                remedy=f"raise timeout_s in config.json or check {base_url}",
            ) from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, socket.timeout):
                raise LLMError(
                    str(e), kind="timeout",
                    remedy=f"raise timeout_s in config.json or check {base_url}",
                ) from e
            raise LLMError(str(e), kind="connect", remedy=f"nothing listening at {base_url}") from e
        except ConnectionRefusedError as e:
            raise LLMError(str(e), kind="connect", remedy=f"nothing listening at {base_url}") from e
        finally:
            sem.release()

        if status not in _RETRY_STATUSES:
            return _finish(status, raw, profile, body)

        if attempt == ATTEMPTS - 1:
            kind = "rate" if status == 429 else "server"
            raise LLMError(f"HTTP {status} from {base_url} after {ATTEMPTS} attempts", kind=kind, status=status)

        time.sleep(_retry_sleep(resp_headers, attempt))

    raise AssertionError("unreachable")  # pragma: no cover


def claude_env(profile=None):
    env = {k: v for k, v in os.environ.items() if k not in _CLAUDE_ENV_DROP}
    if profile and profile.get("config_dir"):
        env["CLAUDE_CONFIG_DIR"] = profile["config_dir"]
    return env


_NO_PUSH_PROTOCOLS = ("https", "http", "ssh", "git", "file", "ext")


def no_push_env():
    """Env additions that make every git transport fail, so nothing spawned under the agent can
    push (or fetch). git reads these as config above the repo's own."""
    keys = ["protocol.allow"] + [f"protocol.{p}.allow" for p in _NO_PUSH_PROTOCOLS]
    env = {"GIT_CONFIG_COUNT": str(len(keys))}
    for n, key in enumerate(keys):
        env[f"GIT_CONFIG_KEY_{n}"] = key
        env[f"GIT_CONFIG_VALUE_{n}"] = "never"
    return env


def _classify_claude_error(parsed, stderr, profile):
    status = parsed.get("api_error_status")
    text = f"{parsed.get('result') or ''} {stderr or ''}".lower()

    if status in (401, 403) or any(marker in text for marker in _CLAUDE_AUTH_MARKERS):
        return "auth", _CLAUDE_AUTH_REMEDY
    if status == 404 or "unrecognized_model" in (stderr or ""):
        return "notfound", f"check model for profile {profile.get('name')}"
    if status == 429 or any(marker in text for marker in _CLAUDE_RATE_MARKERS):
        return "rate", _CLAUDE_RATE_REMEDY
    if any(marker in text for marker in _CLAUDE_CONTEXT_MARKERS):
        return "context", f"this batch is too big for {profile.get('model')}: lower batch_max_lines"
    if parsed.get("subtype") == "error_max_turns":
        return "rounds", None
    return "error", None


def claude_usage(parsed):
    usage_raw = parsed.get("usage") or {}
    return {
        "prompt_tokens": (usage_raw.get("input_tokens", 0) + usage_raw.get("cache_creation_input_tokens", 0)
                           + usage_raw.get("cache_read_input_tokens", 0)),
        "completion_tokens": usage_raw.get("output_tokens", 0),
        "estimated": False,
    }


def claude_call(profile, system, prompt, *, cwd, tools, schema=None, timeout=300):
    if not cwd or not Path(cwd).is_dir():
        raise LLMError(
            "cwd is not a directory", kind="config",
            remedy="re-run /code-walkthrough on the same target",
        )

    argv = [
        "claude", "-p", "--safe-mode", "--restricted", f"--tools={tools}",
        "--strict-mcp-config", "--no-session-persistence", "--permission-prompts", "none",
        f"--model={profile['model']}", "--output-format", "json", f"--system-prompt={system}",
    ]
    if schema:
        argv.append(f"--json-schema={json.dumps(schema)}")

    sem = _current_semaphore()
    sem.acquire()
    try:
        try:
            proc = subprocess.Popen(
                argv, cwd=cwd, env=claude_env(profile), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, start_new_session=True,
            )
        except FileNotFoundError as e:
            raise LLMError(
                "claude not on PATH", kind="config", remedy=_CLAUDE_NOT_ON_PATH_REMEDY,
            ) from e
        try:
            out, err = proc.communicate(input=prompt, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate()
            raise LLMError(
                "claude process timed out", kind="timeout",
                remedy="raise timeout_s in config.json",
            ) from e
    finally:
        sem.release()

    try:
        parsed = json.loads(out)
    except ValueError as e:
        error = LLMError((err or out)[-300:], kind="protocol")
        error.usage = None
        raise error from e

    usage = claude_usage(parsed)

    if parsed.get("is_error") or proc.returncode != 0:
        kind, remedy = _classify_claude_error(parsed, err, profile)
        message = (parsed.get("result") or err or "")[:300]
        error = LLMError(message, kind=kind, remedy=remedy)
        error.usage = usage
        raise error

    return parsed, usage


# cw_mcp imports this module, so the names are literal here; test_cw_task pins them to outcome_tools.
THREAD_ALLOWED = ",".join(f"mcp__cw__{n}" for n in (
    "propose_resolve", "propose_page_edit", "propose_github_draft", "propose_task", "post_draft_comments"))


def thread_argv(profile, system, *, mcp_config, add_dir, session_id=None, resume=None):
    # --safe-mode disables MCP servers; CLAUDE.md verified not loaded under --restricted.
    argv = [
        "claude", "-p", "--restricted", "--tools=Read,Grep,Glob", "--strict-mcp-config",
        f"--mcp-config={json.dumps(mcp_config)}", f"--allowedTools={THREAD_ALLOWED}",
        "--permission-prompts", "none", f"--model={profile['model']}",
        "--output-format", "stream-json", "--verbose", "--include-partial-messages",
        f"--system-prompt={system}", f"--add-dir={add_dir}",
    ]
    return argv + (["--resume", resume] if resume else ["--session-id", session_id])


# Best-effort, not a sandbox: these deny rules and the transport block in no_push_env can both be bypassed.
FULL_DISALLOWED = ",".join((
    "Bash(git push:*)", "Bash(rtk git push:*)", "Bash(git -c:*)", "Bash(git --no-pager push:*)",
    "Bash(env:*)", "Bash(*GIT_CONFIG*)", "Bash(gh pr create:*)", "Bash(gh pr review:*)",
    "Bash(gh pr merge:*)", "Bash(gh pr comment:*)", "Bash(gh api:*)",
    # gh issue view stays allowed so a ticket on GitHub can be read; everything that changes an issue is not.
    *(f"Bash(gh issue {v}:*)" for v in ("create", "edit", "comment", "close", "reopen", "delete", "transfer",
                                         "lock", "unlock", "pin", "unpin", "develop")),
    "mcp__code-walkthrough", "mcp__plugin_code-walkthrough_code-walkthrough",
    "WebFetch", "WebSearch",
))


def _under_tmp(path):
    real = os.path.realpath(path)
    return real == "/tmp" or real.startswith("/tmp/")


def _docker_sockets():
    # --ro-bind needs the destination to exist on the read-only root, so only hide sockets that do.
    found = {os.path.realpath(p) for p in ("/var/run/docker.sock", "/run/docker.sock") if os.path.exists(p)}
    return sorted(found)


def _secret_files():
    paths = (cw_store.token_path(), cw_store.server_json_path(), cw_store.config_path())
    return sorted({os.path.realpath(p) for p in paths if os.path.exists(p)})


AGENT_MODELS = ("haiku", "sonnet", "opus")


def _is_setup(p):
    p = Path(p)
    return p.is_dir() and ((p / "settings.json").exists() or (p / ".credentials.json").exists())


def _setups_config(config):
    raw = (config if config is not None else cw_store.load_config()).get("setups")
    return raw if isinstance(raw, dict) else {}


def claude_profiles(home=None, config=None, include_hidden=False):
    """Claude Code installs on this machine: ~/.claude and ~/.claude-*, the daemon's own
    CLAUDE_CONFIG_DIR and folders the user added, each holding settings or credentials.
    [{dir, label, named, hidden}]; label is the user's name for it, else the folder's."""
    home = Path(home or Path.home())
    setups = _setups_config(config)
    found = {home / ".claude"} | set(home.glob(".claude-*"))
    if os.environ.get("CLAUDE_CONFIG_DIR"):
        found.add(Path(os.environ["CLAUDE_CONFIG_DIR"]))
    found |= {Path(d) for d, v in setups.items() if isinstance(v, dict) and v.get("added")}
    out = []
    for p in sorted({q.resolve() for q in found if _is_setup(q)}, key=lambda p: (p.name != ".claude", str(p))):
        conf = setups.get(str(p)) if isinstance(setups.get(str(p)), dict) else {}
        hidden = bool(conf.get("hidden"))
        if hidden and not include_hidden:
            continue
        name = conf.get("name") if isinstance(conf.get("name"), str) and conf.get("name").strip() else None
        out.append({"dir": str(p), "label": name or p.name,
                    "named": bool(name), "hidden": hidden})
    return out


def setup_label(config_dir, config=None):
    return next((p["label"] for p in claude_profiles(config=config, include_hidden=True) if p["dir"] == config_dir), None)


_SETUP_INFO = {}
_SETUP_INFO_TTL = 600


def setup_skills(config_dir):
    """Skill and slash-command names this install offers: its own skills/ and commands/, and the
    skills of every plugin its settings enable. Names only, without the plugin prefix."""
    root = Path(config_dir)
    names = {p.parent.name for p in root.glob("skills/*/SKILL.md")}
    names |= {p.stem for p in root.glob("commands/*.md")}
    settings = cw_store.read_json(root / "settings.json", default=None) or {}
    installed = (cw_store.read_json(root / "plugins" / "installed_plugins.json", default=None) or {}).get("plugins") or {}
    for key, on in (settings.get("enabledPlugins") or {}).items():
        for inst in installed.get(key) or [] if on else []:
            base = Path(inst.get("installPath") or "")
            names |= {p.parent.name for p in base.glob("skills/*/SKILL.md")}
            names |= {p.stem for p in base.glob("commands/*.md")}
    return sorted(names)


def setup_info(config_dir, run=subprocess.run, clock=time.monotonic):
    """{email, org, plan, logged_in, skills} for one install, from `claude auth status`.
    ponytail: cached for 10 minutes per dir; a fresh login shows after that or a daemon restart."""
    hit = _SETUP_INFO.get(config_dir)
    if hit and clock() - hit[0] < _SETUP_INFO_TTL:
        return hit[1]
    info = {"email": None, "org": None, "plan": None, "logged_in": False}
    try:
        out = run(["claude", "auth", "status", "--json"], capture_output=True, text=True, timeout=15,
                  env={**claude_env(), "CLAUDE_CONFIG_DIR": config_dir})
        data = json.loads(out.stdout or "{}")
        info = {"email": data.get("email"), "org": data.get("orgName"), "plan": data.get("subscriptionType"),
                "logged_in": bool(data.get("loggedIn"))}
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    info["skills"] = setup_skills(config_dir)
    _SETUP_INFO[config_dir] = (clock(), info)
    return info


def agents_payload(config, selected=None):
    """What a picker shows: every install with who it is signed in as and its skills, the models,
    the current choice and whether the first-run naming step is due."""
    everything = claude_profiles(config=config, include_hidden=True)
    setups = []
    for p in everything:
        info = setup_info(p["dir"])
        setups.append({"config_dir": p["dir"], "label": p["label"], "named": p["named"], "hidden": p["hidden"],
                       "folder": p["dir"].replace(str(Path.home()), "~", 1), "email": info["email"],
                       "org": info["org"], "plan": info["plan"], "logged_in": info["logged_in"],
                       "skills_count": len(info["skills"])})
    return {"setups": setups, "models": list(AGENT_MODELS), "selected": selected,
            "setup_due": len(everything) > 1 and not config.get("setups_seen")}


APPROVABLE_TOOL = re.compile(r"mcp__[A-Za-z0-9_.-]+")


def effective_config_dir(config_dir=None):
    return str(Path(config_dir) if config_dir else claude_dir(Path.home()).resolve())


def approved_tools(config, config_dir=None):
    """MCP tools the user allowed for Full turns on this install, from a "Needs permission" prompt."""
    raw = (config.get("approved_tools") or {}).get(effective_config_dir(config_dir)) or []
    return [t for t in raw if isinstance(t, str) and APPROVABLE_TOOL.fullmatch(t)]


_SLASH_RE = re.compile(r"(?<![\w/])/([a-z][\w-]*(?::[\w-]+)?)")


def missing_skill(text, config_dir, config=None):
    """{skill, alt, alt_label} when text names a /skill this install lacks and another visible one has."""
    have = set(setup_info(config_dir)["skills"])
    for name in (m.split(":")[-1] for m in _SLASH_RE.findall(text or "")):
        if name in have:
            continue
        for p in claude_profiles(config=config):
            if p["dir"] != config_dir and name in setup_info(p["dir"])["skills"]:
                return {"skill": name, "alt": p["dir"], "alt_label": p["label"]}
    return None


def valid_agent(agent, profiles=None):
    """{config_dir, model} when agent names a detected, visible install and a known model, else None.
    The dir ends up in an env var and a bwrap bind, so it is never taken from the request as is."""
    if not isinstance(agent, dict) or agent.get("model") not in AGENT_MODELS:
        return None
    dirs = {p["dir"] for p in (claude_profiles() if profiles is None else profiles)}
    return {"config_dir": agent["config_dir"], "model": agent["model"]} if agent.get("config_dir") in dirs else None


def claude_dir(home, config_dir=None):
    # The Claude Code config dir an agent run uses: the repo's configured one, else CLAUDE_CONFIG_DIR, else ~/.claude.
    return Path(config_dir or os.environ.get("CLAUDE_CONFIG_DIR") or Path(home) / ".claude")


def project_dir(home, cwd, config_dir=None):
    # Claude Code names a cwd's state dir after the path with every non-alphanumeric character replaced by '-'.
    # Resolved: a second profile often symlinks projects/ to the first, and bwrap cannot create a mount point through a symlink.
    return (claude_dir(home, config_dir) / "projects").resolve() / re.sub(r"[^A-Za-z0-9]", "-", str(cwd))


def confine_prefix(worktree, repo_root, home, turns_dir, sockets=None, exists=os.path.exists,
                   secrets=None, config_dir=None):
    """bwrap argv that makes everything but the worktree and turns dir read-only for a Full turn.
    The main repo root is an overlay: agent writes there succeed and are discarded, the daemon
    commits the worktree on the host afterwards. ~/.claude stays read-only except the agent cwd's own
    project dir, so the agent cannot plant hooks, settings, CLAUDE.md or auto-memory the user's other
    sessions would load; /run stays visible for DNS."""
    worktree, repo_root, home, turns_dir = map(str, (worktree, repo_root, home, turns_dir))
    walkthrough = str(Path(turns_dir).parent)
    if any(_under_tmp(p) for p in (worktree, repo_root, walkthrough)):
        raise cw_store.CWError("Full access cannot confine a walkthrough stored under /tmp",
                               remedy="keep CODE_WALKTHROUGH_HOME and the repo outside /tmp")
    prefix = [
        "bwrap", "--ro-bind", "/", "/", "--dev-bind", "/dev", "/dev", "--proc", "/proc",
        "--die-with-parent", "--unshare-pid", "--new-session", "--cap-drop", "ALL",
        "--tmpfs", "/tmp",
    ]
    for sock in _docker_sockets() if sockets is None else sockets:
        prefix += ["--ro-bind", "/dev/null", str(sock)]
    prefix += ["--overlay-src", repo_root, "--tmp-overlay", repo_root]
    # After the overlay, which may sit on $HOME; tmpfs needs the mount point to exist on the read-only root.
    for hidden in ("/run/user", f"{home}/.ssh", f"{home}/.gnupg"):
        if exists(hidden):
            prefix += ["--tmpfs", hidden]
    prefix += [
        "--ro-bind", walkthrough, walkthrough,
    ]
    for secret in _secret_files() if secrets is None else secrets:
        prefix += ["--ro-bind", "/dev/null", str(secret)]
    prefix += [
        "--bind", worktree, worktree, "--bind", turns_dir, turns_dir,
    ]
    cdir = claude_dir(home, config_dir)
    writable = [str(project_dir(home, worktree, config_dir))] + [str((cdir / rel).resolve()) for rel in (
        "session-env", "sessions", ".credentials.json")] + [f"{home}/.local/share/rtk"]
    for path in writable:
        prefix += ["--bind-try", path, path]
    return prefix + ["--chdir", worktree, "--"]


_confined = {"ok": False, "failed_at": None}
_confined_lock = threading.Lock()
CONFINEMENT_RETRY_S = 60


def confinement_available(run=subprocess.run, clock=time.monotonic):
    # A failed probe may be transient (timeout), so it is retried after a minute; success is final.
    if _confined["ok"]:
        return True
    with _confined_lock:
        if _confined["ok"]:
            return True
        failed_at = _confined["failed_at"]
        if failed_at is not None and clock() - failed_at < CONFINEMENT_RETRY_S:
            return False
        _confined["ok"] = _probe_confinement(run)
        _confined["failed_at"] = None if _confined["ok"] else clock()
        return _confined["ok"]


def _probe_confinement(run):
    # Runs the real prefix so the probe cannot drift from what a turn executes.
    cache = Path.home() / ".cache"
    with tempfile.TemporaryDirectory(dir=cache if cache.is_dir() else None) as t:
        t = Path(t)
        wt, root, turns = t / "w" / "agent", t / "repo", t / "w" / "turns"
        for p in (wt, root / ".git", turns):
            p.mkdir(parents=True)
        try:
            prefix = confine_prefix(wt, root, Path.home(), turns)
            return run(prefix + ["true"], capture_output=True, timeout=10).returncode == 0
        except (OSError, subprocess.SubprocessError, cw_store.CWError):
            return False


def git_common_dir(worktree):
    out = subprocess.run(["git", "-C", str(worktree), "rev-parse", "--git-common-dir"],
                         capture_output=True, text=True, timeout=10)
    if out.returncode != 0:
        raise OSError(out.stderr.strip() or "git rev-parse failed")
    return (Path(worktree) / out.stdout.strip()).resolve()


def overlay_root(common_dir):
    common_dir = Path(common_dir)
    return common_dir.parent if common_dir.name == ".git" else common_dir


def full_argv(profile, system, *, mcp_config, allowed, add_dir, session_id=None, resume=None):
    # No --restricted, --strict-mcp-config or --tools on purpose: the agent is the user's own Claude Code
    # and --setting-sources user supplies its permissions and MCP servers.
    argv = [
        "claude", "-p", "--setting-sources", "user", f"--mcp-config={json.dumps(mcp_config)}",
        f"--allowedTools={allowed}", f"--disallowedTools={FULL_DISALLOWED}",
        "--permission-mode", "acceptEdits", "--permission-prompts", "none",
        f"--model={profile['model']}", "--output-format", "stream-json", "--verbose",
        "--include-partial-messages", f"--append-system-prompt={system}", f"--add-dir={add_dir}",
    ]
    return argv + (["--resume", resume] if resume else ["--session-id", session_id])


def task_argv(profile, system, *, mcp_config):
    # Fresh session: sessions are keyed by cwd, and a task runs in its own worktree. Without
    # --safe-mode (it disables MCP), --restricted ignores the PR's .claude settings and hooks and
    # --strict-mcp-config its .mcp.json and the user's own servers; only run_check is allowed.
    return [
        "claude", "-p", "--strict-mcp-config", "--restricted",
        f"--mcp-config={json.dumps(mcp_config)}", "--allowedTools=mcp__cw__run_check",
        "--tools=Read,Grep,Glob,Edit,Write",
        "--permission-mode", "acceptEdits", "--permission-prompts", "none",
        f"--model={profile['model']}", "--output-format", "stream-json", "--verbose",
        "--include-partial-messages", "--no-session-persistence",
        f"--system-prompt={system}",
    ]


def claude_stream(argv, prompt, *, cwd, timeout, on_line, on_spawn=None, env=None):
    """Run argv, feeding each parsed stdout JSON line to on_line. Returns (returncode,
    stderr_tail); a timeout raises LLMError(kind="timeout") like claude_call. env overlays
    claude_env()."""
    sem = _current_semaphore()
    sem.acquire()
    try:
        try:
            proc = subprocess.Popen(
                argv, cwd=cwd, env={**claude_env(), **(env or {})}, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
            )
        except FileNotFoundError as e:
            raise LLMError(
                "claude not on PATH", kind="config", remedy=_CLAUDE_NOT_ON_PATH_REMEDY,
            ) from e

        def kill():
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        err_buf = []

        def drain():
            size = 0
            for chunk in iter(lambda: proc.stderr.read(4096), ""):
                err_buf.append(chunk)
                size += len(chunk)
                while size > 65536 and len(err_buf) > 1:
                    size -= len(err_buf.pop(0))

        def feed():
            try:
                proc.stdin.write(prompt)
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass

        timed_out = threading.Event()

        def expire():
            timed_out.set()
            kill()

        timer = threading.Timer(timeout, expire)
        threads = [threading.Thread(target=drain, daemon=True), threading.Thread(target=feed, daemon=True)]
        try:
            if on_spawn:
                on_spawn(proc)
            timer.start()
            for t in threads:
                t.start()
            for line in iter(proc.stdout.readline, ""):
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    on_line(obj)
        except BaseException:
            kill()
            raise
        finally:
            timer.cancel()
            proc.wait()
            for t in threads:
                t.join(timeout=5)
    finally:
        sem.release()

    if timed_out.is_set():
        raise LLMError(
            "claude process timed out", kind="timeout", remedy="raise timeout_s in config.json",
        )
    return proc.returncode, "".join(err_buf)[-65536:]


def _render_claude_prompt(messages):
    parts = []
    for message in messages:
        if message.get("role") == "assistant":
            parts.append(f"Your previous answer:\n```json\n{message.get('content')}\n```")
        else:
            parts.append(message.get("content") or "")
    return "\n\n".join(parts)


_JSON_TYPES = {
    "array": list, "object": dict, "null": type(None), "string": str,
    "boolean": bool, "integer": int, "number": (int, float),
}


def _allowed_types(prop):
    types = set()
    for sub in [prop] + list(prop.get("anyOf") or []) + list(prop.get("oneOf") or []):
        t = sub.get("type")
        types.update(t if isinstance(t, list) else [t] if t else [])
    return types


def _type_ok(value, types):
    for t in types:
        py = _JSON_TYPES.get(t)
        if py and isinstance(value, py) and not (isinstance(value, bool) and t in ("integer", "number")):
            return True
    return False


def coerce_structured(structured, schema):
    """claude-code sometimes returns non-string fields as JSON-encoded strings; decode those
    top-level values back when the schema forbids a string there and the decode fits."""
    props = (schema or {}).get("properties") or {}
    out = dict(structured)
    for key, value in structured.items():
        types = _allowed_types(props.get(key) or {})
        if not isinstance(value, str) or not types:
            continue
        if "string" in types:
            if value == "null" and "null" in types:
                out[key] = None
            continue
        try:
            parsed = json.loads(value)
        except ValueError:
            continue
        if _type_ok(parsed, types):
            out[key] = parsed
    return out


def _run_claude_tools(profile, messages, tools, handlers, *, max_rounds, timeout, on_usage, cwd):
    # ponytail: max_tokens/nudge are HTTP-path knobs; each claude process runs its own tool
    # loop and timeout_s is the only budget here.
    submit = [t for t in tools if t["function"]["name"] not in CLAUDE_TOOLS]
    if len(submit) > 1:
        raise LLMError("claude-code backend supports at most one submit tool", kind="config")
    tools_arg = ",".join(
        CLAUDE_TOOLS[t["function"]["name"]] for t in tools if t["function"]["name"] in CLAUDE_TOOLS
    )
    schema = (submit[0]["function"]["parameters"] or None) if submit else None

    system = messages[0]["content"] + (
        "\n\nIn this session read_file, grep and list_dir are the Read, Grep and Glob tools, "
        "rooted at the current directory (the repo at head)."
    )
    if submit:
        system += (
            f"\n\nCalling `{submit[0]['function']['name']}` means returning its arguments as "
            "your final structured output; there is no tool by that name."
        )

    for _round in range(max_rounds):
        prompt = _render_claude_prompt(messages[1:])
        try:
            out, usage = claude_call(
                profile, system, prompt, cwd=cwd, tools=tools_arg, schema=schema, timeout=timeout,
            )
        except LLMError as e:
            if on_usage and getattr(e, "usage", None):
                on_usage(e.usage)
            raise
        if on_usage:
            on_usage(usage)

        if schema is None:
            result = out.get("result") or ""
            messages.append({"role": "assistant", "content": result})
            return result

        structured = out.get("structured_output")
        if not isinstance(structured, dict):
            raise LLMError("claude-code reply had no structured output", kind="protocol")
        structured = coerce_structured(structured, schema)
        messages.append({
            "role": "assistant", "content": json.dumps(structured),
            "permission_denials": out.get("permission_denials"),
        })
        result = handlers[submit[0]["function"]["name"]](structured)
        if isinstance(result, Done):
            return result.value
        messages.append({"role": "user", "content": result})

    raise LLMError("tool loop exceeded max_rounds", kind="rounds")


def run_tools(profile, messages, tools, handlers, *, max_rounds=10, max_tokens=200000,
              timeout=300, on_usage=None, nudge="Reply only by calling one of the tools.", cwd=None):
    if profile.get("kind") == "claude-code":
        return _run_claude_tools(
            profile, messages, tools, handlers, max_rounds=max_rounds, timeout=timeout,
            on_usage=on_usage, cwd=cwd,
        )
    total_tokens = 0
    for _round in range(max_rounds):
        message, usage = chat(profile, messages, tools, timeout=timeout)
        if on_usage:
            on_usage(usage)
        messages.append(message)

        total_tokens += usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        if total_tokens > max_tokens:
            raise LLMError(
                "conversation exceeded max_conversation_tokens", kind="tokens",
                remedy="raise max_conversation_tokens in config.json",
            )

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            if nudge is None:
                return message.get("content") or ""
            messages.append({"role": "user", "content": nudge})
            continue

        for call in tool_calls:
            call_id = call.get("id")
            name = call["function"]["name"]
            arguments_raw = call["function"].get("arguments") or "{}"
            try:
                arguments = json.loads(arguments_raw)
            except ValueError as e:
                messages.append({"role": "tool", "tool_call_id": call_id,
                                  "content": f"invalid JSON arguments: {e}"})
                continue

            handler = handlers.get(name)
            if handler is None:
                messages.append({"role": "tool", "tool_call_id": call_id,
                                  "content": f"unknown tool {name}"})
                continue

            result = handler(arguments)
            if isinstance(result, Done):
                messages.append({"role": "tool", "tool_call_id": call_id, "content": "accepted"})
                return result.value
            messages.append({"role": "tool", "tool_call_id": call_id, "content": result})

    raise LLMError("tool loop exceeded max_rounds", kind="rounds")


def cost_usd(profile, usage):
    pricing = profile.get("price_per_mtok")
    if not pricing:
        return None
    prompt_cost = usage.get("prompt_tokens", 0) * pricing.get("input", 0)
    completion_cost = usage.get("completion_tokens", 0) * pricing.get("output", 0)
    return (prompt_cost + completion_cost) / 1e6


def check(profile, *, timeout=60):
    if profile.get("kind") == "claude-code":
        try:
            auth = subprocess.run(
                ["claude", "auth", "status", "--json"], env=claude_env(),
                capture_output=True, text=True, timeout=timeout,
            )
        except FileNotFoundError as e:
            raise LLMError(
                "claude not on PATH", kind="config", remedy=_CLAUDE_NOT_ON_PATH_REMEDY,
            ) from e
        except subprocess.TimeoutExpired as e:
            raise LLMError(
                "claude auth status timed out", kind="timeout",
                remedy="raise timeout_s in config.json",
            ) from e
        try:
            auth_data = json.loads(auth.stdout)
        except ValueError:
            auth_data = {}
        if not auth_data.get("loggedIn"):
            raise LLMError("claude is not logged in", kind="auth", remedy=_CLAUDE_AUTH_REMEDY)

        schema = {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}
        with tempfile.TemporaryDirectory() as tmp:
            out, _usage = claude_call(
                profile, "", "Return ok: true.", cwd=tmp, tools="", schema=schema, timeout=timeout,
            )
        if not isinstance(out.get("structured_output"), dict):
            raise LLMError(
                "model did not reply with structured output", kind="no_tools",
                remedy="point this role at a model that supports structured output",
            )
        return

    messages = [{"role": "user", "content": "Call the ping tool now."}]
    tools = [{"type": "function", "function": {"name": "ping", "description": "ping",
                                                "parameters": {"type": "object", "properties": {}}}}]
    message, _usage = chat(profile, messages, tools, timeout=timeout)
    if not message.get("tool_calls"):
        raise LLMError(
            "model did not reply with a tool call", kind="no_tools",
            remedy="point this role at a model that supports tool calls",
        )
