#!/usr/bin/env python3
"""
Nizo browser sidecar — a headless Playwright (Chromium) driver behind a tiny HTTP API.

Why a sidecar (not in-JVM): mirrors the voice/music sidecar pattern, keeps the ~150 MB browser
driver out of the Java uber-jar, and lets the browser's lifecycle (persistent context, sessions)
live independently of the agent process. Nizo's Java `BrowserTool` calls this over loopback.

This is what `web_fetch` can't do: render JavaScript-heavy SPAs, click, fill, and read dynamic
content across a multi-step session — the capability that unlocks Coles-style cart assembly and
the long tail of app-like sites.

Security posture (defence in depth, NOT a hardened sandbox):
  * Binds to 127.0.0.1 only.
  * REFUSES to type into password / payment fields (type=password, autocomplete cc-*, or a
    name/id that looks like card/cvv/expiry) — returns needs_human instead.
  * REFUSES to click commit-purchase controls ("place order", "pay now", "confirm purchase",
    "buy now") — returns needs_human so the human completes checkout. Navigation to a cart or
    "proceed to checkout" page is allowed; committing money is not.
  * Sessions idle-evict after BROWSER_SESSION_TTL_S.

Run:  uvicorn browser_sidecar:app --host 127.0.0.1 --port 7781
Env:  BROWSER_HEADLESS=1 (default), BROWSER_SESSION_TTL_S=900, BROWSER_NAV_TIMEOUT_MS=30000
"""
import asyncio
import os
import re
import time
import uuid
from typing import Dict, Optional

import logging
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    from playwright.async_api import async_playwright, Page, BrowserContext
except Exception as e:  # pragma: no cover - import guard for clearer error on a box without playwright
    raise RuntimeError("playwright not installed — run: pip install playwright && playwright install chromium") from e

HEADLESS = os.getenv("BROWSER_HEADLESS", "1") not in ("0", "false", "False")
SESSION_TTL_S = int(os.getenv("BROWSER_SESSION_TTL_S", "900"))
NAV_TIMEOUT_MS = int(os.getenv("BROWSER_NAV_TIMEOUT_MS", "30000"))
MAX_TEXT_CHARS = int(os.getenv("BROWSER_MAX_TEXT_CHARS", "12000"))

# Field/selector patterns we refuse to TYPE into — entering secrets is the human's job.
_SECRET_FIELD = re.compile(r"pass\s*word|passwd|\bpin\b|otp|2fa|card\s*number|cardnum|cc-?num|"
                           r"cvv|cvc|security\s*code|expir|card\s*holder", re.I)
# Visible-label patterns we refuse to CLICK — these commit money / place an order.
_COMMIT_CLICK = re.compile(r"place\s+order|pay\s+now|pay\s+\$|confirm\s+(?:and\s+)?pay|"
                           r"confirm\s+order|complete\s+purchase|buy\s+now|submit\s+payment", re.I)

# Cookie/consent dismissal — PRIVACY-PRESERVING: we click reject / necessary-only / close only.
# We deliberately NEVER auto-"accept all" — that's a consent choice the user owns; if only an
# accept-all exists we leave the banner and report it.
_CONSENT_SELECTORS = ["#onetrust-reject-all-handler", "[data-testid=reject-all]",
                      "button[aria-label*='reject' i]", "button[aria-label*='decline' i]"]
_CONSENT_TEXTS = ["reject all", "reject non-essential", "necessary only", "only necessary",
                  "essential only", "decline all", "decline", "continue without accepting", "reject"]


async def _dismiss_consent(page) -> str:
    """Best-effort, privacy-first dismissal of a cookie/consent overlay. Returns a note or ''."""
    for sel in _CONSENT_SELECTORS:
        try:
            el = await page.query_selector(sel)
            if el and await el.is_visible():
                await el.click(timeout=2000)
                await page.wait_for_timeout(400)
                return f"consent dismissed via {sel}"
        except Exception:
            pass
    for txt in _CONSENT_TEXTS:
        try:
            loc = page.get_by_role("button", name=re.compile(txt, re.I))
            if await loc.count() > 0 and await loc.first.is_visible():
                await loc.first.click(timeout=2000)
                await page.wait_for_timeout(400)
                return f"consent dismissed via '{txt}'"
        except Exception:
            pass
    return ""

app = FastAPI()


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logging.error("422 on %s | body=%r | errors=%s", request.url.path, body[:600], exc.errors())
    return JSONResponse(status_code=422, content={"ok": False, "error": "validation: " + str(exc.errors())[:400]})


class Session:
    def __init__(self, ctx: BrowserContext, page: Page):
        self.ctx = ctx
        self.page = page
        self.last_used = time.time()


class _State:
    def __init__(self):
        self.pw = None
        self.browser = None
        self.sessions: Dict[str, Session] = {}
        self.lock = asyncio.Lock()

    async def ensure_browser(self):
        if self.browser is None:
            self.pw = await async_playwright().start()
            self.browser = await self.pw.chromium.launch(headless=HEADLESS, args=["--no-sandbox"])

    async def new_session(self) -> str:
        await self.ensure_browser()
        ctx = await self.browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/126.0 Safari/537.36",
            viewport={"width": 1280, "height": 900},
        )
        page = await ctx.new_page()
        sid = uuid.uuid4().hex[:12]
        self.sessions[sid] = Session(ctx, page)
        await self._evict_idle()
        return sid

    async def get(self, sid: str) -> Optional[Session]:
        s = self.sessions.get(sid)
        if s:
            s.last_used = time.time()
        return s

    async def _evict_idle(self):
        now = time.time()
        for sid in [k for k, v in self.sessions.items() if now - v.last_used > SESSION_TTL_S]:
            await self._close(sid)

    async def _close(self, sid: str):
        s = self.sessions.pop(sid, None)
        if s:
            try:
                await s.ctx.close()
            except Exception:
                pass


STATE = _State()


class ActReq(BaseModel):
    sessionId: Optional[str] = None
    action: str
    url: Optional[str] = None
    selector: Optional[str] = None
    text: Optional[str] = None
    submit: Optional[bool] = False


@app.get("/health")
async def health():
    return {"status": "ok", "sessions": len(STATE.sessions), "headless": HEADLESS}


@app.post("/session")
async def session():
    async with STATE.lock:
        sid = await STATE.new_session()
    return {"sessionId": sid}


async def _page_state(page: Page, include_links: bool = True) -> dict:
    title = await page.title()
    # innerText of body gives the *rendered* visible text (post-JS) — the whole point vs web_fetch.
    try:
        text = await page.inner_text("body", timeout=3000)
    except Exception:
        text = await page.content()
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    truncated = len(text) > MAX_TEXT_CHARS
    out = {"url": page.url, "title": title,
           "text": text[:MAX_TEXT_CHARS] + ("\n…[truncated]" if truncated else "")}
    if include_links:
        links = await page.eval_on_selector_all(
            "a[href]",
            "els => els.slice(0,40).map(e => ({t:(e.innerText||'').trim().slice(0,60), h:e.href}))"
            ".filter(l => l.t)")
        out["links"] = links
        # Interactive controls — buttons / inputs / selects with a usable selector + label, so the
        # agent can target a search box or button precisely instead of guessing. This is what makes
        # clicking/typing on a JS SPA (e.g. Coles search) actually work.
        controls = await page.eval_on_selector_all(
            "button, input:not([type=hidden]), textarea, select, [role=button]",
            "els => els.slice(0,30).map(e => {"
            " const lbl=(e.innerText||e.value||e.placeholder||e.getAttribute('aria-label')||e.name||'').trim().slice(0,50);"
            " let sel=''; if(e.id){sel='#'+(window.CSS&&CSS.escape?CSS.escape(e.id):e.id);}"
            " else if(e.name){sel=e.tagName.toLowerCase()+'[name=\\\"'+e.name+'\\\"]';}"
            " return {tag:e.tagName.toLowerCase(), type:e.type||'', label:lbl, sel};"
            "}).filter(c => c.label || c.sel)")
        out["controls"] = controls
    return out


@app.post("/act")
async def act(req: ActReq):
    action = (req.action or "").lower().strip()

    if action == "goto" and not req.sessionId:
        async with STATE.lock:
            req.sessionId = await STATE.new_session()

    s = await STATE.get(req.sessionId) if req.sessionId else None
    if s is None:
        return {"ok": False, "error": "no such session — call /session or use action=goto first"}
    page = s.page

    try:
        if action == "goto":
            if not req.url:
                return {"ok": False, "error": "goto requires url"}
            await page.goto(req.url, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
            await page.wait_for_timeout(800)  # let late JS settle
            consent = await _dismiss_consent(page)   # privacy-first cookie/consent dismissal
            if consent:
                await page.wait_for_timeout(500)
            st = await _page_state(page)
            if consent:
                st["note"] = consent
            return {"ok": True, "sessionId": req.sessionId, **st}

        if action in ("read", "state"):
            return {"ok": True, "sessionId": req.sessionId, **await _page_state(page)}

        if action == "click":
            target = req.selector or req.text
            if not target:
                return {"ok": False, "error": "click requires selector or text"}
            # Refuse to commit a purchase — that stays with the human.
            label = req.text or target
            if _COMMIT_CLICK.search(label):
                return {"ok": False, "needs_human": True,
                        "error": f"refusing to click a commit-purchase control ({label!r}); "
                                 f"the human must complete checkout / payment"}
            if req.text and not req.selector:
                await page.get_by_text(req.text, exact=False).first.click(timeout=NAV_TIMEOUT_MS)
            else:
                await page.click(req.selector, timeout=NAV_TIMEOUT_MS)
            await page.wait_for_timeout(800)
            return {"ok": True, "sessionId": req.sessionId, **await _page_state(page)}

        if action == "type":
            if not req.selector:
                return {"ok": False, "error": "type requires selector"}
            # Refuse secret/payment fields. Inspect the element's attributes.
            attrs = await page.eval_on_selector(
                req.selector,
                "e => ({type:e.type||'', name:e.name||'', id:e.id||'', ac:e.autocomplete||''})") \
                if await page.query_selector(req.selector) else {}
            blob = " ".join(str(attrs.get(k, "")) for k in ("type", "name", "id", "ac"))
            if str(attrs.get("type", "")).lower() == "password" or _SECRET_FIELD.search(blob) \
                    or str(attrs.get("ac", "")).lower().startswith("cc-"):
                return {"ok": False, "needs_human": True,
                        "error": "refusing to type into a password/payment field; the human enters secrets"}
            await page.fill(req.selector, req.text or "", timeout=NAV_TIMEOUT_MS)
            if req.submit:
                await page.press(req.selector, "Enter")
                await page.wait_for_timeout(800)
            return {"ok": True, "sessionId": req.sessionId, **await _page_state(page)}

        if action == "wait":
            try:
                if req.selector:
                    await page.wait_for_selector(req.selector, timeout=NAV_TIMEOUT_MS)
                else:
                    await page.wait_for_timeout(1500)
            except Exception as e:
                return {"ok": False, "sessionId": req.sessionId,
                        "error": f"wait: {type(e).__name__}: {e}", **await _page_state(page)}
            return {"ok": True, "sessionId": req.sessionId, **await _page_state(page)}

        if action == "dismiss":
            note = await _dismiss_consent(page)
            st = await _page_state(page)
            st["note"] = note or "no privacy-preserving consent control found"
            return {"ok": True, "sessionId": req.sessionId, **st}

        if action in ("back", "forward"):
            await (page.go_back() if action == "back" else page.go_forward())
            await page.wait_for_timeout(500)
            return {"ok": True, "sessionId": req.sessionId, **await _page_state(page)}

        if action == "close":
            async with STATE.lock:
                await STATE._close(req.sessionId)
            return {"ok": True, "closed": req.sessionId}

        return {"ok": False, "error": f"unknown action {action!r}"}
    except Exception as e:
        return {"ok": False, "sessionId": req.sessionId, "error": f"{type(e).__name__}: {e}"}
