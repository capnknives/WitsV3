"""
Web Search Tool for WitsV3.

Provides *real* web search results (title, URL, snippet) via a provider
fallback chain, so the tool works with no API key at all and gets better when
one is supplied:

    1. Tavily        (if TAVILY_API_KEY set)  — LLM-optimized results + answer
    2. Brave Search  (if BRAVE_SEARCH_API_KEY set) — high-quality SERP
    3. DuckDuckGo HTML endpoint (keyless) — scrapes real organic results
    4. DuckDuckGo Lite endpoint (keyless) — lighter fallback, dodges bot walls
    5. DuckDuckGo Instant Answer API (keyless) — last resort (disambiguation only)

Why the rewrite: the old implementation used ONLY the DuckDuckGo Instant
Answer API (api.duckduckgo.com). That endpoint is not a general web search —
it returns Wikipedia-style "RelatedTopics" for a tiny set of queries and 202s
under load — so most real searches came back empty or blocked. The HTML/Lite
endpoints return the actual organic result list.

Configuration lives under `web_search:` in config.yaml (provider, max_results,
timeout, region, safesearch). API keys come from the environment (.env), never
config.yaml.
"""

import asyncio
import html
import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import aiohttp
from bs4 import BeautifulSoup

from core.base_tool import BaseTool

logger = logging.getLogger(__name__)

# A realistic desktop User-Agent — the keyless DuckDuckGo endpoints return 202
# (bot wall) to the default aiohttp UA. This makes them serve real results.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

_SAFESEARCH_VQD = {"off": "-2", "moderate": "-1", "strict": "1"}


class WebSearchTool(BaseTool):
    """Multi-provider web search with a keyless-friendly fallback chain."""

    def __init__(self):
        super().__init__(
            name="web_search",
            description=(
                "Search the live web and return a list of results (title, link, "
                "snippet). Uses Tavily or Brave if an API key is configured, "
                "otherwise DuckDuckGo. Use this for current events, facts, and "
                "anything not in local memory or documents."
            ),
        )
        # Populated by set_dependencies(); safe defaults until then.
        self.config = None

    def set_dependencies(self, config, llm_interface=None, memory_manager=None, **kwargs) -> None:
        """Called by WitsV3System at startup to share the live config."""
        self.config = config

    # --- config helpers -------------------------------------------------

    def _settings(self):
        return getattr(self.config, "web_search", None) if self.config else None

    def _get(self, attr: str, default: Any) -> Any:
        settings = self._settings()
        return getattr(settings, attr, default) if settings is not None else default

    def _tavily_key(self) -> str:
        return self._get("tavily_api_key", "") or os.getenv("TAVILY_API_KEY", "")

    def _brave_key(self) -> str:
        return self._get("brave_api_key", "") or os.getenv("BRAVE_SEARCH_API_KEY", "")

    # --- entry point ----------------------------------------------------

    async def execute(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Run a web search.

        In "auto" mode with both keys set, Tavily and Brave are queried
        concurrently and their results merged, so the caller gets one AI
        summary (from Tavily) plus independent sources from both engines to
        cross-check it against. Returns {success, provider, results:
        [{title, link, snippet}], answer?, answer_provider?, error?}.
        """
        query = (query or "").strip()
        if not query:
            return {"success": False, "error": "Empty search query", "results": []}

        provider = str(self._get("provider", "auto")).lower()
        max_results = max(1, int(max_results or self._get("max_results", 5)))

        if provider == "tavily":
            return await self._run_chain([self._search_tavily], query, max_results)
        if provider == "brave":
            return await self._run_chain([self._search_brave], query, max_results)
        if provider == "duckduckgo":
            return await self._run_chain(
                [self._search_ddg_html, self._search_ddg_lite, self._search_ddg_instant],
                query, max_results,
            )

        # "auto": query the configured keyed engines together and merge; fall
        # back to keyless DuckDuckGo only if neither returns anything.
        keyed = []
        if self._tavily_key():
            keyed.append(self._search_tavily)
        if self._brave_key():
            keyed.append(self._search_brave)
        if keyed:
            merged = await self._gather_and_merge(keyed, query, max_results)
            if merged["success"]:
                return merged
        return await self._run_chain(
            [self._search_ddg_html, self._search_ddg_lite, self._search_ddg_instant],
            query, max_results,
        )

    async def _run_chain(self, chain, query: str, max_results: int) -> Dict[str, Any]:
        """Try providers in order; first one with results wins."""
        errors: List[str] = []
        for search_fn in chain:
            name = search_fn.__name__.replace("_search_", "")
            try:
                results, answer = await search_fn(query, max_results)
            except Exception as e:  # keep trying the next provider
                logger.warning("web_search provider '%s' failed: %s", name, e)
                errors.append(f"{name}: {e}")
                continue
            if results:
                out: Dict[str, Any] = {"success": True, "provider": name, "results": results[:max_results]}
                if answer:
                    out["answer"] = answer
                    out["answer_provider"] = name
                return out
            errors.append(f"{name}: no results")

        logger.error("web_search: all providers failed for %r (%s)", query, "; ".join(errors))
        return {
            "success": False,
            "error": "All search providers failed or returned no results. " + "; ".join(errors),
            "results": [],
        }

    async def _gather_and_merge(self, providers, query: str, max_results: int) -> Dict[str, Any]:
        """Query keyed providers concurrently and merge their results.

        Keeps the first available AI summary (Tavily's) and interleaves each
        engine's results, deduped by URL, so both engines are represented.
        """
        import asyncio as _asyncio  # local alias; asyncio imported at module top

        outcomes = await _asyncio.gather(
            *[p(query, max_results) for p in providers], return_exceptions=True
        )
        answer: Optional[str] = None
        answer_provider: Optional[str] = None
        result_lists: List[List[Dict[str, str]]] = []
        names_ok: List[str] = []
        for prov, outcome in zip(providers, outcomes):
            name = prov.__name__.replace("_search_", "")
            if isinstance(outcome, Exception):
                logger.warning("web_search provider '%s' failed: %s", name, outcome)
                continue
            results, prov_answer = outcome
            if prov_answer and not answer:
                answer, answer_provider = prov_answer, name
            if results:
                result_lists.append(results)
                names_ok.append(name)

        if not result_lists:
            return {"success": False, "results": []}

        # More sources when merging engines, so the model can cross-check.
        cap = min(max_results * 2, 8) if len(result_lists) > 1 else max_results
        merged = self._interleave_dedupe(result_lists, cap)
        out: Dict[str, Any] = {"success": True, "provider": "+".join(names_ok), "results": merged}
        if answer:
            out["answer"] = answer
            out["answer_provider"] = answer_provider
        return out

    @staticmethod
    def _interleave_dedupe(result_lists: List[List[Dict[str, str]]], cap: int) -> List[Dict[str, str]]:
        """Round-robin across providers' result lists, deduping by URL."""
        from itertools import zip_longest

        seen = set()
        merged: List[Dict[str, str]] = []
        for group in zip_longest(*result_lists):
            for r in group:
                if not r or not r.get("link"):
                    continue
                key = r["link"].rstrip("/").lower()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(r)
                if len(merged) >= cap:
                    return merged
        return merged

    # --- providers ------------------------------------------------------

    async def _search_tavily(self, query: str, max_results: int):
        key = self._tavily_key()
        if not key:
            raise RuntimeError("TAVILY_API_KEY not set")
        # 'advanced' depth is materially more accurate for date/fact queries —
        # 'basic' returned a wrong answer for "who died on <date>" in testing.
        payload = {
            "query": query,
            "max_results": max_results,
            "search_depth": self._get("tavily_search_depth", "advanced"),
            "include_answer": True,
        }
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        async with self._session() as session:
            async with session.post(
                "https://api.tavily.com/search", json=payload, headers=headers
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                data = await resp.json()
        results = [
            {
                "title": r.get("title", ""),
                "link": r.get("url", ""),
                "snippet": (r.get("content", "") or "").strip(),
            }
            for r in data.get("results", [])
            if r.get("url")
        ]
        return results, data.get("answer")

    async def _search_brave(self, query: str, max_results: int):
        key = self._brave_key()
        if not key:
            raise RuntimeError("BRAVE_SEARCH_API_KEY not set")
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": key,
        }
        params = {"q": query, "count": min(max_results, 20)}
        async with self._session() as session:
            async with session.get(
                "https://api.search.brave.com/res/v1/web/search",
                params=params,
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                data = await resp.json()
        web = (data.get("web") or {}).get("results", [])
        results = [
            {
                "title": r.get("title", ""),
                "link": r.get("url", ""),
                "snippet": self._strip_html(r.get("description", "")),
            }
            for r in web
            if r.get("url")
        ]
        return results, None

    async def _search_ddg_html(self, query: str, max_results: int):
        """Scrape the DuckDuckGo HTML endpoint (real organic results, keyless)."""
        data = {"q": query, "kl": self._get("region", "wt-wt")}
        markup = await self._ddg_post("https://html.duckduckgo.com/html/", data)
        return self._parse_ddg_html(markup, max_results), None

    async def _search_ddg_lite(self, query: str, max_results: int):
        """Scrape the DuckDuckGo Lite endpoint — simpler markup, dodges walls."""
        data = {"q": query, "kl": self._get("region", "wt-wt")}
        markup = await self._ddg_post("https://lite.duckduckgo.com/lite/", data)
        return self._parse_ddg_lite(markup, max_results), None

    async def _ddg_post(self, url: str, data: Dict[str, str]) -> str:
        """POST to a DuckDuckGo scrape endpoint, retrying on the 202/429 soft
        rate-limit with exponential backoff before giving up on this provider."""
        delays = [0.0, 0.8, 1.8]  # first try immediate, then back off
        last_status = None
        for delay in delays:
            if delay:
                await asyncio.sleep(delay)
            async with self._session() as session:
                async with session.post(url, data=data) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    last_status = resp.status
                    if resp.status not in (202, 429):
                        break  # a hard error won't clear by retrying
        raise RuntimeError(f"HTTP {last_status}")

    async def _search_ddg_instant(self, query: str, max_results: int):
        """DuckDuckGo Instant Answer API — last resort (disambiguation only)."""
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }
        async with self._session() as session:
            async with session.get("https://api.duckduckgo.com/", params=params) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                data = await resp.json(content_type=None)

        results: List[Dict[str, str]] = []
        abstract = (data.get("AbstractText") or "").strip()
        if abstract and data.get("AbstractURL"):
            results.append(
                {
                    "title": data.get("Heading", query),
                    "link": data.get("AbstractURL", ""),
                    "snippet": abstract,
                }
            )
        for topic in data.get("RelatedTopics", []):
            # RelatedTopics can nest under "Topics" (category groupings).
            subtopics = topic.get("Topics", [topic]) if isinstance(topic, dict) else []
            for sub in subtopics:
                if sub.get("FirstURL") and sub.get("Text"):
                    results.append(
                        {
                            "title": sub.get("Text", ""),
                            "link": sub.get("FirstURL", ""),
                            "snippet": sub.get("Text", ""),
                        }
                    )
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break
        answer = (data.get("Answer") or "").strip() or None
        return results, answer

    # --- parsing helpers ------------------------------------------------

    def _parse_ddg_html(self, markup: str, max_results: int) -> List[Dict[str, str]]:
        soup = BeautifulSoup(markup, "html.parser")
        results: List[Dict[str, str]] = []
        for result in soup.select("div.result, div.web-result"):
            link_el = result.select_one("a.result__a")
            if not link_el:
                continue
            href = self._decode_ddg_url(link_el.get("href", ""))
            if not href:
                continue
            snippet_el = result.select_one("a.result__snippet, .result__snippet")
            results.append(
                {
                    "title": link_el.get_text(" ", strip=True),
                    "link": href,
                    "snippet": snippet_el.get_text(" ", strip=True) if snippet_el else "",
                }
            )
            if len(results) >= max_results:
                break
        return results

    def _parse_ddg_lite(self, markup: str, max_results: int) -> List[Dict[str, str]]:
        soup = BeautifulSoup(markup, "html.parser")
        results: List[Dict[str, str]] = []
        # Lite renders results as <a class="result-link"> with the snippet in the
        # following <td class="result-snippet">.
        for link_el in soup.select("a.result-link"):
            href = self._decode_ddg_url(link_el.get("href", ""))
            if not href:
                continue
            snippet = ""
            row = link_el.find_parent("tr")
            snip_row = row.find_next_sibling("tr") if row else None
            if snip_row:
                snip_el = snip_row.select_one(".result-snippet")
                if snip_el:
                    snippet = snip_el.get_text(" ", strip=True)
            results.append(
                {
                    "title": link_el.get_text(" ", strip=True),
                    "link": href,
                    "snippet": snippet,
                }
            )
            if len(results) >= max_results:
                break
        return results

    @staticmethod
    def _decode_ddg_url(href: str) -> str:
        """DuckDuckGo wraps result links as /l/?uddg=<encoded-target>. Unwrap it."""
        if not href:
            return ""
        if href.startswith("//"):
            href = "https:" + href
        parsed = urlparse(href)
        if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
            target = parse_qs(parsed.query).get("uddg", [""])[0]
            if target:
                return unquote(target)
        # Skip DDG-internal ad/redirect links with no real target.
        if parsed.netloc.endswith("duckduckgo.com"):
            return ""
        return href

    @staticmethod
    def _strip_html(text: str) -> str:
        if not text:
            return ""
        return BeautifulSoup(html.unescape(text), "html.parser").get_text(" ", strip=True)

    def _session(self) -> aiohttp.ClientSession:
        timeout = aiohttp.ClientTimeout(total=float(self._get("timeout_seconds", 12.0)))
        return aiohttp.ClientSession(
            timeout=timeout,
            headers={"User-Agent": _USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
        )

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "web_search",
            "description": (
                "Search the live web. Returns a list of results with title, link, "
                "and snippet. Uses Tavily/Brave when an API key is configured, "
                "otherwise DuckDuckGo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        }
