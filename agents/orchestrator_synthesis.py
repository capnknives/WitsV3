"""Final-answer synthesis guardrails for the ReAct orchestrator."""

from __future__ import annotations

import re
from typing import Any

from agents.routing_classifier import needs_codebase_intro

_DESKTOP_ACTION_RE = re.compile(
    r"\b(open|launch|start|run)\b.{0,40}\b("
    r"edge|chrome|firefox|browser|notepad|excel|word|spotify|discord|app"
    r")\b",
    re.IGNORECASE,
)


class OrchestratorSynthesisMixin:
    """Validate and auto-synthesize final answers from tool observations."""

    _CODEBASE_HALLUCINATION_PHRASES = (
        "witwatersrand",
        "wits university",
        "university of the wits",
    )

    _WITSV3_PROJECT_MARKERS = (
        "witsv3",
        "wits v3",
        "ollama",
        "orchestrat",
        "fastapi",
        "react",
        "tool registry",
        "control center",
        "run_web",
        "local-first",
        "agents/",
    )

    @classmethod
    def _answer_hallucinates_external_org(cls, answer: str) -> bool:
        lowered = (answer or "").lower()
        return any(phrase in lowered for phrase in cls._CODEBASE_HALLUCINATION_PHRASES)

    @classmethod
    def _answer_mentions_witsv3_project(cls, answer: str) -> bool:
        lowered = (answer or "").lower()
        return any(marker in lowered for marker in cls._WITSV3_PROJECT_MARKERS)

    _REFUSAL_PHRASES = (
        "don't have access",
        "do not have access",
        "cannot access",
        "can't access",
        "not uploaded",
        "no documents",
        "don't have your",
        "do not have your",
        "knowledge cutoff",
        "training data",
        "as an ai",
        "i'm not able to browse",
        "i cannot browse",
    )

    @staticmethod
    def _latest_observation_prefix(observations: list[str], prefix: str) -> str | None:
        for obs in reversed(observations):
            if prefix in obs:
                return obs
        return None

    @staticmethod
    def _significant_terms(text: str, min_len: int = 4) -> set:
        words = re.findall(r"[a-z0-9]+", text.lower())
        stop = {
            "that",
            "this",
            "with",
            "from",
            "have",
            "your",
            "about",
            "what",
            "when",
            "where",
            "which",
            "their",
            "there",
            "would",
            "could",
            "should",
            "been",
            "were",
            "they",
            "them",
            "than",
            "then",
            "into",
            "only",
            "also",
            "just",
            "like",
            "some",
            "such",
            "very",
            "does",
            "answer",
            "using",
            "below",
            "results",
            "search",
            "document",
        }
        return {w for w in words if len(w) >= min_len and w not in stop}

    def _answer_denies_access(self, answer: str) -> bool:
        lowered = answer.lower()
        return any(phrase in lowered for phrase in self._REFUSAL_PHRASES)

    def _answer_references_evidence(
        self, answer: str, evidence_terms: set, min_overlap: int = 2
    ) -> bool:
        if not evidence_terms:
            return True
        answer_terms = self._significant_terms(answer)
        return len(answer_terms & evidence_terms) >= min_overlap

    @classmethod
    def _extract_document_evidence_terms(cls, observation: str) -> set:
        terms: set = set()
        for line in observation.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("document_search"):
                continue
            if stripped.startswith("("):
                continue
            if stripped.startswith("["):
                header, _, body = stripped.partition("]")
                terms |= cls._significant_terms(header)
                if body.strip():
                    terms |= cls._significant_terms(body)
            else:
                terms |= cls._significant_terms(stripped)
        return terms

    @classmethod
    def _extract_web_summary(cls, observation: str) -> str | None:
        for line in observation.splitlines():
            if "summary" in line.lower() and ":" in line:
                return line.split(":", 1)[1].strip()
        return None

    def _goal_expects_documents(self, goal: str) -> bool:
        lowered = (goal or "").lower()
        signals = (
            "document",
            "report",
            "file",
            "upload",
            "audit",
            "my notes",
            "ingested",
            "summarize",
        )
        return any(sig in lowered for sig in signals)

    @staticmethod
    def _document_search_weak(doc_obs: str) -> bool:
        if not doc_obs:
            return False
        if "(no matching passages" in doc_obs or "(search failed:" in doc_obs:
            return True
        scores = [float(m) for m in re.findall(r"relevance=([0-9.]+)", doc_obs)]
        return bool(scores) and max(scores) < 0.25

    @staticmethod
    def _answer_acknowledges_gap(answer: str) -> bool:
        lowered = (answer or "").lower()
        gap_phrases = (
            "not in your",
            "couldn't find",
            "could not find",
            "no matching",
            "don't have",
            "do not have",
            "not uploaded",
            "insufficient",
            "try uploading",
            "broader query",
            "didn't find",
            "did not find",
        )
        return any(phrase in lowered for phrase in gap_phrases)

    def _insufficient_document_evidence_message(self, goal: str) -> str:
        return (
            f"I searched your ingested documents for “{goal}” but didn't find passages "
            "that clearly answer the question. Try rephrasing, naming a specific file, "
            "or uploading the document if it isn't in the documents folder yet."
        )

    def _validate_final_answer_synthesis(
        self, final_answer: str, state: dict[str, Any]
    ) -> str | None:
        """
        Return a guard message when final_answer ignores usable search observations.
        """
        observations = state.get("observations", [])
        if not observations or not final_answer or not str(final_answer).strip():
            return None

        goal = state.get("goal", "")
        if needs_codebase_intro(goal) and not self._has_codebase_file_observation(observations):
            return (
                "Codebase intro question but no read_file/list_directory observations. "
                "Read README.md and AGENTS.md before answering."
            )

        if needs_codebase_intro(goal) and self._has_codebase_file_observation(observations):
            if self._answer_hallucinates_external_org(final_answer):
                return (
                    "Answer confuses this WitsV3 repo with an external organization. "
                    "Summarize only README.md / AGENTS.md content."
                )
            if not self._answer_mentions_witsv3_project(final_answer):
                return (
                    "read_file returned project docs but the answer does not describe "
                    "this WitsV3 codebase. Summarize only what was read."
                )
            terms = self._extract_codebase_evidence_terms(observations)
            if terms and not self._answer_references_evidence(
                final_answer, terms, min_overlap=1
            ):
                return (
                    "read_file returned project docs but the answer does not reference "
                    "WitsV3/README/agents content. Summarize only what was read."
                )

        doc_obs = self._latest_observation_prefix(observations, "document_search results")
        if doc_obs and self._document_search_weak(doc_obs) and self._goal_expects_documents(goal):
            if not self._answer_acknowledges_gap(final_answer):
                return (
                    "document_search found no strong excerpts but the answer sounds definitive. "
                    "Say clearly that the uploaded documents do not contain enough evidence."
                )

        if (
            doc_obs
            and "(no matching passages" not in doc_obs
            and not self._document_search_weak(doc_obs)
        ):
            if self._answer_denies_access(final_answer):
                return (
                    "document_search returned excerpts but the answer claims no access. "
                    "Summarize the numbered EXCERPTS only."
                )
            evidence = self._extract_document_evidence_terms(doc_obs)
            if evidence and not self._answer_references_evidence(final_answer, evidence):
                return (
                    "Answer does not appear grounded in document_search EXCERPTS. "
                    "Cite or paraphrase the numbered passages."
                )

        web_obs = self._latest_observation_prefix(observations, "web_search results")
        if web_obs and "(no results found)" not in web_obs:
            if state.get("lookup_search_done") and self._answer_denies_access(final_answer):
                return (
                    "web_search already returned SOURCES but the answer refuses or deflects. "
                    "Use the summary and SOURCES in observations."
                )
            summary = self._extract_web_summary(web_obs)
            if (
                summary
                and len(final_answer.strip()) < 50
                and not self._answer_references_evidence(
                    final_answer, self._significant_terms(summary), min_overlap=1
                )
            ):
                return "Answer is too thin given an available web_search summary."

        return None

    def _auto_synthesize_from_observations(self, state: dict[str, Any]) -> str | None:
        """Build a grounded answer from the latest search observation when the model won't."""
        observations = state.get("observations", [])
        goal = state.get("goal", "")

        if needs_codebase_intro(goal) and self._has_codebase_file_observation(observations):
            codebase = self._auto_synthesize_codebase_from_observations(observations)
            if codebase:
                return codebase

        web_obs = self._latest_observation_prefix(observations, "web_search results")
        if web_obs and "(no results found)" not in web_obs:
            summary = self._extract_web_summary(web_obs)
            if summary:
                return summary
            lines = [ln.strip() for ln in web_obs.splitlines() if ln.strip().startswith("[")]
            if lines:
                return f"Based on web search for your question: {lines[0]}"

        doc_obs = self._latest_observation_prefix(observations, "document_search results")
        if doc_obs and self._document_search_weak(doc_obs):
            return self._insufficient_document_evidence_message(goal)

        if doc_obs and "(no matching passages" not in doc_obs:
            excerpt_lines = [
                ln.strip()
                for ln in doc_obs.splitlines()
                if ln.strip().startswith("[") or (ln.startswith("    ") and ln.strip())
            ]
            if excerpt_lines:
                body = "\n".join(excerpt_lines[:4])
                return f"From your uploaded documents (re: {goal}):\n{body}"

        return None

    def _resolve_final_answer(self, final_answer: str, state: dict[str, Any]) -> tuple:
        """
        Apply synthesis guard before accepting final_answer.

        Returns:
            (resolved_answer, completed) — completed False means retry the ReAct loop.
        """
        guard_msg = self._validate_final_answer_synthesis(final_answer, state)
        if guard_msg:
            retries = state.get("synthesis_guard_retries", 0)
            if retries == 0:
                state["synthesis_guard_retries"] = 1
                state["observations"].append(f"Synthesis guard: {guard_msg}")
                return final_answer, False
            fallback = self._auto_synthesize_from_observations(state)
            if fallback:
                final_answer = fallback

        save_guard = self._validate_save_claim(final_answer, state)
        if save_guard:
            retries = state.get("save_guard_retries", 0)
            if retries == 0:
                state["save_guard_retries"] = 1
                state["observations"].append(f"Save guard: {save_guard}")
                return final_answer, False
            final_answer = (
                "I could not save the conversation to disk — the write was empty or failed. "
                "Try: Save our conversation as exports/chat_log.txt"
            )

        desktop_msg = self._desktop_action_unavailable_message(state)
        if desktop_msg and _DESKTOP_ACTION_RE.search(state.get("goal", "")):
            final_answer = desktop_msg
            state["completed"] = True
            state["final_answer"] = final_answer
            return final_answer, True

        state["completed"] = True
        state["final_answer"] = final_answer
        return final_answer, True

    @staticmethod
    def _has_codebase_file_observation(observations: list[str]) -> bool:
        for obs in observations:
            if obs.startswith("Tool read_file result:") or obs.startswith(
                "Tool list_directory result:"
            ):
                return True
        return False

    @classmethod
    def _auto_synthesize_codebase_from_observations(cls, observations: list[str]) -> str | None:
        """Build a short grounded summary from read_file bootstrap observations."""
        lines: list[str] = []
        for obs in observations:
            if not obs.startswith("Tool read_file result:"):
                continue
            body = obs[len("Tool read_file result:") :].strip()
            if not body or body.startswith("Error"):
                continue
            for raw in body.splitlines():
                stripped = raw.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("|") or stripped.startswith("---"):
                    continue
                lines.append(stripped)
                if len(lines) >= 6:
                    break
            if len(lines) >= 6:
                break
        if not lines:
            return None
        summary = " ".join(lines[:6])
        if len(summary) > 900:
            summary = summary[:900].rsplit(" ", 1)[0] + "…"
        return (
            "From this project's README and docs (read from disk):\n"
            f"{summary}"
        )

    @staticmethod
    def _extract_codebase_evidence_terms(observations: list[str]) -> set[str]:
        """Pull project-specific tokens from read_file observations."""
        terms: set[str] = set()
        anchors = (
            "witsv3",
            "wits v3",
            "orchestrat",
            "ollama",
            "fastapi",
            "react",
            "readme",
            "agents.md",
            "control center",
            "tool registry",
        )
        blob = "\n".join(
            obs for obs in observations if obs.startswith("Tool read_file result:")
        ).lower()
        for anchor in anchors:
            if anchor in blob:
                terms.add(anchor.split()[0] if " " not in anchor else anchor)
        return terms

    def _validate_save_claim(self, final_answer: str, state: dict[str, Any]) -> str | None:
        goal = state.get("goal", "")
        if not self._goal_saves_conversation(goal) and not self._should_inject_conversation_content(
            goal, state.get("observations", [])
        ):
            return None
        lowered = (final_answer or "").lower()
        if not re.search(r"\b(saved|written|wrote|exported)\b", lowered):
            return None
        observations = state.get("observations", [])
        for obs in reversed(observations):
            if "write_file" in obs and self._observation_indicates_failure(obs):
                return "Claimed save but write_file failed in observations."
            if "write_file" in obs and "Successfully" in obs:
                return None
        return "Claimed save/export but no successful write_file observation exists."

    @staticmethod
    def _desktop_action_unavailable_message(state: dict[str, Any]) -> str | None:
        goal = state.get("goal", "")
        if not _DESKTOP_ACTION_RE.search(goal):
            return None
        observations = state.get("observations", [])
        listed = any("list_mcp_tools" in o for o in observations)
        has_mcp = any("mcp_" in o for o in observations)
        if listed and not has_mcp:
            return (
                "I can't open desktop applications from here unless an MCP tool provides that "
                "capability. Check the /mcp page to connect a desktop automation server, or run "
                "the app yourself."
            )
        return None
