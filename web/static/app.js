/* WitsV3 Web UI */
"use strict";

const $ = (sel) => document.querySelector(sel);

const chatEl = $("#chat");
const inputEl = $("#input");
const sendBtn = $("#send-btn");
const statusDot = $("#status-dot");

let sessionId = localStorage.getItem("wits_session") || null;
let token = localStorage.getItem("wits_token") || "";
let guestToken = localStorage.getItem("wits_guest_token") || "";
let guestName = localStorage.getItem("wits_guest_name") || "";
let isGuest = Boolean(guestToken) && !token;
let busy = false;

function isLocalhost() {
  const h = location.hostname;
  return h === "localhost" || h === "127.0.0.1" || h === "[::1]";
}

function wantsOwnerLogin() {
  return new URLSearchParams(location.search).get("owner") === "1";
}

function consumeOwnerUrlToken() {
  const params = new URLSearchParams(location.search);
  const ownerParam = params.get("owner_token") || params.get("token");
  if (!ownerParam) return;
  // Owner magic links are localhost-only. LAN URLs must never embed the owner token.
  if (!isLocalhost()) {
    history.replaceState(null, "", location.pathname);
    return;
  }
  token = ownerParam.trim();
  localStorage.setItem("wits_token", token);
  localStorage.removeItem("wits_guest_token");
  guestToken = "";
  isGuest = false;
  history.replaceState(null, "", location.pathname);
}

/* ---------------------------------------------------------- helpers */
function activeToken() {
  return isGuest ? guestToken : token;
}

function authHeaders(extra = {}) {
  const t = activeToken();
  return t ? { Authorization: `Bearer ${t}`, ...extra } : extra;
}

async function api(path, opts = {}) {
  const res = await fetch(path, { ...opts, headers: authHeaders(opts.headers || {}) });
  if (res.status === 401) {
    if (isGuest) {
      guestToken = "";
      localStorage.removeItem("wits_guest_token");
      location.href = "/join";
      throw new Error("unauthorized");
    }
    token = "";
    localStorage.removeItem("wits_token");
    showTokenModal();
    throw new Error("unauthorized");
  }
  if (res.status === 403 && isGuest) {
    throw new Error("forbidden");
  }
  return res;
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scrollDown() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

/* ---------------------------------------------------------- token modal */
function showTokenModal() {
  if (isGuest || guestToken) {
    location.href = "/join";
    return;
  }
  $("#token-modal").hidden = false;
  $("#token-input").focus();
}

function applyGuestChrome() {
  if (!isGuest) return;
  document.body.classList.add("guest-mode");
  const settings = $("#header-settings-link");
  if (settings) settings.hidden = true;
  const panelSettings = $("#settings-link");
  if (panelSettings) panelSettings.hidden = true;
  const memTab = document.querySelector('.panel-tabs [data-tab="memory"]');
  const docsTab = document.querySelector('.panel-tabs [data-tab="docs"]');
  if (memTab) memTab.hidden = true;
  if (docsTab) docsTab.hidden = true;
  if (guestName) {
    const title = document.querySelector(".title-text");
    if (title) title.textContent = `WITS · ${guestName}`;
  }
}

$("#token-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const errEl = $("#token-error");
  const btn = $("#token-save");
  const candidate = $("#token-input").value.trim();

  if (!candidate) {
    errEl.textContent = "Paste the token first — the field is empty.";
    errEl.hidden = false;
    return;
  }

  // Validate against the server BEFORE closing the modal, with a raw fetch
  // so a 401 here doesn't loop back through api()'s modal handling.
  btn.disabled = true;
  btn.textContent = "Checking…";
  try {
    const res = await fetch("/api/status", { headers: { Authorization: `Bearer ${candidate}` } });
    if (res.ok) {
      token = candidate;
      localStorage.setItem("wits_token", token);
      localStorage.removeItem("wits_guest_token");
      guestToken = "";
      isGuest = false;
      $("#token-input").value = "";
      errEl.hidden = true;
      $("#token-modal").hidden = true;
      addAssistantMsg("✅ Token accepted — you're connected.");
      checkStatus();
    } else if (res.status === 401) {
      errEl.textContent = "Token rejected — it doesn't match WITSV3_WEB_TOKEN on the server.";
      errEl.hidden = false;
    } else {
      errEl.textContent = `Server error (HTTP ${res.status}) — try again.`;
      errEl.hidden = false;
    }
  } catch {
    errEl.textContent = "Can't reach the server — is run_web.py still running?";
    errEl.hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = "Connect";
  }
});

/* ---------------------------------------------------------- status */
async function checkStatus() {
  try {
    const res = await api("/api/status");
    if (res.ok) {
      const s = await res.json();
      if (s.role === "guest") {
        isGuest = true;
        if (s.display_name) {
          guestName = s.display_name;
          localStorage.setItem("wits_guest_name", guestName);
        }
        applyGuestChrome();
      }
      if (s.ollama && s.ollama.available === false) {
        statusDot.className = "dot warn";
        statusDot.title = `Web UI OK — Ollama is not running (${s.ollama.url})`;
      } else {
        statusDot.className = "dot ok";
        const who = isGuest && guestName ? ` · ${guestName}` : "";
        statusDot.title = `${s.project} v${s.version}${who} — ${s.models.default}, ${s.tool_count} tools`;
      }
      if (!isGuest) checkEscalations();
      return;
    }
    statusDot.className = "dot bad";
  } catch (e) {
    statusDot.className = "dot bad";
  }
}

/* ---------------------------------------------- ask-Claude approvals */
const seenEscalations = new Set();

async function checkEscalations() {
  try {
    const res = await api("/api/escalations");
    if (!res.ok) return;
    const { requests } = await res.json();
    for (const r of requests) {
      if (r.status === "pending" && !seenEscalations.has(r.id)) {
        seenEscalations.add(r.id);
        addEscalationCard(r);
      }
    }
  } catch (e) { /* handled by api() */ }
}

function addEscalationCard(r) {
  const card = el("div", "escalation-card");
  card.appendChild(el("div", "esc-title", "🤖 WITS wants to ask Claude"));
  card.appendChild(el("div", "esc-question", r.question));
  if (r.context) {
    const details = el("details", "esc-context");
    details.appendChild(el("summary", null, "context it will send"));
    details.appendChild(el("div", null, r.context));
    card.appendChild(details);
  }
  card.appendChild(el("div", "esc-cost",
    `${r.model} — worst case ≈ $${r.estimate.max_cost_usd} (nothing is sent until you approve)`));

  const actions = el("div", "esc-actions");
  const approveBtn = el("button", "esc-approve", "Approve & send");
  const denyBtn = el("button", "esc-deny", "Deny");

  approveBtn.addEventListener("click", async () => {
    approveBtn.disabled = denyBtn.disabled = true;
    approveBtn.textContent = "Asking Claude…";
    try {
      const res = await api(`/api/escalations/${r.id}/approve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
      const result = await res.json();
      card.remove();
      if (res.ok && result.status === "answered") {
        const cost = result.cost_usd != null ? ` (cost: $${result.cost_usd})` : "";
        addAssistantMsg(`💠 Claude (${result.model})${cost}:\n\n${result.answer}`);
      } else {
        addAssistantMsg(`Claude escalation failed: ${result.error || result.detail || "unknown error"}`, true);
      }
    } catch (e) {
      approveBtn.disabled = denyBtn.disabled = false;
      approveBtn.textContent = "Approve & send";
    }
  });

  denyBtn.addEventListener("click", async () => {
    approveBtn.disabled = denyBtn.disabled = true;
    try { await api(`/api/escalations/${r.id}/deny`, { method: "POST" }); } catch (e) { /* ok */ }
    card.remove();
    addAssistantMsg("Escalation denied — no tokens were spent.");
  });

  actions.appendChild(approveBtn);
  actions.appendChild(denyBtn);
  card.appendChild(actions);
  chatEl.appendChild(card);
  scrollDown();
}

/* ---------------------------------------------------------- chat */
function addUserMsg(text) {
  chatEl.appendChild(el("div", "msg user", text));
  scrollDown();
}

function addAssistantMsg(text, isError = false) {
  chatEl.appendChild(el("div", `msg assistant${isError ? " error" : ""}`, text));
  scrollDown();
}

function addErrorMsg(message, userError) {
  if (userError && userError.code === "ollama_unavailable") {
    const card = el("div", "msg assistant error ollama-error");
    card.appendChild(el("strong", null, `⚠ ${userError.message}`));
    if (userError.hint) card.appendChild(el("div", "error-hint", userError.hint));
    chatEl.appendChild(card);
  } else if (userError && userError.hint) {
    const card = el("div", "msg assistant error");
    card.appendChild(el("div", null, message));
    card.appendChild(el("div", "error-hint", userError.hint));
    chatEl.appendChild(card);
  } else {
    addAssistantMsg(message, true);
  }
  scrollDown();
}

function addThinking(text) {
  const details = el("details", "thinking");
  const summary = el("summary", null, "thinking…");
  details.appendChild(summary);
  details.appendChild(el("div", null, text));
  chatEl.appendChild(details);
  scrollDown();
}

function addToolChip(text) {
  chatEl.appendChild(el("div", "tool-chip", `🔧 ${text.slice(0, 120)}`));
  scrollDown();
}

function addObservationCard(text, source) {
  const safe = text || "";
  const label = source ? `${source} output` : "Tool output";
  const details = el("details", "tool-output");
  details.appendChild(el("summary", null, `🧾 ${label}`));
  details.appendChild(el("div", "tool-output-text", safe));
  chatEl.appendChild(details);
  scrollDown();
}

async function sendMessage(text) {
  if (busy || !text.trim()) return;
  busy = true;
  sendBtn.disabled = true;
  addUserMsg(text);

  const typing = el("div", "typing", "WITS is working");
  chatEl.appendChild(typing);
  scrollDown();

  try {
    const res = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let sawResult = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const raw = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);

        let event = "message", data = "";
        for (const line of raw.split("\n")) {
          if (line.startsWith("event: ")) event = line.slice(7).trim();
          else if (line.startsWith("data: ")) data += line.slice(6);
        }
        if (!data) continue;
        const payload = JSON.parse(data);

        if (event === "session") {
          sessionId = payload.session_id;
          localStorage.setItem("wits_session", sessionId);
        } else if (event === "stream") {
          if (payload.type === "thinking") addThinking(payload.content);
          else if (payload.type === "tool_call" || payload.type === "action") addToolChip(payload.content);
          else if (payload.type === "observation") addObservationCard(payload.content, payload.source);
          else if (payload.type === "result") { addAssistantMsg(payload.content); sawResult = true; }
          else if (payload.type === "error") {
            addErrorMsg(payload.content, payload.user_error);
            sawResult = true;
          }
        } else if (event === "done") {
          if (payload.owner_action === "shutdown") {
            statusDot.className = "dot bad";
            statusDot.title = "WITS is shutting down…";
          } else if (payload.owner_action === "restart") {
            statusDot.className = "dot warn";
            statusDot.title = "WITS is restarting…";
          }
          if (!sawResult && payload.final) {
            if (payload.final.includes("Can't reach Ollama")) {
              addErrorMsg(payload.final.split("\n\n")[0], {
                code: "ollama_unavailable",
                message: payload.final.split("\n\n")[0],
                hint: payload.final.split("\n\n").slice(1).join("\n\n"),
              });
            } else {
              addAssistantMsg(payload.final);
            }
          }
        }
      }
    }
  } catch (e) {
    if (e.message !== "unauthorized") {
      addErrorMsg("Can't reach the WITS web server.", {
        code: "generic",
        message: "Can't reach the WITS web server.",
        hint: "Is run_web.py still running on this machine?",
      });
    }
  } finally {
    typing.remove();
    busy = false;
    sendBtn.disabled = false;
    inputEl.focus();
    checkEscalations(); // pick up any ask-Claude request queued during this turn
  }
}

$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = inputEl.value;
  inputEl.value = "";
  inputEl.style.height = "auto";
  sendMessage(text);
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("#composer").requestSubmit();
  }
});

inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 130) + "px";
});

$("#new-chat-btn").addEventListener("click", () => {
  sessionId = null;
  localStorage.removeItem("wits_session");
  chatEl.innerHTML = "";
  addAssistantMsg("New chat started. What can I do for you?");
});

$("#export-btn").addEventListener("click", async () => {
  if (!sessionId) {
    addAssistantMsg("Nothing to export yet — send a message first.", true);
    return;
  }
  try {
    const res = await api("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
    const data = await res.json();
    if (res.ok) {
      addAssistantMsg(`✅ ${data.message}`);
    } else {
      addAssistantMsg(`Export failed: ${data.detail || res.status}`, true);
    }
  } catch (e) {
    if (e.message !== "unauthorized") addAssistantMsg(`Export failed: ${e.message}`, true);
  }
});

/* ---------------------------------------------------------- panel */
const panel = $("#panel");
const backdrop = $("#panel-backdrop");

function openPanel() {
  panel.hidden = false;
  backdrop.hidden = false;
  loadTools();
  loadDocs();
}
function closePanel() {
  panel.hidden = true;
  backdrop.hidden = true;
}

$("#menu-btn").addEventListener("click", openPanel);
$("#panel-close").addEventListener("click", closePanel);
backdrop.addEventListener("click", closePanel);

let memLoadedOnce = false;
document.querySelectorAll(".panel-tabs button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".panel-tabs button").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    btn.classList.add("active");
    $(`#tab-${btn.dataset.tab}`).classList.add("active");
    if (btn.dataset.tab === "memory" && !memLoadedOnce) {
      memLoadedOnce = true;
      memRecentOffset = 0;
      loadRecentMemory();
    }
  });
});

async function loadTools() {
  try {
    const res = await api("/api/tools");
    const { tools } = await res.json();
    const box = $("#tab-tools");
    box.innerHTML = "";
    tools.forEach((t) => {
      const item = el("div", "tool-item");
      item.appendChild(el("b", null, t.name));
      item.appendChild(el("p", null, t.description));
      box.appendChild(item);
    });
  } catch (e) { /* handled by api() */ }
}

$("#memory-go").addEventListener("click", async () => {
  const q = $("#memory-q").value.trim();
  if (!q) return;
  try {
    const res = await api(`/api/memory/search?q=${encodeURIComponent(q)}&limit=8`);
    const { results } = await res.json();
    const box = $("#memory-results");
    box.innerHTML = "";
    if (!results.length) box.appendChild(el("div", "mem-item", "No matches."));
    results.forEach((r) => {
      const item = el("div", "mem-item");
      item.appendChild(el("div", "meta", `${r.type} · ${r.source} · ${r.relevance}`));
      item.appendChild(el("div", null, r.text));
      box.appendChild(item);
    });
  } catch (e) { /* handled */ }
});

let memRecentOffset = 0;

function memRecentLimit() {
  return Math.max(1, Math.min(parseInt($("#memory-recent-limit").value, 10) || 20, 100));
}

async function loadRecentMemory() {
  const box = $("#memory-recent-results");
  try {
    const limit = memRecentLimit();
    const segmentType = ($("#memory-recent-type").value || "").trim();
    const source = ($("#memory-recent-source").value || "").trim();

    const qs = new URLSearchParams({ limit: String(limit), offset: String(memRecentOffset) });
    if (segmentType) qs.set("segment_type", segmentType);
    if (source) qs.set("source", source);

    const res = await api(`/api/memory/recent?${qs.toString()}`);
    const body = await res.json();
    const results = body.results || [];
    box.innerHTML = "";

    if (!results.length) {
      box.appendChild(el("div", "mem-item", memRecentOffset > 0 ? "No more segments." : "No recent memory segments."));
    } else {
      for (const r of results) {
        const item = el("div", "mem-item");
        item.appendChild(el("div", "meta", `${r.type || "?"} · ${r.source || "?"} · ${r.timestamp || ""}`));
        const lines = [];
        if (r.tool_name) lines.push(`tool: ${r.tool_name}`);
        if (r.text) lines.push(r.text);
        item.appendChild(el("div", null, lines.join("\n\n")));
        box.appendChild(item);
      }
    }

    $("#memory-recent-pageinfo").textContent = results.length
      ? `${memRecentOffset + 1}–${memRecentOffset + results.length}`
      : "—";
    $("#memory-recent-prev").disabled = memRecentOffset <= 0;
    $("#memory-recent-next").disabled = !body.has_more;
  } catch (e) {
    /* handled by api() */
  }
}

$("#memory-recent-go").addEventListener("click", () => {
  memRecentOffset = 0;
  loadRecentMemory();
});
$("#memory-recent-prev").addEventListener("click", () => {
  memRecentOffset = Math.max(0, memRecentOffset - memRecentLimit());
  loadRecentMemory();
});
$("#memory-recent-next").addEventListener("click", () => {
  memRecentOffset += memRecentLimit();
  loadRecentMemory();
});

$("#memory-prune-go").addEventListener("click", async () => {
  try {
    const segmentType = ($("#memory-prune-type").value || "").trim();
    const source = ($("#memory-prune-source").value || "").trim();
    const confirm = ($("#memory-prune-confirm").value || "").trim();

    const status = $("#memory-prune-status");
    status.textContent = "";

    if (confirm !== "PRUNE") {
      status.textContent = `Type PRUNE to confirm.`;
      return;
    }

    const filter = {};
    if (segmentType) filter.type = segmentType;
    if (source) filter.source = source;

    if (!Object.keys(filter).length) {
      status.textContent = "Provide at least one filter: type and/or source.";
      return;
    }

    const res = await api("/api/memory/prune", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filter_dict: filter, confirm: "PRUNE" }),
    });

    if (res.ok) {
      const body = await res.json();
      status.textContent = `Pruned ${body.removed} memory segments.`;
      $("#memory-prune-confirm").value = "";

      // Reload the list (best-effort).
      memRecentOffset = 0;
      loadRecentMemory();
    } else {
      const body = await res.json().catch(() => ({}));
      status.textContent = body.detail || `Prune failed (HTTP ${res.status}).`;
    }
  } catch (e) {
    /* handled by api() */
  }
});

const DOC_ICONS = {
  pdf: "📕", md: "📝", txt: "📄", json: "🗂", csv: "📊",
  py: "🐍", js: "📜", html: "🌐", docx: "📘", doc: "📘",
};

function docIcon(ext) {
  return DOC_ICONS[(ext || "").toLowerCase()] || "📄";
}

function fmtSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function fmtDate(iso) {
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
  } catch (e) {
    return "";
  }
}

function docStatus(msg, kind) {
  const box = $("#doc-status");
  if (!msg) { box.hidden = true; box.textContent = ""; return; }
  box.hidden = false;
  box.className = "doc-status" + (kind ? ` ${kind}` : "");
  box.textContent = msg;
}

async function loadDocs() {
  const box = $("#doc-list");
  try {
    const res = await api("/api/documents");
    const body = await res.json();
    const files = body.files || [];
    $("#docs-count").textContent = files.length
      ? `${files.length} document${files.length === 1 ? "" : "s"} · ${body.total_chunks || 0} chunks`
      : "No documents yet";
    box.innerHTML = "";
    if (!files.length) {
      box.appendChild(el("div", "doc-empty", "Nothing here yet. Upload a file to make it searchable."));
      return;
    }
    files.forEach((f) => {
      const item = el("div", "doc-item");

      const icon = el("div", "doc-icon", docIcon(f.ext));

      const main = el("div", "doc-main");
      main.appendChild(el("div", "doc-name", f.name));
      const metaBits = [fmtSize(f.size), `${f.chunks} chunk${f.chunks === 1 ? "" : "s"}`];
      if (f.modified) metaBits.push(fmtDate(f.modified));
      main.appendChild(el("div", "meta", metaBits.join(" · ")));
      if (!f.chunks) {
        const warn = el("div", "doc-warn", "⚠ not indexed — try Re-index");
        main.appendChild(warn);
      }

      const del = el("button", "doc-del", "🗑");
      del.title = "Delete document";
      del.addEventListener("click", () => confirmDeleteDoc(del, f.name));

      item.appendChild(icon);
      item.appendChild(main);
      item.appendChild(del);
      box.appendChild(item);
    });
  } catch (e) { /* handled */ }
}

function confirmDeleteDoc(btn, name) {
  if (btn.dataset.confirm === "1") {
    deleteDoc(name);
    return;
  }
  btn.dataset.confirm = "1";
  btn.textContent = "Delete?";
  btn.classList.add("confirming");
  const reset = () => {
    btn.dataset.confirm = "";
    btn.textContent = "🗑";
    btn.classList.remove("confirming");
  };
  btn._resetTimer = setTimeout(reset, 3500);
  btn.addEventListener("blur", reset, { once: true });
}

async function deleteDoc(name) {
  try {
    docStatus(`Deleting ${name}…`);
    const res = await api("/api/documents/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (res.ok) {
      const body = await res.json();
      docStatus(`Deleted ${name} (${body.removed_chunks} chunk${body.removed_chunks === 1 ? "" : "s"} removed).`, "ok");
    } else {
      const body = await res.json().catch(() => ({}));
      docStatus(body.detail || `Delete failed (HTTP ${res.status}).`, "err");
    }
  } catch (e) {
    /* handled by api() */
  }
  loadDocs();
}

async function uploadFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;
  let ok = 0;
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    docStatus(`Uploading ${i + 1}/${files.length}: ${file.name}…`);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await api("/api/documents/upload", { method: "POST", body: form });
      if (res.ok) ok++;
    } catch (err) {
      /* handled by api() */
    }
  }
  docStatus(`Uploaded & indexed ${ok}/${files.length} file${files.length === 1 ? "" : "s"}.`, ok ? "ok" : "err");
  loadDocs();
}

$("#doc-file").addEventListener("change", (e) => {
  uploadFiles(e.target.files);
  e.target.value = "";
});

const dropZone = $("#doc-drop");
["dragenter", "dragover"].forEach((ev) =>
  dropZone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropZone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
  })
);
dropZone.addEventListener("drop", (e) => {
  if (e.dataTransfer && e.dataTransfer.files) uploadFiles(e.dataTransfer.files);
});

$("#docs-reindex").addEventListener("click", async () => {
  try {
    docStatus("Re-indexing all documents…");
    const res = await api("/api/documents/reindex", { method: "POST" });
    if (res.ok) {
      const body = await res.json();
      const ing = body.ingest || {};
      docStatus(
        ing.success === false
          ? `Re-index failed: ${ing.error || "unknown error"}`
          : `Re-indexed: ${ing.files_ingested ?? "?"} file(s), ${ing.chunks_added ?? "?"} chunk(s).`,
        ing.success === false ? "err" : "ok"
      );
    }
  } catch (e) {
    /* handled by api() */
  }
  loadDocs();
});

/* ---------------------------------------------------------- boot */
async function bootstrapAuth() {
  consumeOwnerUrlToken();

  let guestAccessEnabled = false;
  try {
    const res = await fetch("/api/guest/status");
    if (res.ok) {
      guestAccessEnabled = (await res.json()).enabled;
    }
  } catch {
    /* server down — fall through */
  }

  // Stale owner token from an old LAN magic link / QR — drop it when guest mode is on.
  if (guestAccessEnabled && !isLocalhost() && token && !wantsOwnerLogin()) {
    token = "";
    localStorage.removeItem("wits_token");
    isGuest = Boolean(guestToken);
  }

  applyGuestChrome();

  if (!activeToken()) {
    if (localStorage.getItem("wits_guest_token")) {
      location.href = "/join";
      return;
    }
    if (guestAccessEnabled && !wantsOwnerLogin()) {
      location.href = "/join";
      return;
    }
    showTokenModal();
    return;
  }

  if (isGuest) {
    addAssistantMsg(
      guestName
        ? `Welcome back, ${guestName} — you're chatting as a guest tester.`
        : "You're chatting as a guest tester. Ask me anything."
    );
  } else {
    addAssistantMsg("Hey Richard — WITS is online. Ask me anything, or open the ☰ panel for tools, memory and documents. Owner commands: /shutdown · /restart (require your web token).");
  }
  checkStatus();
  setInterval(checkStatus, 30000);
}

bootstrapAuth();
