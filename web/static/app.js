/* WitsV3 Web UI */
"use strict";

const $ = (sel) => document.querySelector(sel);

const chatEl = $("#chat");
const inputEl = $("#input");
const sendBtn = $("#send-btn");
const statusDot = $("#status-dot");

let sessionId = localStorage.getItem("wits_session") || null;
let token = localStorage.getItem("wits_token") || "";
let busy = false;

// Magic login link: /?token=XYZ stores the token and cleans the URL,
// so phones never have to type it manually.
const urlToken = new URLSearchParams(location.search).get("token");
if (urlToken) {
  token = urlToken.trim();
  localStorage.setItem("wits_token", token);
  history.replaceState(null, "", location.pathname);
}

/* ---------------------------------------------------------- helpers */
function authHeaders(extra = {}) {
  return token ? { Authorization: `Bearer ${token}`, ...extra } : extra;
}

async function api(path, opts = {}) {
  const res = await fetch(path, { ...opts, headers: authHeaders(opts.headers || {}) });
  if (res.status === 401) {
    // Whatever token we had is wrong - drop it so it can't stick around
    token = "";
    localStorage.removeItem("wits_token");
    showTokenModal();
    throw new Error("unauthorized");
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
  $("#token-modal").hidden = false;
  $("#token-input").focus();
}

$("#token-save").addEventListener("click", () => {
  token = $("#token-input").value.trim();
  localStorage.setItem("wits_token", token);
  $("#token-modal").hidden = true;
  checkStatus();
});

/* ---------------------------------------------------------- status */
async function checkStatus() {
  try {
    const res = await api("/api/status");
    if (res.ok) {
      const s = await res.json();
      statusDot.className = "dot ok";
      statusDot.title = `${s.project} v${s.version} — ${s.models.default}, ${s.tool_count} tools`;
      return;
    }
    statusDot.className = "dot bad";
  } catch (e) {
    statusDot.className = "dot bad";
  }
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
          else if (payload.type === "result") { addAssistantMsg(payload.content); sawResult = true; }
          else if (payload.type === "error") { addAssistantMsg(payload.content, true); sawResult = true; }
        } else if (event === "done") {
          if (!sawResult && payload.final) addAssistantMsg(payload.final);
        }
      }
    }
  } catch (e) {
    if (e.message !== "unauthorized") addAssistantMsg(`Connection error: ${e.message}`, true);
  } finally {
    typing.remove();
    busy = false;
    sendBtn.disabled = false;
    inputEl.focus();
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

document.querySelectorAll(".panel-tabs button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".panel-tabs button").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    btn.classList.add("active");
    $(`#tab-${btn.dataset.tab}`).classList.add("active");
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

async function loadDocs() {
  try {
    const res = await api("/api/documents");
    const { files } = await res.json();
    const box = $("#doc-list");
    box.innerHTML = "";
    if (!files.length) box.appendChild(el("div", "doc-item", "No documents yet — upload one!"));
    files.forEach((f) => {
      const item = el("div", "doc-item");
      item.appendChild(el("div", null, f.name));
      item.appendChild(el("div", "meta", `${(f.size / 1024).toFixed(1)} KB · ${f.chunks} chunks`));
      box.appendChild(item);
    });
  } catch (e) { /* handled */ }
}

$("#doc-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await api("/api/documents/upload", { method: "POST", body: form });
    if (res.ok) loadDocs();
  } catch (err) { /* handled */ }
  e.target.value = "";
});

/* ---------------------------------------------------------- boot */
addAssistantMsg("Hey Richard — WITS is online. Ask me anything, or open the ☰ panel for tools, memory and documents.");
checkStatus();
setInterval(checkStatus, 30000);
