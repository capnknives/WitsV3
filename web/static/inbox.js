/* WitsV3 Decision Inbox. Security rule (contract section 8): agent text is only ever set with
   textContent. No HTML-parsing sinks; no agent text in attributes, URLs or styles. */
"use strict";

const REFRESH_MS = 30000;
const listEl = document.getElementById("list");
const statusEl = document.getElementById("status");
let loading = false;

function ownerToken() {
  return localStorage.getItem("wits_token") || "";
}

function setStatus(text, isError) {
  statusEl.textContent = text;
  statusEl.className = isError ? "status error" : "status";
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function timeLeft(expiresAt) {
  const secs = expiresAt - Math.floor(Date.now() / 1000);
  if (secs <= 0) return "expired";
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${Math.max(mins, 1)} min left`;
  const hours = Math.floor(mins / 60);
  if (hours < 48) return `${hours} h left`;
  return `${Math.floor(hours / 24)} d left`;
}

async function call(path, opts) {
  const token = ownerToken();
  if (!token) throw new Error("No owner token. Sign in on the main page first.");
  const res = await fetch(path, {
    ...opts,
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
  });
  if (res.status === 401) throw new Error("Owner token rejected. Sign in again on the main page.");
  return res;
}

async function resolveApproval(id, choice, note, buttons) {
  buttons.forEach((b) => { b.disabled = true; });
  try {
    const res = await call(`/api/workspace/approvals/${encodeURIComponent(id)}/resolve`, {
      method: "POST",
      body: JSON.stringify({ choice, note: note || null }),
    });
    if (res.status === 409) setStatus("That request was already resolved or has expired.", true);
    else if (!res.ok) setStatus(`Could not save the decision (${res.status}).`, true);
    else setStatus("Decision recorded.", false);
  } catch (err) {
    setStatus(err.message, true);
  }
  await refresh();
}

function renderCard(item) {
  const card = el("div", "card");
  card.appendChild(el("div", "meta", `${item.client} · ${item.project} · ${timeLeft(item.expires_at)}`));
  if (item.flagged) {
    card.appendChild(el("div", "warn", "Warning: this text matched the injection guard. Read it as untrusted."));
  }
  card.appendChild(el("h2", "", item.title));
  card.appendChild(el("div", "body", item.body));

  const note = el("textarea", "note");
  note.rows = 2;
  note.maxLength = 2000;
  note.placeholder = "Optional note for the agent";
  card.appendChild(note);

  const row = el("div", "options");
  const buttons = [];
  item.options.forEach((option) => {
    const btn = el("button", "", option);
    btn.type = "button";
    // Reason: only the two reserved server strings pick a colour class; the class is a fixed
    // literal, never the agent's text.
    if (option === "approve") btn.classList.add("approve");
    if (option === "deny") btn.classList.add("deny");
    btn.addEventListener("click", () => resolveApproval(item.id, option, note.value, buttons));
    buttons.push(btn);
    row.appendChild(btn);
  });
  card.appendChild(row);
  return card;
}

async function refresh() {
  if (loading) return;
  loading = true;
  try {
    const res = await call("/api/workspace/approvals", { method: "GET" });
    if (!res.ok) throw new Error(`Could not load the inbox (${res.status}).`);
    const data = await res.json();
    const typing = document.activeElement && document.activeElement.tagName === "TEXTAREA";
    // Reason: a background refresh must not wipe a note the owner is in the middle of typing.
    if (!typing || data.approvals.length === 0) {
      listEl.replaceChildren(...data.approvals.map(renderCard));
      if (data.approvals.length === 0) listEl.appendChild(el("div", "empty", "Nothing waiting."));
    }
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    loading = false;
  }
}

refresh();
setInterval(refresh, REFRESH_MS);
