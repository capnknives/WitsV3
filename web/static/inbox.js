/* WitsV3 Decision Inbox. Security rule (contract section 8): agent text is only ever set with
   textContent. No HTML-parsing sinks; no agent text in attributes, URLs or styles. */
"use strict";

const REFRESH_MS = 30000;
const listEl = document.getElementById("list");
const statusEl = document.getElementById("status");
const foldersEl = document.getElementById("folders");
let loading = false;

const MODE_WORDS = {
  read: "Read files in this folder",
  read_write: "Read and save files in this folder",
  index: "Read and index files in this folder",
};
const CHOICE_LABELS = { approve: "Approve 30 days", "one day": "One day", deny: "Deny" };

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
  // Reason: Safari does not move focus to a tapped button, so the note box can still be
  // focused here; force the redraw or the resolved card would stay on screen.
  await refresh(true);
}

function renderCard(item) {
  const card = el("div", "card");
  card.appendChild(el("div", "meta", `${item.client} · ${item.project} · ${timeLeft(item.expires_at)}`));
  if (item.flagged) {
    card.appendChild(el("div", "warn", "Warning: this text matched the injection guard. Read it as untrusted."));
  }
  const folder = item.folder || null;
  if (folder) {
    card.appendChild(el("h2", "", "Folder access"));
    card.appendChild(el("div", "body", `Folder: ${folder.path}`));
    card.appendChild(el("div", "body", MODE_WORDS[folder.mode] || "Use this folder"));
    card.appendChild(el("div", "body", `Why: ${folder.reason}`));
  } else {
    card.appendChild(el("h2", "", item.title));
    card.appendChild(el("div", "body", item.body));
  }

  const note = el("textarea", "note");
  note.rows = 2;
  note.maxLength = 2000;
  note.placeholder = "Optional note for the agent";
  card.appendChild(note);

  const row = el("div", "options");
  const buttons = [];
  item.options.forEach((option) => {
    const label = folder && CHOICE_LABELS[option] ? CHOICE_LABELS[option] : option;
    const btn = el("button", "", label);
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

function daysLeft(expiresAt) {
  const secs = expiresAt - Math.floor(Date.now() / 1000);
  return Math.max(Math.ceil(secs / 86400), 0);
}

function renderGrant(grant) {
  const row = el("div", "card");
  row.appendChild(el("h2", "", grant.path));
  row.appendChild(el("div", "meta", MODE_WORDS[grant.mode] || "Use this folder"));
  const when = new Date(grant.expires_at * 1000).toLocaleDateString();
  const days = daysLeft(grant.expires_at);
  row.appendChild(el("div", "meta", `Expires ${when} (${days} ${days === 1 ? "day" : "days"} left)`));
  const btn = el("button", "", "Revoke");
  btn.type = "button";
  btn.addEventListener("click", () => revokeGrant(grant.id, btn));
  const opts = el("div", "options");
  opts.appendChild(btn);
  row.appendChild(opts);
  return row;
}

async function revokeGrant(id, btn) {
  btn.disabled = true;
  try {
    const res = await call(`/api/workspace/folder-grants/${encodeURIComponent(id)}/revoke`, {
      method: "POST",
    });
    if (res.status === 404) setStatus("That folder access had already ended.", true);
    else if (!res.ok) setStatus(`Could not revoke (${res.status}).`, true);
    else setStatus("Folder access revoked.", false);
  } catch (err) {
    setStatus(err.message, true);
  }
  await refreshFolders();
}

async function refreshFolders() {
  try {
    const res = await call("/api/workspace/folder-grants", { method: "GET" });
    if (!res.ok) throw new Error(`Could not load folders (${res.status}).`);
    const data = await res.json();
    const nodes = [];
    nodes.push(el("div", "meta", "Always available:"));
    (data.base_folders || []).forEach((b) => {
      const word = b.mode === "index" ? "read and index" : "read and save";
      nodes.push(el("div", "body", `${b.name}: ${word} (${b.path})`));
    });
    const live = (data.grants || []).filter((g) => g.live);
    nodes.push(el("div", "meta", "Granted by you:"));
    if (live.length === 0) nodes.push(el("div", "body", "No extra folders are shared right now."));
    live.forEach((g) => nodes.push(renderGrant(g)));
    foldersEl.replaceChildren(...nodes);
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function refresh(force) {
  if (loading) return;
  loading = true;
  try {
    const res = await call("/api/workspace/approvals", { method: "GET" });
    if (!res.ok) throw new Error(`Could not load the inbox (${res.status}).`);
    const data = await res.json();
    const typing = document.activeElement && document.activeElement.tagName === "TEXTAREA";
    // Reason: a background refresh must not wipe a note the owner is in the middle of typing.
    if (force || !typing || data.approvals.length === 0) {
      listEl.replaceChildren(...data.approvals.map(renderCard));
      if (data.approvals.length === 0) listEl.appendChild(el("div", "empty", "Nothing waiting."));
      await refreshFolders();
    }
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    loading = false;
  }
}

refresh(false);
setInterval(() => refresh(false), REFRESH_MS);
