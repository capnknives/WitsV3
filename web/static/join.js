/* Guest join page */
"use strict";

const $ = (sel) => document.querySelector(sel);

function deviceId() {
  let id = localStorage.getItem("wits_guest_device_id");
  if (!id) {
    id = crypto.randomUUID ? crypto.randomUUID() : `dev-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    localStorage.setItem("wits_guest_device_id", id);
  }
  return id;
}

async function checkEnabled() {
  try {
    const res = await fetch("/api/guest/status");
    const data = await res.json();
    if (!data.enabled) {
      $("#join-disabled").hidden = false;
      $("#join-form").hidden = true;
    }
  } catch {
    $("#join-disabled").textContent = "Can't reach the server.";
    $("#join-disabled").hidden = false;
  }
}

$("#join-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const err = $("#join-error");
  const btn = $("#join-btn");
  err.hidden = true;
  btn.disabled = true;
  btn.textContent = "Joining…";
  try {
    const res = await fetch("/api/guest/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        invite_code: $("#invite").value.trim(),
        display_name: $("#display-name").value.trim(),
        device_id: deviceId(),
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      err.textContent = data.detail || `Could not join (HTTP ${res.status})`;
      err.hidden = false;
      return;
    }
    localStorage.setItem("wits_guest_token", data.guest_token);
    localStorage.setItem("wits_guest_name", data.display_name || "");
    localStorage.removeItem("wits_token");
    location.href = "/";
  } catch {
    err.textContent = "Can't reach the server — is WITS running?";
    err.hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = "Continue";
  }
});

checkEnabled();
