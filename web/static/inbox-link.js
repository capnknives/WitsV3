/* Shows the Inbox link in the main page header with the number of waiting decisions.
   Hidden on any failure (no token, 401/403/404, network error). Text via textContent only. */
"use strict";

(function () {
  const REFRESH_MS = 60000;
  const link = document.getElementById("inbox-link");
  const badge = document.getElementById("inbox-count");
  if (!link || !badge) return;

  function hide() {
    link.hidden = true;
  }

  function show(count) {
    link.hidden = false;
    if (count > 0) {
      badge.textContent = String(count);
      badge.hidden = false;
      link.setAttribute("aria-label", "Inbox, " + count + " waiting");
    } else {
      badge.textContent = "";
      badge.hidden = true;
      link.setAttribute("aria-label", "Inbox");
    }
  }

  async function refresh() {
    const token = localStorage.getItem("wits_token") || "";
    if (!token) {
      hide();
      return;
    }
    try {
      const res = await fetch("/api/workspace/approvals", {
        headers: { Authorization: "Bearer " + token },
      });
      if (res.status !== 200) {
        hide();
        return;
      }
      const data = await res.json();
      const list = data && Array.isArray(data.approvals) ? data.approvals : [];
      show(list.length);
    } catch (err) {
      hide();
    }
  }

  document.addEventListener("visibilitychange", function () {
    if (document.visibilityState === "visible") refresh();
  });
  setInterval(refresh, REFRESH_MS);
  refresh();
})();
