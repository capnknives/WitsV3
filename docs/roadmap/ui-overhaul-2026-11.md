---
title: "WITS UI and UX Overhaul Plan"
created: "2026-11-22"
last_updated: "2026-11-22"
status: "draft"
---

# WITS UI and UX overhaul plan

**Status:** draft. Nothing here is built. The owner has picked a default look from mockups
(November 22, 2026): **Horizon**, with the others kept as alternative themes. See section 3.

**Scope:** the WITS web app under `web/static/` (chat, side panel, settings, MCP page, guest
join, personality, and the new Inbox). The portfolio site is a separate project and is not covered.

## 1. What the UI is today

Checked against `main` on November 22, 2026.

| Area | What is there | Problem |
|---|---|---|
| Look | Navy background (`#0f1117`), blue accent (`#5b8cff`), system font, 14px rounded corners | It is the default look of an AI chat app. Nothing about it says WITS |
| Icons | Emoji and symbol characters (☰ 🎛 ⬇ ＋ ➤) | They draw differently on every device and cannot be styled |
| Chat | One column of bubbles and a composer | WITS routes, guards, plans and calls tools before it answers, and none of that is visible |
| Side panel | One drawer with four tabs: chats, tools, memory, documents | Heavy forms (memory search, paging, pruning) are squeezed into a drawer |
| Settings | One long page, about 35 KB of HTML with its own inline styles | Hard to find anything; styled apart from the rest |
| Styles | `style.css`, `mcp.css`, and inline `<style>` blocks in three pages | No shared tokens, so pages drift apart |
| Inbox | Its own page with its own styles | Nothing links to it. It does not feel like part of WITS |
| Agents | No page. Creating or revoking an agent token is a PowerShell command | The owner cannot manage agents from the phone |
| Script | `app.js` is one 37 KB file | Hard to change one part without reading all of it |

What already works and should be kept: it is fast, it has no build step, it installs on a phone,
the slash-command menu, streaming replies, and the rule that agent text is only ever shown as text.

## 2. What current practice says

Sources are listed at the end. Four findings shaped this plan.

1. **The products people point to as well designed are dark-first, quiet and dense.** Linear,
   Raycast, Arc and Warp all ship a dark interface as the designed version: a near-black ground,
   hairline borders, one restrained accent, raised surfaces only for things that float, and a
   keyboard path for every action. Linear's own account of its refresh gives the rule as "Don't
   compete for attention you haven't earned."
2. **2026 trend reports split in two directions, and both are useful here.** One is calm and
   structured: fewer decorations, type doing the work, motion only when it means something,
   accessibility planned from the start. The other is expressive: raw schematic layouts, visible
   structure, retro-futurist and terminal looks. Several reports say the same thing about AI:
   it should read as a copilot the user directs, and trust should be made visible.
3. **Chat alone is no longer considered enough for an AI product.** Chat stays, because working
   with a model is iterative. But the writing on AI interfaces agrees that a reply is not a
   message from a person: it has streaming, tool calls, sources and uncertainty, and those need
   their own parts. Purpose-built views beat asking for everything through a text box.
4. **The terminal look is mainstream again, with a known trap.** Full CRT costume (scanlines,
   glow, all-monospace) hurts reading and puts off anyone who is not the author. The projects
   that do it well keep the structure and the type discipline and make the costume a setting.

## 3. Direction

WITS is an instrument the owner runs, not a chat friend. The look is dark, and it is allowed to
be beautiful: light, depth and one strong visual idea per theme.

The first round of mockups followed a rule that said "type and line, not cards and glow". Five
designs from three different models all came back in the same restrained family, and the owner
called it the usual thing. That rule is dropped. The rules that stay are the ones about behavior.

Five rules, each one checkable:

1. **Quiet until something happens.** The resting screen is calm. Strong color and light are
   spent on state: a model loading, a tool running, a decision waiting, an error.
2. **Show the work.** Every reply carries its trace: how it was routed, what the guards said,
   which tools ran and what came back. One tap to open. This is the thing WITS has that a hosted
   chat app does not, and it is the same idea as Trace on the portfolio.
3. **One console.** A command bar reaches every action and every page (Ctrl+K on a keyboard, a
   button on a phone). The slash menu grows into it.
4. **When state changes, the room changes.** Which model is loaded, whether a tool is running,
   whether an agent is connected, whether a decision is waiting: each has a fixed place on screen
   and looks different when it changes. A change that shows nothing is not allowed.
5. **The phone is a first-class seat.** Everything the owner needs away from the desk (Inbox,
   chat, stop, revoke) is reachable with a thumb, and every target is at least 44 px. No emoji
   as controls; icons are drawn from one small set.

### The default theme: Horizon

Near-black ground with one very large lit arc across the top of the screen, like the edge of a
planet with a thin band of atmosphere under it. The name sits in the dark above the line. The
rest of the screen is spare, with cold white text, so the arc carries the look.

The arc is also the state display, which is how rule 4 is met:

- At rest it is a plain cold-white line.
- While a tool runs, a stretch of it lights amber.
- In the Inbox it carries one amber segment per waiting decision.
- Other states (a model loading, the heavy model in use, an error) get their own treatment of
  the same arc, to be settled in package U1.

Amber is the only warm color and always means "running or waiting on you". A coral red is
reserved for warnings. Inner pages use a shallower arc so content keeps its room.

### Alternative themes

All themes share one structure and one token file, so adding one is a set of values plus its
signature element, not a rebuild. A theme may change labels and density as well as color.

| Theme | Where its look comes from | Signature element |
|---|---|---|
| **Horizon** (default) | Scale | The lit arc |
| **Filament** | Light | A glowing wire for the trace, lamps that bloom, film grain |
| **Orrery** | Ornament | Brass linework and the trace drawn as a dial |
| **Relay room** | Restraint | Rules, mono labels, indicator lamps, one amber signal |

Horizon ships first. The others follow as separate small packages after the gate, in whatever
order the owner wants. Two light-ground designs from the mockup round (Field log and a similar
outside one) are not carried forward, because the owner wants a darker theme.

The mockups are on a design canvas outside the repository; the owner has the link.

## 4. What changes for the user

**Shell.** A rail on the left on a wide screen, a bar at the bottom on a phone, with six places:
Chat, Inbox (with a count), Workspace, Memory, Files, Settings. A status strip shows the loaded
model, speed of the last reply, and connection state.

**Chat.** A reply has parts: the answer, a trace line (route, guards, tools, time, model), tool
calls as compact rows that open to show input and result, a stop control while it runs, and a
plain-words error state with one suggested next step. Bubbles go; turns are separated by rule and
label, which reads better at length and on a narrow screen.

**Command bar.** Search and run: new chat, switch model, open a page, find a past chat, revoke an
agent, toggle a setting. Everything in the slash menu is in it.

**Inbox.** Becomes a place in the shell, not a separate page. Same cards and the same rendering
rule. Adds the count badge and a "nothing waiting" state.

**Workspace (new).** The owner's view of what agents are doing: projects, open tasks and who
holds them, recent handoffs, linked artifacts. Read-only at first.

**Agents (new, inside Workspace).** Create an agent token with its scopes and projects, see it
once (with a QR code), list agents with last-seen time, revoke with one tap. This replaces the
PowerShell commands in the setup guide.

**Memory and Files.** Move out of the drawer into full pages with room for search, filters and
paging. Pruning moves behind a confirmation of its own.

**Settings.** Split into sections with a search box. Each setting says what it does in one line.

**Across all of it.** Contrast at WCAG AA, visible focus, reduced-motion respected, empty and
loading states designed, no layout shift while a reply streams.

## 5. How it is built

- **No build step stays.** Plain HTML, CSS and JavaScript served as files, as the repo rules say.
- **One token file.** `tokens.css` holds color, type, spacing, radius and motion as custom
  properties. Every page uses it. A theme is a second set of values.
- **Script split into native modules.** `app.js` becomes ES modules loaded with
  `type="module"`: shell, chat, trace, command bar, and one per page. No bundler.
- **Fonts and icons served locally.** One sans and one mono, self-hosted, and one SVG sprite.
  Nothing is fetched from another site, which keeps the local-first rule and allows a strict
  content security policy.
- **The rendering rule extends to the whole app.** Text from models, tools and agents is written
  with `textContent`. A test checks every script for HTML-parsing sinks, as the Inbox test does.
- **The trace needs one server change.** The stream already carries thinking, action,
  observation and result events. The chat needs them tagged with route and guard outcomes. That
  is the only backend work in this plan and it gets its own contract.

## 6. Packages

Same working rules as the unified plan: a brief per package, one pull request each, small enough
to read, the owner merges. Order matters only where shown.

| # | Package | Builder | Needs | Check |
|---|---|---|---|---|
| U0 | Mockups of chat and Inbox on a phone. **Done:** Horizon picked. Still to draw: Horizon on a desktop, and the shell | Lead | Nothing | Owner signs off the desktop layout |
| U1 | Tokens, base styles, fonts, icon sprite and the Horizon arc as one reusable piece with its states. No page changes yet | Sonnet | U0 | Old pages still render; contrast test passes; the arc shows each state |
| U2 | Shell and navigation, phone and desktop | Sonnet | U1 | Every page reachable; works at 360 px and 1440 px |
| U3 | Trace contract, then chat anatomy and trace | Opus | U2 | Streaming tests; trace matches the server's events |
| U4 | Command bar | Sonnet | U2 | Keyboard-only run-through of every command |
| U5 | Inbox in the shell, Workspace view, Agents page. Review: **high** | Opus | U2, Stage 1 | Token shown once; rendering-rule test; owner routes refuse non-owners |
| U6 | Memory, Files and Settings as full pages | Sonnet | U2 | Existing settings tests pass; search finds every setting |
| U7 | Accessibility and speed pass, then the gate | Sonnet, then an independent review | All | Owner hand-test on phone, laptop and PC |
| U8+ | One alternative theme per package: Filament, Orrery, Relay room | Sonnet | U7 | Switching theme visibly changes the app and breaks nothing |

U3, U4, U5 and U6 touch different files and can run side by side after U2.

**Where this sits in the unified plan.** After the Stage 1 gate. It can run beside Stage 2
(foundations), because Stage 2 is storage and model plumbing and this is `web/static/`. The one
overlap is the chat stream on the server: U3 adds route and guard outcomes to it, so its brief
must be checked against whatever Stage 2 is changing in the same code at the time. `web/server.py`
and `web/static/app.js` are already over the size rule, so new code goes in new files.

**Gate.** From the phone, away from home Wi-Fi: open WITS, ask a question and open its trace,
approve a decision in the Inbox, create an agent and revoke it, and find a setting by search.
No step needs the desk.

## 7. Open decisions for the owner

1. Settled: Horizon is the default; Filament, Orrery and Relay room are alternative themes. Open: the order the alternatives are built in.
2. Whether the status strip should show GPU load and temperature. The readings exist on the PC;
   showing them means a small local endpoint.
3. Whether guests get the new shell or keep a simpler one.

## Sources

- Linear, "Behind the latest design refresh": https://linear.app/blog/behind-the-latest-design-refresh
- Linear, "How we redesigned the Linear UI": https://linear.app/blog/how-we-redesigned-the-linear-ui
- Muzli, "Dark Mode Design Systems": https://feed.muz.li/?p=201477
- Muzli, "Mobile App Design Trends 2026: UI Patterns": https://muz.li/blog/whats-changing-in-mobile-app-design-ui-patterns-that-matter-in-2026/
- Tubik Studio, "7 UI Design Trends of 2026": https://blog.tubikstudio.com/?p=17271
- UX Studio, "UI trends 2026": https://www.uxstudioteam.com/ux-blog/ui-trends-2019
- daisyUI, "18 UI design trends for 2026": https://trends.daisyui.com/ui-design-trends-2026/
- Smashing Magazine, "Design Patterns For AI Interfaces": https://www.smashingmagazine.com/2025/07/design-patterns-ai-interfaces/
- Setproduct, "Designing AI chat interfaces": https://www.setproduct.com/blog/ai-chat-interface-ui-design
- UX Collective, "Beyond chat: 8 core user intents driving AI interaction": https://uxdesign.cc/beyond-chat-8-core-user-intents-driving-ai-interaction-4f573685938a
- "Re-imagining AI interfaces": https://www.lesswrong.com/posts/TiP9iS4iMnHsGvtGg/re-imagining-ai-interfaces
- "The Terminal Renaissance: Designing Beautiful TUIs in the Age of AI": https://dev.to/hyperb1iss/the-terminal-renaissance-designing-beautiful-tuis-in-the-age-of-ai-24do
- "I love terminal aesthetics. Not everyone does.": https://dev.to/micronink/i-love-terminal-aesthetics-not-everyone-does-heres-how-i-solved-that-56ef
