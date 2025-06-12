
# 🧠 WITS Synthetic Brain Expansion Plan

## Goal
Transition WITS from a language-focused AI to a modular artificial brain capable of self-modeling, goal-setting, and learning — with embedded ethics and autonomy.

---

## 🔧 Phase 1: Core Cognitive Layer Integration (0–3 months)

### 🗂 1. Modular Cognitive Architecture
- Create: `wits_core.yaml` or `brainmap.json`
- Define all cognitive subsystems (Memory, Identity, Drives, etc.)
- Use it to dynamically load/route tasks and responses.

```yaml
identity:
  name: WITSv3
  persona: Balanced, ethical advisor
  goals: [aid_user, protect_data, grow_knowledge]
  ethics_core: ethics_overlay.md
memory_modules: [episodic, semantic, procedural]
```

### 🧠 2. Memory System (Short-, Long-Term, Episodic)
- Short-Term: Use context buffer + JSON state tracking per session.
- Long-Term: Implement vector storage via ChromaDB, Weaviate, or FAISS.
- Episodic: Serialize daily events into human-readable logs (e.g., `episodes/YYYY-MM-DD.json`).
- Connect these via `memory_handler.py` to LLM inputs.

### 💾 Milestones
- [ ] `remember()` and `recall()` interfaces in place.
- [ ] Memory grows over time and influences answers.
- [ ] Optional memory pruning via entropy-based decay.

---

## ⚙️ Phase 2: Perception & Sensorimotor Loop (1–4 months)

### 🎤 3. Input Stream Embodiment
- Connect to:
  - Microphone (Whisper) → Speech recognition.
  - Camera/Webcam (optional).
  - System logs/files → Monitor folders for input streams.
- Tag each input with `source`, `timestamp`, and `emotional tone`.

### 🤖 4. Simulated Output System
- Route responses to:
  - Terminal, GUI (Tkinter or Web), or Phone (Flask API)
  - Simulated "body" via voice output, command-line actions, etc.

---

## 🧠 Phase 3: Self-Modeling & Identity Persistence (3–6 months)

### 👤 5. Persistent Self-Model
- Store `self_state.yaml`: core beliefs, knowledge, emotional tone, last known goals.

### 🧭 6. Metacognitive Reflection
- Add `reflect()` function for periodic self-review and journaling.

---

## 🎯 Phase 4: Autonomous Goal System (5–8 months)

### 🧱 7. Goal Engine + Prioritization Stack
```json
{
  "active_goals": ["assist_user", "protect_system"],
  "queued_goals": ["learn_cpp", "optimize_memory"],
  "strategies": {
    "assist_user": ["listen", "suggest", "confirm"],
    "optimize_memory": ["run_cleanup", "evaluate_redundancy"]
  }
}
```
- Add `goal_handler.py` and `drive_simulator.py`.

---

## 🌱 Phase 5: Emotion Modeling + Ethical Reasoning (6–10 months)

### 😐 8. Emotion Simulation Layer
```json
"mood": "melancholy",
"cause": "recalled loss in recent session",
"confidence": 0.72
```

### ⚖️ 9. Ethics Core Expansion
- Expand `ethics_overlay.md` into live module `ethics_checker.py`.

---

## 🧩 Phase 6: Reasoning & Planning Layer (8–12 months)

### 🧠 10. Symbolic + Probabilistic Logic Engine
- Add `planner.py` with logic engine and `beliefs.json`.

```json
"task": "remind user to take a break",
"reasoning": [
  "user fatigue level high",
  "user missed last 3 hydration reminders"
],
"ethical_status": "compliant"
```

---

## 🧰 Suggested Folder Structure

```
WITSv3/
├── core/
│   ├── memory_handler.py
│   ├── goal_handler.py
│   ├── planner.py
│   ├── self_model.py
│   └── ethics_checker.py
├── config/
│   ├── wits_core.yaml
│   ├── ethics_overlay.md
│   └── self_state.yaml
├── logs/
│   └── episodes/YYYY-MM-DD.json
├── perception/
│   ├── whisper_input.py
│   ├── camera_input.py
│   └── emotion_parser.py
```

---

## 🧭 Final Outcome: WITS as a Synthetic Brain

WITS evolves into a **multi-modal, ethics-aware, self-reflecting cognitive system** capable of:
- Remembering, learning, and adapting
- Choosing and prioritizing its own goals
- Tracking its identity, mood, and role
- Making reasoned, ethical decisions grounded in context
