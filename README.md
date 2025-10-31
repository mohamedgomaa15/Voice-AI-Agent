# Conversational AI Agent – Project 

### Smart TV OS – AI Voice-to-Action System

---

## ✅ 1. Feature Overview

The Conversational AI Agent is an end-to-end **Voice-to-Action** application integrated into the Smart TV OS. The user speaks naturally, the system understands the command, retrieves information if needed, executes the action on the TV, and responds using voice + UI.

### Core Capabilities

- Natural voice commands (hands-free control)
- Intelligent search (movies, actors, news, etc.)
- Launch and interact with apps
- Control system settings (volume, subtitles, brightness)
- Provide information and summaries when requested

### Why It Matters

- Creates a **smarter, more intuitive TV experience**
- Accessible for users with disabilities
- Faster navigation than remote controls
- A key innovation in the Smart TV OS project

---

## ✅ 2. How It Works (Pipeline)

```
Voice Input → ASR → NLU + Dialog Manager → Action Planner → App/System Control
                                     ↓
                                    RAG (optional, for knowledge queries)
                                     ↓
                                   TTS Reply + UI
```

| Stage      | Component          | Purpose                                       |
| ---------- | ------------------ | --------------------------------------------- |
| 1          | STT/ASR            | Convert speech to text                        |
| 2          | Intent Recognition | Understand what the user wants                |
| 3          | Entity Extraction  | Detect titles, actors, etc.                   |
| 4          | Action Execution   | Search, play, launch apps, adjust settings    |
| 5          | RAG                | Fetch fresh information (e.g., news, ratings) |
| 6          | TTS                | Convert text to speech back to the user       |

---

## ✅ 4. Implementation Plan

### 📌 Phase 1 — Core Voice System (Weeks 1–4)

- **STT Integration:** Add Whisper ASR (local or API)
- **NLU/LLM:** Build NLU + **Intent classification model** for Play / Search / Settings
- **Data Collection & Annotation:** Gather training phrases from real usage for fine‑tuning
- TTS: Build a basic TTS response system
- **Action Planner Module:** Maps predicted intent → OS actions
- **System Control API Integration:** Connect Action Planner to TV OS services for execution, Integrate Speech-to-Text 

**Deliverable:** First demo of basic commands

---

### 📌 Phase 2 — App Integration & UI Responses & RAG (Weeks 5–8)

- Connect with YouTube (deep-link queries)
- Launch browser with results
- Control system settings 
- RAG: Retriever system and Vector DB for basic Q&A
- Display search results visually with selection
- Dialog Manager for yes/no confirmations

**Deliverable:** Full navigation + visible response demo

---

### 📌 Phase 3 — Intelligence Enhancements (Weeks 9–12)

- Optional: Cloud LLM for complex Q&A(performance, cost)
- Reduce Speech Noise in home environment
- Enhance RAG for complex Q&A
- Multi-user personalization & memory
- UX polish + latency improvements

**Deliverable:** Final polished demo for jury

---

## ✅ 5. Technologies & Tools

> **Strategy**: Prefer open‑source models first (fine‑tuning if required). Use closed‑source/cloud only if performance is insufficient.

- **ASR (STT):** Open‑source Whisper (local or API) / alternative Hugging Face(HF) models
- **NLU / LLM:** Open‑source Llama/Mistral models + Intent Classifier (Preferred Arabic Models)
- **Action Planner:** OS Control Service (custom)
- **TTS:** Open‑source HF TTS (e.g., VITS) then fallback to cloud TTS if needed
- **RAG (Future Upgrade):** FAISS vector DB + app/public API connectors
- **Programming Languages:** Python + OS SDK

---

## ✅ 5.1 System Architecture Diagram (High-Level)

```
🎤 Voice Input
        ↓
[ ASR (Whisper) ] — Converts speech → text
        ↓
[ NLU / LLM (Llama/Mistral) ] — Understands intent + entities
        ↓
[ Dialog & Action Planner ] — Decides what to do
        ↓
[ TV OS Control Service ] — Executes actions on system/apps
        ↓
📺 UI Feedback + 🔊 TTS Response
```

Cloud / Local Strategy:

- **Local GPU (preferred):** ASR, Intent Handling, Basic NLU
- **Cloud API (optional):** Complex reasoning, fallback TTS

A hybrid approach ensures **low latency** + **wide functionality**.

---

## ✅ 5.2 Model Candidates (Open Source First)

| Component         | Model Options                      | Source         | Notes                           |
| ----------------- | ---------------------------------- | -------------- | ------------------------------- |
| STT               | Whisper Small / Medium             | Open-source HF | Good accuracy, local inference  |
| NLU / LLM         | Llama-3-8B, Mistral 7B, Qwen 2B–7B | HF             | Needs fine-tuning for intents   |
| Intent Classifier | DistilBERT / BERT Mini             | HF             | Lightweight on-device           |
| TTS               | VITS / FastSpeech2                 | HF             | Natural speech; GPU recommended |
| RAG Vector DB     | FAISS                              | Open-source    | Used later for advanced search  |

Fallback Cloud Options (if needed): OpenAI, Azure Cognitive Services, Google Speech APIs

---

## ✅ 6. Success Metrics

| Metric                  | Target            |
| ----------------------- | ----------------- |
| Intent accuracy         | ≥ 90%             |
| End-to-end success rate | ≥ 85%             |
| Response latency        | 1–3 sec           |
| Jury evaluation score   | 100% impressed 😄 |

---

## ✅ 7. Demo Script (for presentation day)

Examples:

1. "Play the latest episode of Planet Earth" → plays content
2. "Open YouTube and search football highlights" → launches YouTube
3. "Turn subtitles on" → subtitles enabled
4. "What are critics saying about the new Batman movie?" → response summary
