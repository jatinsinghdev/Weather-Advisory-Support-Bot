# Weather-Advisory Support Bot

This repository contains a deterministic, LangGraph-backed AI agent that cross-references live weather conditions against hardcoded safety rules (SOPs).

## 1. Setup and Run Instructions (Backend & Frontend)

### Prerequisites
- Python 3.9+
- A Groq API Key

### Local Setup
1. Clone this repository and open a terminal in the root directory.
2. Create a virtual environment and activate it (optional but recommended).
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file and insert your key:
   ```text
   GROQ_API_KEY=your_groq_api_key_here
   ```

### Running the Backend (Eval Suite)
The core logic resides in `graph.py`. To execute the backend tests:
```bash
python eval_suite.py
```

### Running the Frontend (Streamlit UI)
To launch the interactive chat interface (`app.py`):
```bash
streamlit run app.py
```

---

## 2. SOPs Format and Rationale
**Form chosen:** JSON (`sops.json` containing 21 policies).
**One-line note on why:** I chose a dynamic JSON structure because it completely decouples safety thresholds from the Python code, allowing policy teams to add new rules (and entirely new weather variables like `cloud_cover`) on-the-fly without requiring an engineer to touch the application logic.

---

## 3. The LangGraph Implementation
The core graph is implemented in `graph.py`. 
- **Separation of Concerns:** The LLM is strictly used for intent extraction (parsing the activity/location) and formatting the final output. The actual safety evaluation is a deterministic Python dictionary match against the `sops.json` thresholds. The LLM cannot hallucinate thresholds.
- **Conversational Memory:** Uses LangGraph's `MemorySaver` bound to a session `thread_id`. The LLM receives the full message history, and the state dictionary seamlessly carries over previously extracted locations and activities (e.g., answering follow-ups like *"What about this evening instead?"*) so the user doesn't have to repeat themselves.
- **Conflict Resolution:** If multiple SOPs apply, the system surfaces *all* of them for full risk transparency, but deterministically sorts them by `severity` (CRITICAL > HIGH > MODERATE > LOW) before injecting them into the LLM prompt.

---

## 4. Eval Suite, Results, and Notes
The test suite is located in `eval_suite.py`.
- **Results:** All 8 tests currently evaluate to `PASS`.
- **Honest Notes on Implementation & Failures:** A major consideration in the requirements was how to handle a "Severe Weather" test without relying on the unpredictable real-world weather in Bhopal holding up on grading day. To solve this and keep the suite stable, I utilized `unittest.mock.patch` to intercept the Open-Meteo API calls during testing. This guarantees that our threshold and severity-override logic is perfectly validated against controlled, deterministic weather data, keeping the test suite reliable long after the real-world storm has passed. No tests currently fail in the mocked environment.
