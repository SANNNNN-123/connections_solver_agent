# Connection Solver AI Agent

This is a NYT Connection Solver AI agent that helps solve connection daily puzzle
https://www.nytimes.com/games/connections


## Installation

1. Clone the repository:
```bash
git clone "https://github.com/SANNNNN-123/connections_solver_agent.git"
cd connection_solver_agent
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv myenv
myenv\Scripts\activate

# Linux/Mac
python -m venv myenv
source myenv/bin/activate
```

3. Install the required dependencies:
I usually used blazing fast UV module to install
```bash
pip install uv
uv pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key

# Google API Key
GOOGLE_API_KEY=your_google_api_key

# Grok API Key
XAI_API_KEY=your_grok_api_key
```

## Running the Application

You can run the Connection Solver AI Agent in two ways: via the terminal or using the web interface.

### 1. Running via Terminal

1. **Activate your virtual environment:**
   - **Windows:**
     ```bash
     myenv\Scripts\activate
     ```
   - **Linux/Mac:**
     ```bash
     source myenv/bin/activate
     ```

2. **Run the main script:**
   You can specify which LLM interface to use by setting the `--llm_interface` argument to `gemini`, `openai`, or `grok`:
   ```bash
   # For Gemini
   python main.py --llm_interface gemini

   # For OpenAI
   python main.py --llm_interface openai

   # For Grok
   python main.py --llm_interface grok
   ```

---

### 2. Running via Web Interface

1. **Activate your virtual environment:**
   - **Windows:**
     ```bash
     myenv\Scripts\activate
     ```
   - **Linux/Mac:**
     ```bash
     source myenv/bin/activate
     ```

2. **Start the backend server:**
   You can specify which LLM interface to use by setting the `--llm_interface` argument to `gemini`, `openai`, or `grok`:
   ```bash
   # For Gemini
   python webapp/backend.py --llm_interface gemini

   # For OpenAI
   python webapp/backend.py --llm_interface openai

   # For Grok
   python webapp/backend.py --llm_interface grok
   ```

3. **Access the application:**
   - **API Documentation:** Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.
   - **Web Interface:** Open `webapp/index.html` directly in your browser.

## How It Works

The Connection Solver AI Agent uses an AI agent workflow pattern with LangGraph and LangChain to solve NYT Connections puzzles. Here's the complete workflow:

```
┌─────────────────────────────────────────────────────────┐
│ 1. INPUT: 16 words from NYT Connections puzzle          │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 2. GENERATE VOCABULARY: LLM creates multiple meanings   │
│    per word (noun, verb, adjective, etc.)               │
│    Example: "bank" → ["noun: financial institution",  │
│                        "noun: river edge",              │
│                        "verb: to deposit money"]        │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 3. GENERATE EMBEDDINGS: Create vector embeddings       │
│    for each definition (not the word itself)            │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 4. STORE IN DATABASE: Save word, definition, embedding  │
│    in SQLite database (vocabulary.db)                   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 5. FIND GROUPS: Two strategies work in parallel:        │
│                                                  │
│    A. EmbedVec Recommender:                            │
│       - Calculate cosine similarity between embeddings │
│       - Find groups of 4 words with high similarity     │
│       - LLM explains the connection                    │
│                                                  │
│    B. LLM Recommender:                                 │
│       - Directly asks LLM to find related groups       │
│       - Uses semantic understanding                    │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 6. LANGGRAPH ORCHESTRATION:                             │
│    - Manages state (words remaining, mistakes, etc.)   │
│    - Routes between recommenders                       │
│    - Handles user feedback                             │
│    - Adapts strategy based on results                   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 7. USER FEEDBACK: Accept/reject suggestions             │
│    - If correct: Remove words, continue                 │
│    - If wrong: Try different strategy                   │
│    - Loop until puzzle solved or max mistakes           │
└─────────────────────────────────────────────────────────┘
```

