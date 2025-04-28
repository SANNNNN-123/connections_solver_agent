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

