# Connection Solver AI Agent

This is a NYT Connection Solver AI agent that helps solve connection daily puzzle
https://www.nytimes.com/games/connections


## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd connection_solver/V2
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
```bash
pip install -r requirements.txt
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
   ```bash
   python main.py
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
   ```bash
   python webapp/backend.py
   ```

3. **Access the application:**
   - **API Documentation:** Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.
   - **Web Interface:** Open `webapp/index.html` directly in your browser.

