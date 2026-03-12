# AI Research Agent with LangGraph & RAG

A powerful AI agent that browses the web, indexed findings in a vector database, and synthesizes structured research reports.

## Prerequisites
- **Python 3.10+** (Recommended: 3.13)
- **Google API Key**: Get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Setup Instructions

### 1. Copy the Files
Ensure you have the following files in your project folder:
- `main.py`
- `nodes.py`
- `state.py`
- `tools.py`
- `requirements.txt`

### 2. Install Dependencies
Open your terminal (PowerShell or Command Prompt) and run:
```powershell
pip install -r requirements.txt
```

### 3. Configure the API Key
Create a file named `.env` in the same folder and add your key:
```env
GOOGLE_API_KEY=your_key_here_AIza...
```

### 4. Run the Agent
Execute the main script:
```powershell
python main.py
```

## How it Works
1. **Search**: The agent uses Gemini 1.5/2.0 to generate search queries and DuckDuckGo to find information.
2. **Indexer**: Results are stored in an in-memory Qdrant database.
3. **Research**: The agent performs semantic retrieval from the database.
4. **Report**: A final, structured Markdown report is generated.

## Technical Notes
- **Database**: Currently uses `:memory:` (reset on every run). To make it persistent, edit `tools.py` and change `":memory:"` to a local path like `./qdrant_data`.
- **Model**: Uses `gemini-1.5-flash` for high speed and broad availability.
