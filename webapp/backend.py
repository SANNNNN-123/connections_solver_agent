import json
import os
import sys
from typing import List
import tempfile
import uuid
import pprint as pp
import logging
import aiosqlite
import asyncio
import argparse

import pandas as pd

# Add the parent directory to the Python path FIRST
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from the parent package
from tools import llm_interface_registry, read_file_to_word_list, extract_words_from_image_file
from workflow_manager import create_webui_workflow_graph
from puzzle_solver import PuzzleState

# Import LLM interfaces
import gemini_tools
import openai_tools
import grok_tools

from langchain_core.runnables import RunnableConfig

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Web App for NYT Connections Solver")
parser.add_argument(
    "--llm_interface",
    type=str,
    default="gemini",
    choices=["gemini", "openai", "grok"],
    help="Set the LLM interface to use (gemini, openai, or grok), default is 'gemini'",
)
parser.add_argument(
    "--db_path",
    type=str,
    default="data/db/vocabulary.db",
    help="Path to store the vocabulary database",
)
args = parser.parse_args()

pp = pp.PrettyPrinter(indent=4)
logger = logging.getLogger(__name__)

# Ensure database directory exists
os.makedirs(os.path.dirname(args.db_path), exist_ok=True)

db_lock = asyncio.Lock()

# read in workflow instructions
with open("webapp/embedvec_webui_workflow_specification.md", "r") as f:
    workflow_instructions = f.read()

workflow_graph = create_webui_workflow_graph()

async def webui_puzzle_setup_function(puzzle_setup_fp: str, config: RunnableConfig) -> List[str]:
    suffix = puzzle_setup_fp.split(".")[-1]
    if suffix == "txt":
        words = read_file_to_word_list(puzzle_setup_fp)
    elif suffix == "png":
        words = await extract_words_from_image_file(puzzle_setup_fp, config)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return words

# setup interface to LLM
selected_interface = llm_interface_registry.get(args.llm_interface)
if selected_interface is None:
    available_interfaces = list(llm_interface_registry._registry.keys())
    raise ValueError(f"LLM interface '{args.llm_interface}' not found. Available interfaces: {available_interfaces}")

llm_interface = selected_interface()

# setup runtime config
runtime_config = {
    "configurable": {
        "thread_id": str(uuid.uuid4()),
        "workflow_instructions": workflow_instructions,
        "llm_interface": llm_interface,
    },
    "recursion_limit": 50,
}

app = FastAPI(
    title="NYT Connections Solver API",
    description="API for solving NYT Connections puzzles using LLMs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000"],  
    allow_credentials=True,
    allow_methods=["GET", "POST"],  
    allow_headers=["Content-Type"], 
)

# Mount static files and templates
templates = Jinja2Templates(directory="webapp")
app.mount("/static", StaticFiles(directory="webapp"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    print("app.get('/')")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/setup-puzzle")
async def setup_puzzle(request: Request):
    print("app.post('/setup-puzzle')")
    data = await request.json()
    puzzle_setup_fp = data.get("setup")
    puzzle_words = await webui_puzzle_setup_function(puzzle_setup_fp, runtime_config)

    initial_state = PuzzleState(
        puzzle_status="initialized",
        current_tool="setup_puzzle",
        tool_status="initialized",
        workflow_instructions=workflow_instructions,
        llm_temperature=0.7,
        vocabulary_db_fp=args.db_path,  # Use the persistent database path
        recommendation_answer_status="",
        recommendation_correct_groups=[],
        found_count=0,
        mistake_count=0,
        recommendation_count=0,
        llm_retry_count=0,
        invalid_connections=[],
        words_remaining=puzzle_words,
    )

    print("\nGenerating vocabulary and embeddings for the words...this may take several seconds ")
    vocabulary = await llm_interface.generate_vocabulary(puzzle_words)
    rows = []
    for word, definitions in vocabulary.items():
        for definition in definitions:
            rows.append({"word": word, "definition": definition})
    df = pd.DataFrame(rows)

    print("\nGenerating embeddings for the definitions")
    embeddings = llm_interface.generate_embeddings(df["definition"].tolist())
    df["embedding"] = [json.dumps(v) for v in embeddings]

    print("\nStoring vocabulary and embeddings in database")
    async with aiosqlite.connect(args.db_path) as conn:  # Use the persistent database path
        async with db_lock:
            cursor = await conn.cursor()
            await cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS vocabulary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT,
                    definition TEXT,
                    embedding TEXT
                )
                """
            )
            await conn.executemany(
                "INSERT INTO vocabulary (word, definition, embedding) VALUES (?, ?, ?)",
                df.values.tolist(),
            )
            await conn.commit()

    async for chunk in workflow_graph.astream(initial_state, runtime_config, stream_mode="values"):
        pass

    return JSONResponse({"status": "success in getting puzzle words", "puzzle_words": puzzle_words})

@app.post("/update-solution")
async def update_solution(request: Request):
    print("app.post('/update-solution')")
    data = await request.json()
    user_response = data.get("user_response")

    if user_response in ["y", "g", "b", "p"]:
        logger.info(f"User response: {user_response}")
        status_message = "Correct recommendation"
    elif user_response in ["n", "o"]:
        logger.info(f"User response: {user_response}")
        status_message = "Incorrect recommendation"

    current_state = workflow_graph.get_state(runtime_config)
    logger.debug(f"\nCurrent state: {current_state}")
    logger.info(f"\nNext action: {current_state.next}")

    if current_state.next[0] == "apply_recommendation":
        workflow_graph.update_state(
            runtime_config,
            {
                "recommendation_answer_status": user_response,
            },
        )
    else:
        raise RuntimeError(f"Unexpected next action: {current_state.next[0]}")

    async for chunk in workflow_graph.astream(None, runtime_config, stream_mode="values"):
        logger.debug(f"\nstate: {workflow_graph.get_state(runtime_config)}")
        pass

    current_state = workflow_graph.get_state(runtime_config)

    response_dict = {
        "words_remaining": current_state.values["words_remaining"],
        "connection_reason": "",
        "recommeded_words": "",
        "found_count": current_state.values["found_count"],
        "mistake_count": current_state.values["mistake_count"],
        "found_groups": current_state.values["recommendation_correct_groups"],
        "invalid_groups": [x[1] for x in current_state.values["invalid_connections"]],
    }

    if current_state.values["found_count"] == 4:
        response_dict["status"] = "PUZZLE SOLVED!!!"
    elif current_state.values["mistake_count"] == 4:
        response_dict["status"] = "FAILED TO SOLVE PUZZLE!!!"

    return JSONResponse(response_dict)

@app.post("/generate-next")
async def generate_next():
    print("app.post('/generate-next')")
    current_state = workflow_graph.get_state(runtime_config)
    return JSONResponse(
        {
            "status": "Next recommendation will be generated here",
            "recommended_words": sorted(current_state.values["recommended_words"]),
            "connection_reason": current_state.values["recommended_connection"],
            "active_recommender": current_state.values["current_tool"],
        }
    )

@app.post("/manual-override")
async def manual_override():
    print("app.post('/manual-override')")
    current_state = workflow_graph.get_state(runtime_config)
    workflow_graph.update_state(
        runtime_config,
        {
            "puzzle_status": "manual_override",
        },
    )
    return JSONResponse({"status": "success"})

@app.post("/confirm-manual-override")
async def confirm_manual_override(request: Request):
    print("app.post('/confirm-manual-override')")
    data = await request.json()
    words = data.get("words", [])
    current_state = workflow_graph.get_state(runtime_config)
    if set(words).issubset(set(current_state.values["words_remaining"])):
        workflow_graph.update_state(
            runtime_config,
            {"recommended_words": words},
        )
        status_message = "success"
    else:
        status_message = "error"
    return JSONResponse({"status": status_message})

@app.post("/terminate")
async def terminate(background_tasks: BackgroundTasks):
    print("app.post('/terminate')")
    def shutdown():
        os._exit(0)
    background_tasks.add_task(shutdown)
    return JSONResponse({"status": "terminating"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webapp.backend:app", host="127.0.0.1", port=8000, reload=True) 