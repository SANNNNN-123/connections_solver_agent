#!/usr/bin/env python3

import argparse
import asyncio
from functools import partial
import logging
import pickle
import pprint
import json
import os
import uuid
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.runnables import RunnableConfig

from workflow_manager import run_workflow, create_workflow_graph
from puzzle_solver import PuzzleState

from tools import interact_with_user, manual_puzzle_setup_prompt, llm_interface_registry

from gemini_tools import LLMGeminiInterface
from grok_tools import LLMGrokInterface

try:
    from openai_tools import LLMOpenAIInterface
except ImportError:
    print("Warning: OpenAI tools not available")

# Fix import to work when running directly or as a module
try:
    # When running as a module
    from . import __version__
except ImportError:
    # When running directly as a script
    import sys
    import importlib.util
    spec = importlib.util.spec_from_file_location("__init__", os.path.join(os.path.dirname(__file__), "__init__.py"))
    init = importlib.util.module_from_spec(spec)
    sys.modules["__init__"] = init
    spec.loader.exec_module(init)
    __version__ = init.__version__

# create logger
logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter(indent=4)


def configure_logging(log_level):
    # get numeric value of log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Configure the logging settings
    logging.basicConfig(
        level=numeric_level,  # Set the logging level
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Define the log format
        handlers=[
            logging.FileHandler("app.log"),  # Log to a file
            # logging.StreamHandler(),  # Optional: Log to the console as well
        ],
    )


pp = pprint.PrettyPrinter(indent=4)


async def main(puzzle_setup_function: callable = None, puzzle_response_function: callable = None):
    print(f"Running Connection Solver Agent with EmbedVec Recommender {__version__}")

    parser = argparse.ArgumentParser(description="Set logging level for the application.")

    parser.add_argument(
        "--llm_interface",
        type=str,
        default="gemini",
        help="Set the LLM interface to use (e.g., gemini, openai, grok), default is 'openai'",
    )

    parser.add_argument(
        "--log-level", type=str, default="INFO", help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    parser.add_argument(
        "--trace", action="store_true", default=False, help="Enable langsmith tracing for the application."
    )
    parser.add_argument("--snapshot_fp", type=str, default=None, help="File path to save snapshot data")
    parser.add_argument("--db_path", type=str, default="data/db/vocabulary.db", 
                       help="Path to store the vocabulary database")
    parser.add_argument("--clean_db", action="store_true", default=False, 
                       help="Delete the existing database before starting")

    # Parse arguments
    args = parser.parse_args()

    # configure the logger
    configure_logging(args.log_level)

    # setup for tracing if specified
    if args.trace:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        # os.environ["LANGCHAIN_PROJECT"] = "Agent-With-LangGraph"
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    # read in workflow instructions
    with open("embedvec_workflow_specification.md", "r") as f:
        workflow_instructions = f.read()

    workflow_graph = create_workflow_graph()

    try:
        workflow_graph.get_graph().draw_png("images/connection_solver_embedvec_graph.png")
        print("Graph visualization saved to images/connection_solver_embedvec_graph.png")
    except ImportError:
        print("Warning: pygraphviz not installed. Graph visualization will be skipped.")
        print("To install pygraphviz, run: pip install pygraphviz")
        print("Note: On Windows, you may need to install Graphviz first from https://graphviz.org/download/")

    # Get the LLM interface based on the argument
    llm_interface = llm_interface_registry.get(args.llm_interface)()

    # Special handling for different interfaces if needed
    if args.llm_interface == "gemini":
        print("Using Gemini interface with special configuration")
        # Any Gemini-specific setup can go here
    
    runtime_config = RunnableConfig(
        configurable={
            "thread_id": str(uuid.uuid4()),
            "workflow_instructions": workflow_instructions,
            "llm_interface": llm_interface,
        },
        recursion_limit=50,
    )

    setup_this_puzzle = partial(manual_puzzle_setup_prompt, runtime_config)

    # Use a persistent database file instead of a temporary one
    db_path = args.db_path
    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Optionally clean the database if requested
    if args.clean_db and os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"Removed existing database at {db_path}")
        except Exception as e:
            print(f"Warning: Could not remove database: {e}")
    
    initial_state = PuzzleState(
        puzzle_status="",
        current_tool="",
        tool_status="",
        workflow_instructions=None,
        llm_temperature=0.7,
        vocabulary_db_fp=db_path,
        recommendation_correct_groups=[],
    )

    if args.trace:
        with tracing_v2_enabled("Connection_Solver_Agent"):
            result = await run_workflow(
                workflow_graph,
                initial_state,
                runtime_config,
                puzzle_setup_function=setup_this_puzzle,
                puzzle_response_function=interact_with_user,
            )
    else:
        result = await run_workflow(
            workflow_graph,
            initial_state,
            runtime_config,
            puzzle_setup_function=setup_this_puzzle,
            puzzle_response_function=interact_with_user,
        )

    # Dump snapshot if the flag is set
    if args.snapshot_fp:
        snapshot = list(workflow_graph.checkpointer.list(runtime_config))
        # save as pickle file
        with open(args.snapshot_fp, "wb") as f:
            pickle.dump(snapshot, f)
        # log the snapshot
        print(f"Snapshot: {args.snapshot_fp}")
        logger.info(f"Snapshot: {args.snapshot_fp}")

    print("\nFOUND SOLUTIONS")
    pp.pprint(result)


if __name__ == "__main__":
    asyncio.run(main(manual_puzzle_setup_prompt, interact_with_user))