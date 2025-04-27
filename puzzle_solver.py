import asyncio
import aiosqlite
from dataclasses import dataclass, field
import json
import logging
import pprint as pp
import random
from typing import List, TypedDict, Optional, Tuple
import itertools


import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from tools import compute_group_id  # , chat_with_llm

logger = logging.getLogger(__name__)


MAX_ERRORS = 6
RETRY_LIMIT = 8

db_lock = asyncio.Lock()


# used by the embedvec tool to store the candidate groups
@dataclass
class ConnectionGroup:
    group_metric: float = field(
        default=0.0, metadata={"help": "Average cosine similarity of all combinations of words in the group"}
    )
    root_word: str = field(default="", metadata={"help": "Root word of the group"})
    candidate_pairs: list = field(default_factory=list, metadata={"help": "List of candidate word with definition"})
    group_id: Optional[str] = field(default=None, metadata={"help": "Checksum identifer for the group"})

    def add_entry(self, word, connection):
        if len(self.candidate_pairs) < 4:
            self.candidate_pairs.append((word, connection))
            if len(self.candidate_pairs) == 4:
                self.group_id = compute_group_id(self.get_candidate_words())
        else:
            raise ValueError("Group is full, cannot add more entries")

    def get_candidate_words(self):
        sorted_pairs = sorted(self.candidate_pairs, key=lambda x: x[0])
        return [x[0] for x in sorted_pairs]

    def get_candidate_connections(self):
        sorted_pairs = sorted(self.candidate_pairs, key=lambda x: x[0])

        # strip the part of speech tag at the beginning of the connection, which looks like "noun:" or "verb:" etc.
        # find the first colon and take the substring after it
        stripped_connections = [x[1].split(":", 1)[1].strip() if ":" in x[1] else x[1] for x in sorted_pairs]

        return stripped_connections

    def __repr__(self):
        return_string = f"group metric: {self.group_metric}, "
        return_string += f"root word: {self.root_word}, group id: {self.group_id}\n"
        return_string += f"candidate group: {self.get_candidate_words()}\n"
        for connection in self.get_candidate_connections():
            return_string += f"\t{connection}\n"

        return return_string

    # method to determine if the group is equal to another group
    def __eq__(self, other):
        return set(self.get_candidate_words()) == set(other.get_candidate_words())


@dataclass
class RecommendedGroup:
    words: List[str]
    connection_description: str

    def __repr__(self):
        return f"Recommended Group: {self.words}\nConnection Description: {self.connection_description}"


# define the state of the puzzle
class PuzzleState(TypedDict):
    puzzle_status: str = ""
    tool_status: str = ""
    current_tool: str = ""
    vocabulary_db_fp: Optional[str] = None
    tool_to_use: str = ""
    words_remaining: List[str] = []
    invalid_connections: List[Tuple[str, List[str]]] = []
    recommended_words: List[str] = []
    recommended_connection: str = ""
    recommended_correct: bool = False
    recommendation_answer_status: Optional[str] = None
    recommendation_correct_groups: Optional[List[List[str]]] = []
    found_yellow: bool = False
    found_greeen: bool = False
    found_blue: bool = False
    found_purple: bool = False
    mistake_count: int = 0
    llm_retry_count: int = 0
    found_count: int = 0
    recommendation_count: int = 0
    llm_temperature: float = 1.0
    puzzle_checker_response: Optional[str] = None


async def setup_puzzle(state: PuzzleState, config: RunnableConfig) -> PuzzleState:
    logger.info("Entering setup_puzzle:")
    logger.debug(f"\nEntering setup_puzzle State: {pp.pformat(state)}")

    state["current_tool"] = "setup_puzzle"
    print(f"\nENTERED {state['current_tool'].upper()}")

    # initialize the state
    state["puzzle_status"] = "initialized"
    state["tool_status"] = "initialized"
    state["invalid_connections"] = []
    state["mistake_count"] = 0
    state["found_count"] = 0
    state["recommendation_count"] = 0
    state["llm_retry_count"] = 0
    state["recommended_words"] = []

    # read in pre-built vocabulary for testing
    # state["vocabulary_df"] = pd.read_pickle("src/agent_testbed/word_list1.pkl")

    # generate vocabulary for the words
    print("\nGenerating vocabulary and embeddings for the words...this may take several seconds ")
    vocabulary = await config["configurable"]["llm_interface"].generate_vocabulary(state["words_remaining"])

    # Convert dictionary to DataFrame
    rows = []
    for word, definitions in vocabulary.items():
        for definition in definitions:
            rows.append({"word": word, "definition": definition})
    df = pd.DataFrame(rows)

    # Generate embeddings
    print("\nGenerating embeddings for the definitions")
    embeddings = config["configurable"]["llm_interface"].generate_embeddings(df["definition"].tolist())
    # convert embeddings to json strings for storage
    df["embedding"] = [json.dumps(v) for v in embeddings]

    # store the vocabulary in external database
    print("\nStoring vocabulary and embeddings in external database")

    async with aiosqlite.connect(state["vocabulary_db_fp"]) as conn:
        async with db_lock:
            cursor = await conn.cursor()
            # create the table
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

    logger.info("Exiting setup_puzzle:")
    logger.debug(f"\nExiting setup_puzzle State: {pp.pformat(state)}")

    return state


def get_candidate_words(df: pd.DataFrame) -> list:
    """
    Generate a list of candidate word groups based on cosine similarity of their embeddings.

    Args:
        df (pd.DataFrame): DataFrame containing words and their corresponding embeddings. Dataframe should have two columns: 'word', 'definition' and 'embedding', in that order.

    Returns:
        list: A list of unique candidate word groups sorted by their group metric in descending order.
    """

    candidate_list = []

    # Ensure all embeddings have the same dimensionality
    embeddings = df["embedding"].tolist()
    
    # Convert embeddings to numpy arrays for consistency
    import numpy as np
    
    # Check if embeddings are numpy arrays already
    if embeddings and not isinstance(embeddings[0], np.ndarray):
        embeddings_array = np.array([np.array(emb, dtype=np.float32) for emb in embeddings], dtype=np.float32)
    else:
        embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # create cosine similarity matrix for all pairs of the vectors
    cosine_similarities = cosine_similarity(embeddings_array)
    print(cosine_similarities.shape)

    # for each row in the cosine similarity matrix, sort by the cosine similarity
    sorted_cosine_similarites = np.argsort(cosine_similarities, axis=1)
    print(sorted_cosine_similarites.shape)

    # group of words that are most similar to each other
    for r in range(df.shape[0]):

        # get the top 3 closest words that are not the same as the current word and are not already connected
        connected_words = set()
        top3 = []
        for i in range(sorted_cosine_similarites.shape[1] - 2, 0, -1):
            c = sorted_cosine_similarites[r, i]

            # make sure the word is not already connected and not the current word
            if df.iloc[c, 0] not in connected_words and df.iloc[c, 0] != df.iloc[r, 0]:
                connected_words.add(df.iloc[c, 0])
                top3.append(c)
            if len(connected_words) == 3:
                break

        # create candidate group for the current word and the top 3 closest words
        if df.iloc[r, 0] not in connected_words and len(connected_words) == 3:
            candidate_group = ConnectionGroup()
            candidate_group.group_metric = cosine_similarities[r, top3].mean()
            candidate_group.root_word = df.iloc[r, 0]
            candidate_group.add_entry(df.iloc[r, 0], df.iloc[r, 1])

            for c in top3:
                candidate_group.add_entry(df.iloc[c, 0], df.iloc[c, 1])

            combinations = list(itertools.combinations([r] + top3, 2))
            candidate_group.group_metric = np.array([cosine_similarities[r, c] for r, c in combinations]).mean()

            candidate_list.append(candidate_group)

    # sort the candidate list by the group metric in descending order
    candidate_list.sort(key=lambda x: x.group_metric, reverse=True)

    # remove duplicate groups
    found_groups = set()
    unique_candidate_list = []
    for candidate in candidate_list:
        if candidate.group_id not in found_groups:
            unique_candidate_list.append(candidate)
            found_groups.add(candidate.group_id)

    return unique_candidate_list


async def one_away_analyzer(
    state: PuzzleState, one_away_group: List[str], words_remaining: List[str], config: RunnableConfig
) -> List[Tuple[str, List[str]]]:
    print("\nENTERED ONE-AWAY ANALYZER")
    print(f"found count: {state['found_count']}, mistake_count: {state['mistake_count']}")

    single_topic_groups = []
    possible_anchor_words_list = list(itertools.combinations(one_away_group, 3))

    async def process_anchor_words(anchor_list: List[str]) -> List[str]:
        anchor_words = ", ".join(anchor_list)
        response = await config["configurable"]["llm_interface"].analyze_anchor_words_group(anchor_words)
        return anchor_list, response

    single_topic_groups = await asyncio.gather(
        *[process_anchor_words(anchor_list) for anchor_list in possible_anchor_words_list]
    )

    single_topic_groups = [
        RecommendedGroup(
            words=x[0],
            connection_description=x[1]["explanation"],
        )
        for x in single_topic_groups
        if x[1]["response"] == "single"
    ]

    print(f"\n>>>Number of single topic groups: {len(single_topic_groups)}")
    if len(single_topic_groups) > 1:
        # if more than one single topic group is found, select one at random
        print(f"More than one single-topic group recommendations, selecting one at random.")
        selected_word_group = random.choice(single_topic_groups)
    elif len(single_topic_groups) == 1:
        # if only one single topic group is found, select that one
        print(f"Only one single-topic group recommendation found.")
        selected_word_group = single_topic_groups[0]
    else:
        # if no single topic groups are found, select None
        print(f"No single-topic group recommendations found.")
        selected_word_group = None

    if selected_word_group:
        print(f"\n>>>Selected single-topic group:\n{selected_word_group}")
        # remove original one-away invalid group from the remaining word list
        words_to_test = [x for x in words_remaining if x not in one_away_group]
        anchor_words = ",".join(selected_word_group.words)
        candidate_words_remaining = ", ".join(words_to_test)
        logger.info(f"user prompt input:\n {anchor_words}\n{candidate_words_remaining}")

        response = await config["configurable"]["llm_interface"].generate_one_away_recommendation(
            anchor_words, candidate_words_remaining
        )

        logger.info(response)
        new_group = list(selected_word_group.words) + [response["word"]]
        one_away_group_recommendation = RecommendedGroup(
            words=new_group, connection_description=response["explanation"]
        )
        print(f"\n>>>One-away group recommendations:")
        logger.info(one_away_group_recommendation)
    else:
        # if no single topic groups are found, single None
        one_away_group_recommendation = None

    return one_away_group_recommendation


def get_manual_recommendation(state: PuzzleState) -> PuzzleState:
    logger.info("Entering get_manual_recommendation")
    logger.debug(f"Entering get_manual_recommendation State: {pp.pformat(state)}")

    state["current_tool"] = "manual_recommender"
    print(f"\nENTERED {state['current_tool'].upper()}")
    print(f"found count: {state['found_count']}, mistake_count: {state['mistake_count']}")

    # display current recommendation and words remaining
    print(f"\nCurrent recommendation: {sorted(state['recommended_words'])}")
    print(f"Words remaining: {state['words_remaining']}")

    # get user input for manual recommendation
    response = "n"
    while response != "y":
        manual_recommendation = [
            x.strip() for x in input("Enter manual recommendation as comma separated words: ").split(",")
        ]
        print(f"Manual recommendation: {manual_recommendation}")

        if not set(manual_recommendation).issubset(set(state["words_remaining"])) or len(manual_recommendation) != 4:
            print("Manual recommendation is not a subset of words remaining or not 4 words")
            print("try again")
        else:
            response = input("Is the manual recommendation correct? (y/n): ")

    # get user defined connection
    response = "n"
    while response != "y":
        manual_connection = input("Enter manual connection: ")
        print(f"Manual connection: {manual_connection}")
        response = input("Is the manual connection correct? (y/n): ")

    state["recommended_words"] = manual_recommendation
    state["recommended_connection"] = manual_connection
    state["tool_status"] = "have_recommendation"

    logger.info("Exiting get_manual_recommendation")
    logger.debug(f"Exiting get_manual_recommendation State: {pp.pformat(state)}")

    return state


async def get_embedvec_recommendation(state: PuzzleState, config: RunnableConfig) -> PuzzleState:
    logger.info("Entering get_embedvec_recommendation")
    logger.debug(f"Entering get_embedvec_recommendation State: {pp.pformat(state)}")

    state["current_tool"] = "embedvec_recommender"
    print(f"\nENTERED {state['current_tool'].upper()}")
    print(f"found count: {state['found_count']}, mistake_count: {state['mistake_count']}")
    print(f"words_remaining: {state['words_remaining']}")

    # get candidate list of words from database
    async with aiosqlite.connect(state["vocabulary_db_fp"]) as conn:
        async with db_lock:
            # get candidate list of words from database
            sql_query = "SELECT word, definition, embedding FROM vocabulary"

            cursor = await conn.execute(sql_query)
            rows = await cursor.fetchall()

            # Convert to a pandas dataframe
            import pandas as pd
            import numpy as np
            import json
            
            # Process embeddings - they might be stored as JSON strings in SQLite
            processed_rows = []
            for row in rows:
                word, definition, embedding = row
                
                # Try to parse the embedding if it's stored as a string
                if isinstance(embedding, str):
                    try:
                        # Try to load as JSON
                        embedding = json.loads(embedding)
                    except:
                        # If that fails, try to parse the string representation of a list
                        try:
                            embedding = eval(embedding)
                        except:
                            logger.warning(f"Could not parse embedding for word {word}")
                            continue
                
                processed_rows.append((word, definition, embedding))
            
            # Create dataframe
            df = pd.DataFrame(processed_rows, columns=["word", "definition", "embedding"])
            
            # Filter to only include words that are still remaining
            df = df[df["word"].isin(state["words_remaining"])]
            
            print(f"Number of words in dataframe: {df.shape[0]}")
            
            # Check if we have enough words to continue
            if df.shape[0] < 4:
                print("Not enough words to continue with embedvec recommendation")
                state["tool_status"] = "no_recommendation"
                return state

            candidate_list = get_candidate_words(df)

    # output the top 10 candidate groups
    print(f"\n>>>Top 10 candidate groups from embedvec recommendation:")
    for i, candidate in enumerate(candidate_list[:10]):
        print(f"{i}: {candidate}")

    if candidate_list:
        # select the top candidate group from the list
        selected_word_group = candidate_list[0]
        print(f"\n>>>Selected word group:\n{selected_word_group}")

        # set the state
        state["recommended_words"] = selected_word_group.get_candidate_words()
        state["recommended_connection"] = "TBD - Using EmbedVec Recommendation"

        # get explanation for the selected group
        anchor_word = selected_word_group.root_word
        print(f"\n>>>anchor word: {anchor_word}")

        anchor_definition = next((pair[1] for pair in selected_word_group.candidate_pairs if pair[0] == anchor_word), "")
        print(f"anchor definition: {anchor_definition}")

        candidate_definitions = [
            (pair[1], pair[0], 0.0) for pair in selected_word_group.candidate_pairs if pair[0] != anchor_word
        ]

        print(f"candidate definitions: {candidate_definitions}")

        response = await config["configurable"]["llm_interface"].choose_embedvec_item(
            anchor_definition, candidate_definitions
        )

        print(f"response: {response}")

        state["recommended_connection"] = response["explanation"]
        state["tool_status"] = "have_recommendation"
    else:
        # if no candidate groups are found, state no recommendation
        state["tool_status"] = "no_recommendation"

    logger.info("Exiting get_embedvec_recommendation")
    logger.debug(f"Exiting get_embedvec_recommendation State: {pp.pformat(state)}")

    return state


async def get_llm_recommendation(state: PuzzleState, config: RunnableConfig) -> PuzzleState:
    logger.info("Entering get_recommendation")
    logger.debug(f"Entering get_recommendation State: {pp.pformat(state)}")

    state["current_tool"] = "llm_recommender"
    print(f"\nENTERED {state['current_tool'].upper()}")
    print(f"found count: {state['found_count']}, mistake_count: {state['mistake_count']}")

    attempt_count = 0
    while True:
        attempt_count += 1
        if attempt_count > RETRY_LIMIT:
            break
        print(f"attempt_count: {attempt_count}")

        # scramble the remaining words for more robust group selection
        if np.random.uniform() < 0.5:
            random.shuffle(state["words_remaining"])
        else:
            state["words_remaining"].reverse()
        print(f"words_remaining: {state['words_remaining']}")
        words_remaining = ", ".join(state["words_remaining"])

        logger.info(f"\nwords remaining for llm: {words_remaining}")

        # get recommendation from llm
        llm_response = await config["configurable"]["llm_interface"].ask_llm_for_solution(words_remaining)

        if isinstance(llm_response, list):
            logger.debug(f"\nLLM response is list")
            recommended_words = llm_response[0]["words"]
            recommended_connection = llm_response[0]["connection"]
        else:
            logger.debug(f"\nLLM response is dict")
            recommended_words = llm_response["words"]
            recommended_connection = llm_response["connection"]

        if compute_group_id(recommended_words) not in set(x[0] for x in state["invalid_connections"]):
            break
        else:
            print(
                f"\nrepeat invalid group detected: group_id {compute_group_id(recommended_words)}, recommendation: {sorted(recommended_words)}"
            )

    state["recommended_words"] = sorted(recommended_words)
    state["recommended_connection"] = recommended_connection

    if attempt_count <= RETRY_LIMIT:
        state["tool_status"] = "have_recommendation"
    else:
        print(f"Failed to get a valid recommendation after {RETRY_LIMIT} attempts")
        print("Changing to manual_recommender, last attempt to solve the puzzle")
        print(f"last recommendation: {state['recommended_words']} with {state['recommended_connection']}")
        state["tool_status"] = "manual_recommendation"

    logger.info("Exiting get_recommendation")
    logger.debug(f"Exiting get_recommendation State: {pp.pformat(state)}")

    return state


async def apply_recommendation(state: PuzzleState, config: RunnableConfig) -> PuzzleState:
    logger.info("Entering apply_recommendation:")
    logger.debug(f"\nEntering apply_recommendation State: {pp.pformat(state)}")

    state["recommendation_count"] += 1

    # get user response from human input
    found_correct_group = state["recommendation_answer_status"]

    # process result of user response
    if found_correct_group in ["y", "g", "b", "p", "correct"]:
        print(f"Recommendation {sorted(state['recommended_words'])} is correct")
        match found_correct_group:
            case "y":
                state["found_yellow"] = True
            case "g":
                state["found_green"] = True
            case "b":
                state["found_blue"] = True
            case "p":
                state["found_purple"] = True
            case "correct":
                pass

        state["recommendation_correct_groups"].append(state["recommended_words"])

        # remove accepted words from vocabulary.db
        async with aiosqlite.connect(state["vocabulary_db_fp"]) as conn:
            async with db_lock:
                # remove accepted words from vocabulary.db
                # for each word in recommended_words, remove the word from the vocabulary table
                for word in state["recommended_words"]:
                    sql_query = f"DELETE FROM vocabulary WHERE word = '{word}'"
                    await conn.execute(sql_query)
                await conn.commit()

        # remove the words from words_remaining
        state["words_remaining"] = [word for word in state["words_remaining"] if word not in state["recommended_words"]]
        state["recommended_correct"] = True
        state["found_count"] += 1
    elif found_correct_group in ["n", "o"]:
        invalid_group = state["recommended_words"]
        invalid_group_id = compute_group_id(invalid_group)
        state["invalid_connections"].append((invalid_group_id, invalid_group))
        state["recommended_correct"] = False
        state["mistake_count"] += 1

        if state["mistake_count"] < MAX_ERRORS:
            match found_correct_group:
                case "o":
                    print(f"Recommendation {sorted(state['recommended_words'])} is incorrect, one away from correct")

                    # perform one-away analysis
                    one_away_group_recommendation = await one_away_analyzer(
                        state, invalid_group, state["words_remaining"], config
                    )

                    # check if one_away_group_recommendation is a prior mistake
                    if one_away_group_recommendation:
                        one_away_group_id = compute_group_id(one_away_group_recommendation.words)
                        if one_away_group_id in set(x[0] for x in state["invalid_connections"]):
                            print(f"one_away_group_recommendation is a prior mistake")
                            one_away_group_recommendation = None
                        else:
                            print(f"one_away_group_recommendation is a new recommendation")

                case "n":
                    print(f"Recommendation {sorted(state['recommended_words'])} is incorrect")
                    if state["current_tool"] == "embedvec_recommender":
                        print("Changing the recommender from 'embedvec_recommender' to 'llm_recommender'")
                        state["current_tool"] = "llm_recommender"
        else:
            state["recommended_words"] = []
            state["recommended_connection"] = ""
            state["recommended_correct"] = False

    if len(state["words_remaining"]) == 0 or state["mistake_count"] >= MAX_ERRORS:
        if state["mistake_count"] >= MAX_ERRORS:
            logger.info("FAILED TO SOLVE THE CONNECTION PUZZLE TOO MANY MISTAKES!!!")
            print("FAILED TO SOLVE THE CONNECTION PUZZLE TOO MANY MISTAKES!!!")
        else:
            logger.info("SOLVED THE CONNECTION PUZZLE!!!")
            print("SOLVED THE CONNECTION PUZZLE!!!")

        state["tool_status"] = "puzzle_completed"
    elif found_correct_group == "o":
        if one_away_group_recommendation:
            print(f"using one_away_group_recommendation")
            state["recommended_words"] = one_away_group_recommendation.words
            state["recommended_connection"] = one_away_group_recommendation.connection_description
            state["tool_status"] = "have_recommendation"
        else:
            print(f"no one_away_group_recommendation, let llm_recommender try again")
            state["recommended_words"] = []
            state["recommended_connection"] = ""
            state["tool_status"] = "next_recommendation"
    elif found_correct_group == "m":
        print("Changing to manual_recommender")
        state["tool_status"] = "manual_recommendation"

    elif found_correct_group == "s":
        print("switching recommender")
        state["tool_status"] = "switch_recommender"

    else:
        logger.info("Going to next get_recommendation")
        state["tool_status"] = "next_recommendation"

    logger.info("Exiting apply_recommendation:")
    logger.debug(f"\nExiting apply_recommendation State: {pp.pformat(state)}")

    return state