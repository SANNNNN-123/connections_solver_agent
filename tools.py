import base64
import hashlib
import json
import logging
import pprint as pp
from typing import List, TypedDict
from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig


logger = logging.getLogger(__name__)

pp = pp.PrettyPrinter(indent=4)


class LLMInterfaceBase(ABC):
    """base class for LLM Interface"""

    @abstractmethod
    def __init__(self, model_name: str, **kwargs):
        """setups up LLM Model"""
        raise NotImplementedError()

    @abstractmethod
    async def generate_vocabulary(self, words):
        """creates definitions for all the words"""
        raise NotImplementedError()

    @abstractmethod
    def generate_embeddings(self, definitions: List[str]) -> List[List[float]]:
        """generates embeddings for the definitions"""
        raise NotImplementedError()

    @abstractmethod
    async def choose_embedvec_item(self, anchor_definition: str, candidate_definitions: list):
        """chooses an item from a list of candidates"""
        raise NotImplementedError()

    @abstractmethod
    async def ask_llm_for_solution(self, prompt):
        """asks the LLM for a solution to a prompt"""
        raise NotImplementedError()

    @abstractmethod
    async def extract_words_from_image(encoded_image: str) -> List[str]:
        """extracts words from an image"""
        raise NotImplementedError()

    @abstractmethod
    async def analyze_anchor_words_group(self, anchor_words_group):
        """analyzes a group of anchor words"""
        raise NotImplementedError()

    @abstractmethod
    async def generate_one_away_recommendation(self, anchor_words_prompt: str):
        """generates a recommendation for a single word that is one away from the correct group"""
        raise NotImplementedError()

    @abstractmethod
    async def ask_llm_for_next_step(self, instructions: str, puzzle_state: str):
        """asks the LLM for the next step in the workflow"""
        raise NotImplementedError()


class LLMRegistry:
    """
    A singleton class that maintains a registry of classes.

    Attributes:
        _instance (Registry): The singleton instance of the Registry class.
        _registry (dict): A dictionary that stores registered classes with their names as keys.

    Methods:
        register(name):
            A class method that registers a class with a given name.
            Args:
                name (str): The name to register the class under.
            Returns:
                function: A decorator function that registers the class.

        get(name):
            A class method that retrieves a registered class by its name.
            Args:
                name (str): The name of the registered class to retrieve.
            Returns:
                class: The registered class, or None if not found.

        list_registered():
            A class method that lists all registered class names.
            Returns:
                list: A list of all registered class names.
    """

    _instance = None
    _registry = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMRegistry, cls).__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, name):
        def decorator(class_):
            cls._registry[name] = class_
            return class_

        return decorator

    @classmethod
    def get(cls, name):
        return cls._registry.get(name)

    @classmethod
    def list_registered(cls):
        return list(cls._registry.keys())


# singleton instance of the registry for LLMs
llm_interface_registry = LLMRegistry()


def compute_group_id(word_group: list) -> str:
    """
    Computes a unique group ID for a list of words.

    This function takes a list of words, sorts them, concatenates them into a single string,
    and then generates an MD5 hash of that string. The resulting hash is returned as a
    hexadecimal string.

    Args:
        word_group (list): A list of words (strings) to compute the group ID for.

    Returns:
        str: A hexadecimal string representing the MD5 hash of the concatenated and sorted words.
    """
    return hashlib.md5("".join(sorted(word_group)).encode()).hexdigest()


def read_file_to_word_list(file_location: str) -> List[str]:
    """
    Reads a file and returns a list of words separated by commas on the first line of the file.
    Remainder of the file contains the correct word groupings and is ignored.

    Prompts the user to enter the file location, reads the file, and splits its contents by commas.
    Strips any leading or trailing whitespace from each word.

    Returns:
    list: A list of words from the file. If the file is not found or an error occurs, returns an empty list.
    """

    logger.info(f"Reading words from file {file_location}")
    try:
        with open(file_location, "r") as file:
            contents = file.readline()
            words = contents.split(",")
            words = [word.strip() for word in words]
            return words
    except FileNotFoundError:
        print(f"File not found: {file_location}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


async def extract_words_from_image_file(image_fp: str, config: RunnableConfig) -> List[str]:
    """
    Encodes an image to base64 and sends it to the OpenAI LLM to extract words from the image.

    Parameters:
    image_path (str): The path to the image file to be processed.

    Returns:
    dict: The response from the LLM in JSON format.
    """

    logger.info("Entering extract_words_from_image")
    logger.debug(f"Entering extract_words_from_image image_path: {image_fp}")

    # Encode the image
    with open(image_fp, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Get the response from the model
    # response = await chat_with_llm([message])
    response = await config["configurable"]["llm_interface"].extract_words_from_image(base64_image)

    words = [w.lower() for w in response["words"]]

    logger.info("Exiting extract_words_from_image")
    logger.debug(f"Exiting extract_words_from_image response {words}")

    return words


async def manual_puzzle_setup_prompt(config: RunnableConfig) -> List[str]:
    """
    Asynchronously prompts the user to set up a puzzle by specifying the source of the puzzle words.

    Args:
        config (RunnableConfig): Configuration object required for extracting words from an image.

    Returns:
        List[str]: A list of words extracted from the specified source.

    Raises:
        ValueError: If the input source type is neither 'file' nor 'image'.
    """

    # pompt user for puzzle source
    puzzle_source_type = input("Enter 'file' to read words from a file or 'image' to read words from an image: ")
    puzzle_source_fp = input("Please enter the file/image location: ")

    # get puzzle words from indicated source
    if puzzle_source_type == "file":
        words = read_file_to_word_list(puzzle_source_fp)
    elif puzzle_source_type == "image":
        words = await extract_words_from_image_file(puzzle_source_fp, config=config)
    else:
        raise ValueError("Invalid input source. Please enter 'file' or 'image'.")

    return words


def interact_with_user(gen_words, gen_reason, recommender) -> str:
    """
    Interacts with the user by providing a recommendation message and capturing the user's response.

    Args:
        gen_words (str): The generated words to be recommended.
        gen_reason (str): The reason for the recommendation.
        recommender (str): The name of the recommender.

    Returns:
        str: The user's response to the recommendation.

    Logs:
        Logs the recommendation message, the user instruction prompt, and the user's response.
    """
    recommendation_message = f"\n{recommender.upper()}: RECOMMENDED WORDS {gen_words} with connection {gen_reason}"
    logger.info(recommendation_message)
    print(recommendation_message)

    user_instruction = "Is the recommendation accepted? (y/g/b/p/m/s/o/n): "
    logger.info(user_instruction)
    user_response = input(user_instruction)

    logger.info(f"User response: {user_response}")

    return user_response


def check_one_solution(solution, *, gen_words: List[str], gen_reason: str, recommender: str) -> str:
    """
    Checks if the generated solution matches any of the provided solutions.
    This is used by the automated tester.

    Args:
        solution (dict): A dictionary containing groups of words and their reasons.
        gen_words (List[str]): A list of generated words to be checked.
        gen_reason (str): The reason associated with the generated words.
        recommender (str): The name of the recommender.

    Returns:
        str: "correct" if the generated words match exactly with any group of words in the solution,
             "o" if there are three words in common between the generated words and any group of words in the solution,
             "n" otherwise.
    """
    recommendation_message = f"\n{recommender.upper()}: RECOMMENDED WORDS {gen_words} with connection {gen_reason}"
    logger.info(recommendation_message)
    print(recommendation_message)

    for sol_dict in solution["groups"]:
        sol_words = sol_dict["words"]
        sol_reason = sol_dict["reason"]
        if set(gen_words) == set(sol_words):
            print(f"{gen_reason} ~ {sol_reason}: {gen_words} == {sol_words}")
            return "correct"
        elif len(set(gen_words).intersection(set(sol_words))) == 3:
            return "o"
    else:
        return "n"
