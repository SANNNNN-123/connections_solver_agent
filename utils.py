# TODO: functions here may be deprecated, consider removing them

from typing import List


# function to chunk words into groups of 4
def chunk_words(words: List[str]) -> List[List[str]]:
    result = []
    for i in range(0, len(words), 4):
        result.append(words[i : i + 4])
    return result


# function to flatten list of lists to a single list
def flatten_list(list_of_lists: List[List[str]]) -> List[str]:
    return [item for sublist in list_of_lists for item in sublist]
