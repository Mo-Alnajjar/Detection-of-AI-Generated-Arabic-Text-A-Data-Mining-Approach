import numpy as np
import re
import pandas as pd
from collections import Counter


# ------------------ Task 2.2 ----------------------
# Function calculate avg word length
def avg_word_length(text):
    """
    Calculate the average word length in a given text.

    This function splits the text into words (separated by spaces) 
    and computes the average number of characters per word.

    Args:
        text (str): The input text to analyze.

    Returns:
        float: The average word length. Returns 0 if the text contains no words.
    """
    words = text.split()
    if len(words) == 0:
        return 0
    total_chars = sum(len(w) for w in words)
    return total_chars / len(words)

# Function calculate avg sentence length
def avg_sentence_length(text):
    """
    Calculate the average sentence length in words for a given text.

    This function splits the text into sentences based on punctuation marks 
    (., !, ؟, or !) and computes the average number of words per sentence.

    Args:
        text (str): The input text to analyze.

    Returns:
        float: The average sentence length (in words). Returns 0 if no sentences are found.
    """
    sentences = re.split(r"[.!؟!]", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) == 0:
        return 0
    total_words = sum(len(s.split()) for s in sentences)
    return total_words / len(sentences)

# Function calculate type token ratio (vocabulary richness)
def type_token_ratio(text):
    """
    Calculate the Type-Token Ratio (TTR) of a given text.

    The TTR measures vocabulary richness by dividing the number of unique words 
    by the total number of words in the text.

    Args:
        text (str): The input text to analyze.

    Returns:
        float: The type-token ratio (unique words / total words). Returns 0 if no words are found.
    """
    words = text.split()
    if len(words) == 0:
        return 0
    unique_words = set(words)
    return len(unique_words) / len(words)


def lexical_analysis(df, text_col, label_col):
    """
    Perform lexical analysis comparing function words, punctuation, 
    and specific term usage between human and AI-generated texts.
    
    Args:
        df (pd.DataFrame): The dataset containing text and labels.
        text_col (str): Column name containing normalized text.
        label_col (str): Column name indicating 'human' or 'ai_generated'.
        
    Returns:
        dict: Comparison of word and punctuation frequencies.
    """

    # Define Arabic function words (you can expand this list)
    function_words = ["في", "من", "على", "أن", "عن", "ما", "قد", "لا", "هو", "هي", "هذا", "ذلك", "كان"]

    # Separate human and AI texts
    human_texts = " ".join(df[df[label_col] == "human"][text_col])
    ai_texts = " ".join(df[df[label_col] == "ai_generated"][text_col])

    # --- Count function words ---
    def count_words(text, word_list):
        return sum(text.count(w) for w in word_list)

    human_func_count = count_words(human_texts, function_words)
    ai_func_count = count_words(ai_texts, function_words)

    # --- Count punctuation ---
    def count_punctuation(text):
        return len(re.findall(r"[.,!?؟؛]", text))

    human_punct_count = count_punctuation(human_texts)
    ai_punct_count = count_punctuation(ai_texts)

    # --- Most common specific terms (optional extension) ---
    def get_top_terms(text, n=10):
        words = re.findall(r"[\u0600-\u06FF]+", text)
        return Counter(words).most_common(n)

    result = {
        "function_words": {
            "human": human_func_count,
            "ai_generated": ai_func_count
        },
        "punctuation": {
            "human": human_punct_count,
            "ai_generated": ai_punct_count
        },
        "top_terms": {
            "human": get_top_terms(human_texts),
            "ai_generated": get_top_terms(ai_texts)
        }
    }

    return result
