import re
import pandas as pd
import numpy as np
import stanza

# The following functions are used for Data Preprocessing & Exploratory Data Analysis (EDA):
# ------------------ Task 2.1 ----------------------
# Function used to normalize Arabic words for next stages
def normalize_arabic(text):
    """
    Normalize and clean Arabic text for preprocessing tasks.

    This function performs several normalization and cleaning steps to ensure 
    consistency in Arabic text before analysis or modeling. Specifically, it:

    1. Normalizes all forms of Alif (إ، أ، آ، ا) to a simple "ا".
    2. Converts Alif Maqsura (ى) to (ي).
    3. Removes Tatweel (ـ) characters.
    4. Removes Arabic diacritics (e.g., َ ِ ُ ً ٌ ٍ ّ ْ).
    5. Removes any non-Arabic characters except spaces and Arabic numerals (٠–٩).
    6. Collapses multiple spaces into a single space.
    7. Removes the Arabic question mark (؟).

    Args:
        text (str): The Arabic text to normalize.

    Returns:
        str: The cleaned and normalized Arabic text.
    """

    # Normalize Alif forms
    text = re.sub("[إأآا]", "ا", text)
    
    # Normalize Alif Maqsura to Ya
    text = re.sub("ى", "ي", text)
    
    # Remove Tatweel
    text = re.sub("ـ", "", text)
    
    # Remove diacritics
    text = re.sub("[ًٌٍَُِّْ]", "", text)
    
    # Remove non-Arabic characters
    text = re.sub("[^\u0600-\u06FF\s\u0660-\u0669]", " ", text)
    
    # Collapse extra spaces
    text = re.sub("\s+", " ", text).strip()
    
    # Remove Arabic question mark from the text 
    text = re.sub("؟", "", text).strip()

    return text

# ---------------------------- Phase 3: Feature Engineering ---------------------------------
nlp = stanza.Pipeline(lang='ar', processors='tokenize,pos', use_gpu=False, verbose=False)

def honore_r_measure(text):
    """
    Calculate Honoré's R measure for a given text.
    Measures vocabulary sophistication based on word frequency.
    
    Formula:
        R = 100 * log(N) / (1 - (V1 / V))
        
    Where:
        N  = total number of words
        V  = number of unique words
        V1 = number of words appearing only once
        
    Returns:
        float: Honoré's R value (0 if text is empty or invalid).
    """
    words = text.split()
    N = len(words)
    if N == 0:
        return 0.0
    
    unique_counts = {w: words.count(w) for w in set(words)}
    V = len(unique_counts)
    V1 = sum(1 for c in unique_counts.values() if c == 1)

    try:
        R = 100 * np.log(N) / (1 - (V1 / V))
        return round(R, 2)
    except ZeroDivisionError:
        return 0.0
    
def count_arabic_nouns(text):
    """
    Count the number of nouns in Arabic text using Stanza POS tagging.

    A token is counted as a noun if its POS tag is one of:
      - NOUN: common noun
      - PROPN: proper noun

    Args:
        text (str): Arabic text.

    Returns:
        int: Number of nouns in the text.
    """

    if not isinstance(text, str) or text.strip() == "":
        return 0

    doc = nlp(text)
    noun_count = 0

    for sentence in doc.sentences:
        for word in sentence.words:
            if word.upos in ["NOUN", "PROPN"]:
                noun_count += 1

    return noun_count

def count_genitives(text, nlp_pipeline=nlp):
    """
    Estimate genitive (idafa) relations by finding sequences of two consecutive nouns.
    """
    if not isinstance(text, str) or len(text.strip()) < 3:
        return 0

    doc = nlp_pipeline(text)
    count = 0
    for sent in doc.sentences:
        words = sent.words
        for i in range(len(words) - 1):
            if words[i].upos == "NOUN" and words[i + 1].upos == "NOUN":
                count += 1
    return count

nlp_density = stanza.Pipeline(lang='ar', processors='tokenize,pos,ner', use_gpu=False, verbose=False)

def entity_density(text, nlp_pipeline=nlp_density):
    """
    Calculate the ratio of named entities to total words (Entity Density) in Arabic text.

    Parameters:
        text (str): The Arabic text to analyze.
        nlp_pipeline (stanza.Pipeline): Preloaded Stanza Arabic pipeline.

    Returns:
        float: Entity density (0.0–1.0 range typically).
    """
    if not isinstance(text, str) or len(text.strip()) < 3:
        return 0.0

    doc = nlp_pipeline(text)
    num_entities = len(doc.ents)
    total_words = sum(len(s.words) for s in doc.sentences)

    if total_words == 0:
        return 0.0

    return num_entities / total_words