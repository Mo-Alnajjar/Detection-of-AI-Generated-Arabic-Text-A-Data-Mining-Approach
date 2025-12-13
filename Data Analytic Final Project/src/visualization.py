# Use the arabic_reshaper and python-bidi libraries to reshape and reorder the Arabic text for proper rendering
import arabic_reshaper 
from bidi.algorithm import get_display
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import pandas as pd
import numpy as np

font_file = "../src/font/ARIAL.TTF"

def reshape_arabic_text(text):
    """
    Reshape and reorder Arabic text for correct display in visualizations.

    This function uses the `arabic_reshaper` library to connect isolated Arabic 
    characters into their proper contextual forms, and the `python-bidi` library 
    to reorder the text for right-to-left rendering. It is especially useful for 
    preparing Arabic text for visualization tools (e.g., WordCloud, Matplotlib) 
    that do not natively support Arabic shaping.

    Parameters
    ----------
    text : str
        The Arabic text string to be reshaped and reordered.

    Returns
    -------
    str
        The reshaped and bidi-corrected Arabic text, ready for proper display.
    """
    reshaped_text = arabic_reshaper.reshape(text)       # reshape letters
    bidi_text = get_display(reshaped_text)              # correct right-to-left display
    return bidi_text


def plot_top_ngrams(texts, n=20, ngram_range=(1, 1), title="Top n-grams"):
    """
    Plot the most frequent n-grams from Arabic text data with proper letter shaping.

    Parameters
    ----------
    texts : list or pandas.Series
        A collection of Arabic text samples.
    n : int, optional
        The number of top n-grams to display (default is 20).
    ngram_range : tuple, optional
        The n-gram range for CountVectorizer (default is (1, 1)).
    title : str, optional
        The title for the plot.

    Returns
    -------
    None
        Displays a bar chart of the top n-grams.
    """
    
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(texts)
    sum_words = X.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]

    df_freq = pd.DataFrame(words_freq, columns=["ngram", "count"])
    df_freq["ngram"] = df_freq["ngram"].apply(lambda x: get_display(arabic_reshaper.reshape(x)))
    plt.figure(figsize=(10, 6))
    sns.barplot(x="count", y="ngram", data=df_freq, palette="viridis")
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("")
    plt.show()


def generate_wordclouds(df, text_col, label_col, font_path=font_file):
    """
    Generate and display word clouds for each class (human vs. AI-generated) in an Arabic text dataset.

    This function combines text from each class, reshapes Arabic characters for correct display,
    and visualizes them as side-by-side word clouds.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing text and labels.
    text_col : str
        The name of the column containing the normalized Arabic text.
    label_col : str
        The name of the column containing the class labels (e.g., 'human vs. AI_generated').
    font_path : str
        The file path to a font that supports Arabic (default is "../ARIAL.TTF").

    Returns
    -------
    None
        Displays the generated word clouds for both classes.
    """

    # Helper for Arabic reshaping
    def reshape_arabic_text(text):
        reshaped_text = arabic_reshaper.reshape(text)
        return get_display(reshaped_text)

    # Combine all text for each class
    human_text = reshape_arabic_text(" ".join(df[df[label_col].str.lower() == "human"][text_col]))
    ai_text = reshape_arabic_text(" ".join(df[df[label_col].str.lower() == "ai_generated"][text_col]))

    # Generate word clouds
    wc_human = WordCloud(
        font_path=font_path,
        width=800,
        height=400,
        background_color="white",
        colormap="Blues"
    ).generate(human_text)

    wc_ai = WordCloud(
        font_path=font_path,
        width=800,
        height=400,
        background_color="white",
        colormap="Reds"
    ).generate(ai_text)

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(wc_human, interpolation="bilinear")
    axes[0].set_title("النصوص البشرية", fontsize=16)
    axes[0].axis("off")

    axes[1].imshow(wc_ai, interpolation="bilinear")
    axes[1].set_title("النصوص المولدة بالذكاء الاصطناعي", fontsize=16)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


## Task 5.2: For the best-performing traditional model, extract and visualize feature importances/coefficients to understand what features are most indicative of AI-generated text.
def plot_feature_importances(model, feature_names):
    """
    Plot feature importances for a Random Forest model.
    Automatically adjusts to the number of features provided.

    Parameters
    ----------
    model : RandomForestClassifier
        Trained Random Forest model.
    feature_names : list
        Names of your features.
    """

    importances = model.feature_importances_

    # Sort by importance (descending)
    sorted_idx = np.argsort(importances)[::-1]

    # Select only existing features
    sorted_features = np.array(feature_names)[sorted_idx]
    sorted_importances = importances[sorted_idx]

    # Print
    print("\nFeature Importances:")
    for name, score in zip(sorted_features, sorted_importances):
        print(f"{name}: {score:.4f}")

    # Plot
    plt.figure(figsize=(7, 4))
    plt.barh(sorted_features[::-1], sorted_importances[::-1])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    plt.show()