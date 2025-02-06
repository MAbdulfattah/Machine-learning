# CLI
import click

# Data
import pandas
import re
import spacy
from spacy.tokens import Doc, Token
from spacy.lexeme import Lexeme
from urllib.parse import unquote
from nltk.stem.snowball import SnowballStemmer

# Other
from collections import Counter
from os.path import splitext
from typing import Tuple

# Analysation
from wordcloud import WordCloud

nlp = spacy.load("en_core_web_sm")
stemmer = SnowballStemmer(language="english")

# Add some extra stoppy words
nlp.Defaults.stop_words.add("amp")
nlp.Defaults.stop_words.add("like")
nlp.Defaults.stop_words.add("rt")

pandas.set_option("display.max_colwidth", None)


def clean_text(text: str):
    # To lower-case
    text = text.lower()

    # NLP with Spacy
    tokens: list[Token] = nlp(text)

    filtered_str: list[str] = []
    hashtag_list: list[str] = []

    for (i, token) in enumerate(tokens):
        lex: Lexeme = token.lex

        if i - 1 >= 0:
            prev = tokens[i - 1]
            if prev.text == "#":
                hashtag_list.append(token.lemma_)

        # Check if token is not punct or space or non-unicode
        if (
            lex.is_alpha
            and len(token.lemma_) > 1
            and not lex.is_stop
            and not lex.is_punct
            and not lex.like_url
            and not lex.like_num
        ):
            filtered_str.append(token.lemma_)

    text = " ".join(filtered_str)

    if len(hashtag_list) == 0:
        return pandas.Series([text, pandas.NA])

    else:
        hashtags = " ".join(hashtag_list)
        return pandas.Series([text, hashtags])


def clean_keyword(keyword: str) -> str:
    # URL-decode
    keyword = unquote(keyword)

    # To lower-case
    keyword = keyword.lower()

    # NLP with Spacy
    tokens: list[Token] = nlp(keyword)

    keywords = []
    for token in tokens:
        lex: Lexeme = token.lex

        if not lex.is_punct:
            lemma = token.lemma_
            stem = stemmer.stem(lemma)

            keywords.append(stem)

    return " ".join(keywords)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("text")
def test(text: str):
    """Allows you to test the text cleaning without a dataset."""

    new_text = clean_text(text)
    print(f"Text without processing:\n{text}\n\nText with processing:\n{new_text}")


def clean_dataset(data: pandas.DataFrame, limited: bool) -> pandas.DataFrame:
    # Number of instances before cleaning
    print(f"The dataset consists of {data.shape[0]} instances before cleaning.")

    # Drop the location column as we currently have no use for it.
    data = data.drop("location", axis=1)

    # Drop all rows with missing values
    data = data.dropna()

    # Select the first 50 items during testing
    if limited:
        data = data[:50]

    # Re-map the text column with cleaned text
    data[["text", "hashtags"]] = data["text"].apply(clean_text)

    # Re-map the keyword column with the cleaned keyword.
    data["keyword"] = data["keyword"].apply(clean_keyword)

    # Drop the `id` column
    data = data.drop("id", axis=1)
    data = data.reset_index(drop=True)

    # Drop duplicates
    data = data.drop_duplicates(subset=["text"])
    data = data.dropna(subset=["text"])

    if limited:
        print(f"Cleaned items:\n{data}\n")

    data.info()

    # Number of instances after cleaning
    print(f"The dataset consists of {data.shape[0]} instances after cleaning.")

    # Number of unique keywords
    print(f"Number of unique keywords: {data['keyword'].nunique()}")

    return data


@cli.command()
@click.argument("file_name", type=click.Path(exists=True))
@click.option("--limited/--full", default=False, help="Limited output")
def clean(file_name: str, limited: bool):
    """Transforms an entire dataset into clean data."""

    # Open the dataset
    data = pandas.read_csv(file_name)

    # Perform the cleaning
    data = clean_dataset(data, limited)

    # Export data to CSV
    file_name, _ = splitext(file_name)
    file_name = f"{file_name}_{'limited' if limited else 'full'}_clean.csv"
    data.to_csv(file_name, index=False)


@cli.command()
@click.argument("file_name", type=click.Path(exists=True))
def analyse(file_name: str):
    """Analyse the full, cleaned data set"""

    # Open the dataset
    data = pandas.read_csv(file_name)
    data["target"] = data["target"].astype(bool)

    for target in [True, False]:
        # Filter by the current target
        filtered = data[data["target"] == target]

        all_words = []

        for _, row in filtered.iterrows():
            text = row["text"]

            if pandas.notnull(text):
                all_words.extend(text.split(" "))

        counter = Counter(all_words)

        # Generate the word cloud
        wc = WordCloud(width=2000, height=1000, background_color="white")

        wc.generate_from_frequencies(counter)
        wc.to_file(f"train_cloud_{'disaster' if target else 'normal'}.png")


if __name__ == "__main__":
    cli()