# Data
import pandas
import re
import spacy
from spacy.tokens import Doc, Token
#spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

nlp = spacy.load("en_core_web_sm")

# Add some extra stoppy words
nlp.Defaults.stop_words.add("amp")
nlp.Defaults.stop_words.add("like")
nlp.Defaults.stop_words.add("rt")

# pandas.set_option("display.max_colwidth", None)


def clean_text(text: str) -> str:
    # To lower-case
    text = text.lower()

    # Remove full URLs
    text = re.sub("http\S+", "", text)

    # NLP with Spacy
    tokens: list[Token] = nlp(text)

    filtered_str: list[str] = []
    #all_stopwords = spacy_stopwords + ['like', 'm', 's', 'amp']
    for token in tokens:
        # Check if token is not punct or space or non-unicode
        if (
            not token.is_space
            and token.is_alpha
            and not token.is_stop
            #and not token.text in all_stopwords
            and not token.is_punct
        ):
            filtered_str.append(token.lemma_)

    text = " ".join(filtered_str)

    return text

def clean(data, file_name: str, limited: bool):
    """Transforms an entire dataset into clean data."""
    # Open the dataset
#     data = pandas.read_csv(file_name)

    # Drop the location column as we currently have no use for it.
    # TODO: Figure out how to clean/use location?
#     data = data.drop("location", axis=1)

    # Drop all rows with missing values
#     data = data.dropna()

    # Select the first 50 items during testing
    if limited:
        data = data[:50]

    # Re-map the text column with cleaned text
    data["text"] = data["text"].map(clean_text)

    # Drop the `id` column
    data = data.drop("id", axis=1)
    data = data.reset_index(drop=True)

    # Drop duplicates
    data = data.drop_duplicates(subset=["text"])

    if limited:
        print(f"Cleaned items:\n{data}\n")

    print(data.info())

    # Export data to CSV
#     file_name, _ = splitext(file_name)
    file_name = f"{file_name}_clean.csv"
    data.to_csv(file_name, index=False)
