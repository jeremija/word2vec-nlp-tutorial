from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
# import numpy as np
import pandas as pd
import re

english_stopwords = set(stopwords.words("english"))
vectorizer = CountVectorizer(analyzer="word",
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_features=5000)


def load_data(csv_path):
    return pd.read_csv(csv_path, header=0, delimiter="\t", quoting=3)


def clean_strings(review):
    review_text = BeautifulSoup(review, "html.parser").get_text()
    words_only = re.sub("[^a-zA-Z]", " ", review_text).lower().split()
    return " ".join([w for w in words_only if w not in english_stopwords])


def clean_data(data):
    return [clean_strings(item) for item in data]


def create_train_features(data):
    print('preparing train data...')
    clean_train_data = clean_data(data)
    train_data_features = vectorizer.fit_transform(clean_train_data)
    return train_data_features.toarray()


def create_test_features(data):
    print('preparing training data')
    clean_test_data = clean_data(data)
    test_data_features = vectorizer.transform(clean_test_data)
    return test_data_features.toarray()


def main():
    data = load_data("./data/labeledTrainData.tsv")
    split = 20000

    train = {
        "review": data["review"][:split],
        "sentiment": data["sentiment"][:split]
    }
    train_data_features = create_train_features(train["review"])
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_data_features, train["sentiment"])

    test = {
        "review": data["review"][split:],
        "sentiment": data["sentiment"][split:]
    }
    test_data_features = create_test_features(test["review"])

    print('predicting')
    result = forest.predict(test_data_features)

    errors = 0
    for i, item in enumerate(test["sentiment"]):
        res = result[i]
        if item != res:
            errors += 1

    print('errors:', errors, 'of', result.size)
    print('error rate:', round(errors / result.size, 3))


if __name__ == "__main__":
    main()
