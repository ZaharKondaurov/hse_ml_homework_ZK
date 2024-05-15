import click
import pickle

import numpy as np
import pandas as pd
import re
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline

stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    return re.sub(r"[^\w\s]+", '', text).lower().split()


def preprocess_sentence_lem(text: str) -> str:
    prep_text = preprocess_text(text)
    lemmatizer = WordNetLemmatizer()
    singles = [lemmatizer.lemmatize(x) for x in prep_text if x not in stop_words]
    return ' '.join(singles)


@click.group()
def cli1():
    pass


@cli1.command('train')
@click.option('--data')
@click.option('--test', is_flag=False, flag_value=None, default=None)
@click.option('--split', 'test_size', is_flag=False, flag_value=None, default=None, type=float)
@click.option('--seed', is_flag=False, flag_value=None, default=None, type=int)
@click.option('--model')
@click.option('--t', is_flag=False, flag_value=True, default=False)
def train(data, test, test_size, seed, model, t):
    if seed is None:
        seed = 42
    pipe = Pipeline([('bow', CountVectorizer()), ('logreg', LogisticRegression(max_iter=500))])
    if t:
        pipe = Pipeline([('bow', CountVectorizer()), ('logreg', LogisticRegression(max_iter=300))])
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    test_df = None

    assert test_size is None or test is None, "Error: choose either split or test"
    assert os.path.getsize(data) != 0, "Error: data is empty"

    if data.endswith('.csv'):
        train_df = pd.read_csv(data)
    elif data.endswith('.xls') or data.endswith('.xlsx'):
        train_df = pd.read_excel(data)
    else:
        assert False, "Error: Incorrect file type"

    assert 'text' in train_df.columns, "Error: missing 'text' column in train data"
    assert 'rating' in train_df.columns, "Error: missing 'rating' column in train data"

    train_df['rating_num'] = train_df['rating'].map(lambda x: 1 if x >= 4 else 0)
    train_df['text_lem'] = train_df['text'].map(lambda x: preprocess_sentence_lem(x))

    if test_size is not None:
        assert 0 <= float(test_size) <= 1, "Error: split should be between 0 and 1"
        train_df, test_df = train_test_split(train_df, test_size=float(test_size), shuffle=True, random_state=seed)
        if t:
            train_df.loc[:, ~np.isin(train_df.columns, ['rating_num', 'text_lem'])].to_csv('train_debug.csv', index=True, index_label='IND')
            test_df.loc[:, ~np.isin(test_df.columns, ['rating_num', 'text_lem'])].to_csv('test_debug.csv', index=True, index_label='IND')

    if x_train is None:
        x_train = train_df['text_lem']
        y_train = train_df['rating_num']
    else:
        x_train = pd.concat([x_train, train_df['text_lem']], axis=0)
        y_train = pd.concat([y_train, train_df['rating_num']], axis=0)

    if x_test is None and test_df is not None:
        x_test = test_df['text_lem']
        y_test = test_df['rating_num']
    elif x_test is not None and test_df is not None:
        x_test = pd.concat([x_test, test_df['text_lem']], axis=0)
        y_test = pd.concat([y_test, test_df['rating_num']], axis=0)

    if test is not None:
        assert os.path.getsize(test) != 0, "Error: data is empty"
        test_df = pd.read_csv(test)

        assert 'text' in test_df.columns, "Error: missing 'text' column in test data"
        assert 'rating' in test_df.columns, "Error: missing 'rating' column in test data"

        test_df['rating_num'] = test_df['rating'].map(lambda x: 1 if x >= 4 else 0)
        test_df['text_lem'] = test_df['text'].map(lambda x: preprocess_sentence_lem(x))

        if x_test is None:
            x_test = test_df['text_lem']
            y_test = test_df['rating_num']
        else:
            x_test = pd.concat([x_test, test_df['text_lem']], axis=0)
            y_test = pd.concat([y_test, test_df['rating_num']], axis=0)

    pipe.fit(x_train, y_train)

    y_pred = pipe.predict(x_train)
    print(f"train: {f1_score(y_train, y_pred, average='macro')}")
    y_pred = pipe.predict(x_test)
    print(f"test: {f1_score(y_test, y_pred, average='macro')}")

    pickle.dump(pipe, open(model, 'wb'))


@cli1.command('predict')
@click.option('--model')
@click.option('--data')
def predict(model, data):
    if not model.endswith('.pkl'):
        assert False, "Error: Not a pickle file"
    try:
        pickled_model = pickle.load(open(model, 'rb'))
        assert isinstance(pickled_model, Pipeline)
    except pickle.UnpicklingError:
        assert "Error: Invalid model"

    pickled_model = pickle.load(open(model, 'rb'))

    if data.endswith('.csv') and ' ' not in data:
        assert os.path.getsize(data) != 0, "Error: data is empty"
        df = pd.read_csv(data)
    elif (data.endswith('.xlsx') or data.endswith('.xls')) and ' ' not in data:
        assert os.path.getsize(data) != 0, "Error: data is empty"
        df = pd.read_excel(data)
    else:
        df = pd.DataFrame([data], columns=['text'])

    assert 'text' in df.columns, "Error: missing 'text' column in data"

    df['text_lem'] = df['text'].map(lambda x: preprocess_sentence_lem(x))
    y_pred = pickled_model.predict(df['text_lem'])
    for index in range(len(y_pred)):
        print(f'{"positive review" if y_pred[index] else "negative review"} {y_pred[index]}')


if __name__ == '__main__':
    cli1()
