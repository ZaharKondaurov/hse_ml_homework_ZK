import pytest

import os
import re
import subprocess
import pickle
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def extract_assert(err):
    return re.search(r'AssertionError: \s*(.*)', err).group(1).strip()

def test_train():
    out = subprocess.run(['python', 'main.py', 'train', '--data', '../../data/train.csv', '--test', '../../data/test.csv', '--model', 'models/model.pkl', '--t'], stdout=subprocess.PIPE)
    assert os.path.exists('models/model.pkl')
    assert out.returncode == 0
    try:
        model = pickle.load(open('models/model.pkl', 'rb'))
        assert isinstance(model, Pipeline)  # Check that this is a correct model
    except:
        assert False

    # Check assert in train command
    res = subprocess.run(
            ['python', 'main.py', 'train', '--data', '../../data/empty.csv', '--test', '../../data/test.csv', '--model',
             'models/model.pkl', '--t'], capture_output=True, text=True)
    assert "Error: data is empty" == extract_assert(res.stderr)

    res = subprocess.run(
        ['python', 'main.py', 'train', '--data', '../../data/empty_without_text.csv', '--test', '../../data/test.csv',
         '--model',
         'models/model.pkl', '--t'], capture_output=True, text=True)
    assert "Error: missing 'text' column in train data" == extract_assert(res.stderr)

    res = subprocess.run(
        ['python', 'main.py', 'train', '--data', '../../data/test.csv', '--test', '../../data/empty_without_text.csv',
         '--model',
         'models/model.pkl', '--t'], capture_output=True, text=True)
    assert "Error: missing 'text' column in test data" == extract_assert(res.stderr)

    res = subprocess.run(
        ['python', 'main.py', 'train', '--data', '../../data/empty_with_text.csv', '--test', '../../data/test.csv',
         '--model',
         'models/model.pkl', '--t'], capture_output=True, text=True)
    assert "Error: missing 'rating' column in train data" == extract_assert(res.stderr)

    res = subprocess.run(
            ['python', 'main.py', 'train', '--data', '../../data/train.csv', '--test', '../../data/empty_with_text.csv', '--model',
             'models/model.pkl', '--t'], capture_output=True, text=True)
    assert "Error: missing 'rating' column in test data" == extract_assert(res.stderr)

    res = subprocess.run(
        ['python', 'main.py', 'train', '--data', 'test.py', '--test', '../../data/empty_with_text.csv',
         '--model',
         'models/model.pkl', '--t'], capture_output=True, text=True)
    assert "Error: Incorrect file type" == extract_assert(res.stderr)

    os.remove('models/model.pkl')

def test_predict():
    subprocess.run(
        ['python', 'main.py', 'train', '--data', '../../data/train.csv', '--test', '../../data/test.csv', '--model',
         'models/model.pkl', '--t'])

    res = subprocess.run('python main.py predict --model models/model.pkl --data ../../data/review.csv', shell=True, capture_output=True, text=True)
    assert 'negative review 0\npositive review 1\npositive review 1\nnegative review 0\n' == res.stdout # Check an output

    # Check asserts
    res = subprocess.run('python main.py predict --model ../../data/test.csv --data ../../data/review.csv', shell=True, capture_output=True, text=True)
    assert "Error: Not a pickle file" == extract_assert(res.stderr)

    res = subprocess.run('python main.py predict --model models/model.pkl --data ../../data/empty.csv',
                         shell=True,
                         capture_output=True, text=True)
    assert "Error: data is empty" == extract_assert(res.stderr)

    res = subprocess.run('python main.py predict --model models/model.pkl --data ../../data/empty_without_text.csv', shell=True,
                         capture_output=True, text=True)
    assert "Error: missing 'text' column in data" == extract_assert(res.stderr)

    os.remove('models/model.pkl')

def test_split():
    out = subprocess.run(
        'python main.py train --data ../../data/singapore_airlines_reviews.csv --split 0.3 --test ../../data/test.csv --model models/model.pkl --seed 37',
        shell=True, capture_output=True, text=True)
    assert "Error: choose either split or test" == extract_assert(out.stderr)

    out = subprocess.run(
        'python main.py train --data ../../data/singapore_airlines_reviews.csv --split 12.0 --model models/model.pkl --seed 37',
        shell=True, capture_output=True, text=True)
    assert "Error: split should be between 0 and 1" == extract_assert(out.stderr)

    out1 = subprocess.run('python main.py train --data ../../data/singapore_airlines_reviews.csv --split 0.3 --model models/model.pkl --seed 37', shell=True, capture_output=True, text=True)
    os.remove('models/model.pkl')
    out2 = subprocess.run('python main.py train --data ../../data/singapore_airlines_reviews.csv --split 0.3 --model models/model.pkl --seed 73 --t', shell=True, capture_output=True, text=True)

    assert out1.stdout != out2.stdout   # Check random-seed

    data = pd.read_csv('../../data/singapore_airlines_reviews.csv')
    train = pd.read_csv('train_debug.csv', index_col='IND')
    test = pd.read_csv('test_debug.csv', index_col='IND')

    assert train.shape[0] + test.shape[0] == data.shape[0]
    assert test.shape[0] == int(0.3 * data.shape[0])    # Check split

    not_shuffle_train, not_shuffle_test = train_test_split(data, test_size=0.3, shuffle=False, random_state=73)
    assert (not np.array_equal(train.index.array, not_shuffle_train.index.array)) and (not np.array_equal(test.index.array, not_shuffle_test.index.array))  # Check shuffle

    os.remove('models/model.pkl')
    os.remove('train_debug.csv')
    os.remove('test_debug.csv')
