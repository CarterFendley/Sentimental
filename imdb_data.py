import re
import os
import numpy as np
from tqdm import tqdm
from hashlib import sha256
from string import punctuation
from collections import Counter

import torch
from torch.utils.data import DataLoader, TensorDataset

DIRNAME = os.path.dirname(__file__)
SEQ_LENGTH = 400

DEFAULT_IMDB_DIR = os.path.join(DIRNAME, 'data/imdb_raw')
DEFAULT_CACHE_DIR = os.path.join(DIRNAME, 'data/cache_imdb_data')

UTF_8 = 'utf-8'

def load_raw_classes(data_dir=DEFAULT_IMDB_DIR):
    print('Loading positive reviews...')
    _test_dir = os.path.join(data_dir, 'test/pos')
    _train_dir = os.path.join(data_dir, 'train/pos') 

    # TODO: Crap code (memory)
    paths = [f'{_test_dir}/{path}' for path in os.listdir(_test_dir)]
    paths.extend([f'{_train_dir}/{path}'  for path in os.listdir(_train_dir)])
    
    positive = []

    for f in tqdm(paths):
        if os.path.isfile(f):
            with open(f, 'r') as fp:
                positive.append(fp.read())
    
    print('Loading negative reviews...')
    _test_dir = os.path.join(data_dir, 'test/neg')
    _train_dir = os.path.join(data_dir, 'train/neg') 

    # TODO: Crap code (memory)
    paths = [f'{_test_dir}/{path}' for path in os.listdir(_test_dir)]
    paths.extend([f'{_train_dir}/{path}'  for path in os.listdir(_train_dir)])
    
    negative = []

    for f in tqdm(paths):
        if os.path.isfile(f):
            with open(f, 'r') as fp:
                negative.append(fp.read())
    
    return positive, negative

##############################
# Caching Helper Functions   #
##############################

def get_hash_type(string):
    v1_pattern = 'v1-[a-z0-9]+'

    if re.fullmatch(v1_pattern, string):
        return 'v1'

def validate_v1_data(data):
    subsets = (
        'train_x',
        'train_y',
        'test_x',
        'test_y',
        'valid_x',
        'valid_y'
    )

    # Assert types
    for subset in subsets:
        assert type(data[subset]) == np.ndarray, 'Subsets should be numpy arrays'
    assert type(data['vocab_to_int']) == dict, 'Mapping functions should be dictionaries'
    assert type(data['int_to_vocab']) == dict, 'Mapping functions should be dictionaries'

def create_hash_string_v1(data):
    hash_list = [
        data['train_x'],
        data['train_y'],
        data['test_x'],
        data['test_y'],
        data['valid_x'],
        data['valid_y'],
        data['vocab_to_int'],
        data['int_to_vocab']
    ]

    hashes = ''.join([sha256(str(x).encode(UTF_8)).hexdigest() for x in hash_list])

    return f'v1-{sha256(hashes.encode(UTF_8)).hexdigest()}'

def get_latest_cache_path(_cache_dir=DEFAULT_CACHE_DIR):
    return os.path.join(_cache_dir, 'latest')

def latest_cache_exists(latest_cache=None, _cache_dir=DEFAULT_CACHE_DIR):
    if latest_cache is None:
        latest_cache = get_latest_cache_path(_cache_dir=_cache_dir)

    if os.path.exists(latest_cache) and os.path.islink(latest_cache):
        return True

    return False

def update_latest_cache(cache_path, _cache_dir=DEFAULT_CACHE_DIR):
    latest_cache = get_latest_cache_path(_cache_dir=_cache_dir)
    
    if latest_cache_exists(latest_cache=latest_cache):
        # Replace link
        os.remove(latest_cache)

    os.symlink(cache_path, latest_cache)

def is_valid_cache(cache_path):
    check_files = (
        f'{cache_path}/train_x.npy',
        f'{cache_path}/train_y.npy',
        f'{cache_path}/test_x.npy',
        f'{cache_path}/test_y.npy',
        f'{cache_path}/validate_x.npy',
        f'{cache_path}/validate_y.npy',
        f'{cache_path}/vocab_to_int.npy',
        f'{cache_path}/int_to_vocab.npy',
        f'{cache_path}/hash.txt'
    )

    for path in check_files:
        if not os.path.isfile(path):
            return False
    return True

def save_to_cache(data, _cache_dir=DEFAULT_CACHE_DIR, _hash_label=None):
    if _hash_label is None:
        _hash_label = create_hash_string_v1(data)

    data_cache_path = os.path.join(_cache_dir, _hash_label)
    # Check cache for hit on hash
    if not os.path.exists(data_cache_path):
        # If no cache hit, save in cache
        os.makedirs(data_cache_path)

        np.save(f'{data_cache_path}/train_x.npy', data['train_x'])
        np.save(f'{data_cache_path}/train_y.npy', data['train_y'])
        np.save(f'{data_cache_path}/test_x.npy', data['test_x'])
        np.save(f'{data_cache_path}/test_y.npy', data['test_y'])
        np.save(f'{data_cache_path}/validate_x.npy', data['valid_x'])
        np.save(f'{data_cache_path}/validate_y.npy', data['valid_y'])
        np.save(f'{data_cache_path}/vocab_to_int.npy', data['vocab_to_int'])
        np.save(f'{data_cache_path}/int_to_vocab.npy', data['int_to_vocab'])

        with open(f'{data_cache_path}/hash.txt', 'w') as f:
            f.write(_hash_label)
    elif not os.path.isdir(data_cache_path) or not is_valid_cache(data_cache_path):
        raise AssertionError('Cache corrupted (either not a dir or files missing)!')
    
    update_latest_cache(data_cache_path, _cache_dir=_cache_dir)

def save_processed_data(data, save_dir, _cache_dir=None):
    if _cache_dir is None:
        _cache_dir = DEFAULT_CACHE_DIR

    validate_v1_data(data)
    hash_label = create_hash_string_v1(data)

    save_dir = os.path.join(save_dir)
    data_cache_path = os.path.join(_cache_dir, hash_label)
    
    save_to_cache(data, _cache_dir=_cache_dir, _hash_label=hash_label)

    # Symlink to directory
    os.symlink(data_cache_path, save_dir)
    # Update the link to the latest

def load_processed_data(cache_path=get_latest_cache_path()):
    # Check cache files
    assert is_valid_cache(cache_path)

    # Load data
    data = {
            'train_x': np.load(f'{cache_path}/train_x.npy'),
            'train_y': np.load(f'{cache_path}/train_y.npy'),
            'test_x': np.load(f'{cache_path}/test_x.npy'),
            'test_y': np.load(f'{cache_path}/test_y.npy'),
            'valid_x': np.load(f'{cache_path}/validate_x.npy'),
            'valid_y': np.load(f'{cache_path}/validate_y.npy'),
            'vocab_to_int': np.load(f'{cache_path}/vocab_to_int.npy', allow_pickle=True).item(), # Item pulls the dict out I think
            'int_to_vocab': np.load(f'{cache_path}/int_to_vocab.npy', allow_pickle=True).item()
    }

    # Validate format of data
    validate_v1_data(data)

    return data

######################
# Data Processing    #
######################

def filter_reviews(vector):
    processed_vectors = []
    for i, review in tqdm(enumerate(vector)):
        # Convert to lower case
        review = review.lower()

        # Remove punc
        review = ''.join([char for char in review if char not in punctuation])

        processed_vectors.append(review)
    
    return processed_vectors

def tokenize_and_pad(review_vectors, mapping, pad_to=SEQ_LENGTH):
    tokenized_vectors = []
    print('Tokenizing reviews...')
    for review in tqdm(review_vectors):
        tokenized = [mapping[w] for w in review .split()]

        # Reject if of zero length 
        if len(tokenized) > 0:
            tokenized_vectors.append(tokenized)

    # Add padding
    tokenized_features = np.zeros((len(tokenized_vectors), pad_to), dtype=int)
    print('Padding review features...')
    for i, review in tqdm(enumerate(tokenized_vectors)):
        length = len(review)

        # TODO: Can't this be done better
        if length <= pad_to:
            zeros = list(np.zeros(pad_to - length))
            review = zeros+review
        elif length > pad_to:
            review = review[0:pad_to]
        
        tokenized_features[i,:] = np.array(review)
    
    return tokenized_features

def load_data(
    positive=None,
    negative=None,
    preprocess_func=filter_reviews,
    pad_to=SEQ_LENGTH,
    ignore_cache=False,
    _cache_dir=DEFAULT_CACHE_DIR,
    imdb_data_dir=DEFAULT_IMDB_DIR,
    write_to_cache=True,
    confirmation=True):

    if positive is None or negative is None:
        if not ignore_cache:
            if latest_cache_exists(_cache_dir=_cache_dir):
                if confirmation:
                    while True:
                        print('Found saved cached data load!')
                        c = input('Load data from cache [Y/n]:')
                        c = c.lower()
                        if c in ('yes', 'y', ''):
                            return load_processed_data()
                        elif c in ('no', 'n'):
                            break
                else:
                    return load_processed_data()
            print('No cache found!')
        
        print('Genertating data...')
        positive, negative = load_raw_classes()
    
    print('Preprocessing positive reviews...')
    positive = preprocess_func(positive)
    print('Preprocessing negative reviews...')
    negative = preprocess_func(negative)

    num_positive = len(positive)
    num_negative = len(negative)

    # Tokenize set
    combined = positive[:]
    combined.extend(negative)

    all_text = ' '.join(combined)
    words = all_text.split()

    word_counter = Counter(words)
    num_words = len(word_counter)
    sorted_words = word_counter.most_common(num_words)

    # Create forward and reverse tokenizers reserving 0 for padding
    vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
    int_to_vocab = {i+1:w for i, (w,c) in enumerate(sorted_words)}

    vocab_to_int['<!PAD!>'] = 0
    int_to_vocab[0] = '<!PAD!>'

    print('Positive reviews: ')
    positive = tokenize_and_pad(positive, vocab_to_int, pad_to=pad_to)
    print('Negative reviews: ')
    negative = tokenize_and_pad(negative, vocab_to_int, pad_to=pad_to)

    
    # Combine tokenized sets and create matching labels
    print('Shuffling data...')
    combined = np.concatenate((positive, negative), axis=0)
    
    combined_labels = [1]*len(positive)
    combined_labels.extend([0]*len(negative))
    combined_labels = np.array(combined_labels)

    # Use data loader to hackly shuffle the postive / negative together
    combined_data = TensorDataset(torch.from_numpy(combined), torch.from_numpy(combined_labels))
    combined_loader = DataLoader(combined_data, shuffle=True, batch_size=1)

    combined = []
    combined_labels = []

    for review, label in combined_loader:
        # Pull zero index from batch cause hacky shit
        combined.append(np.array(review[0]))
        combined_labels.append(int(label[0]))
    combined = np.array(combined)
    combined_labels = np.array(combined_labels)

    # Split data into different sets
    print('Splitting data...')
    split_frac = 0.8
    total_reviews = len(combined)

    train_x = combined[0:int(split_frac*total_reviews)]
    train_y = combined_labels[0:int(split_frac*total_reviews)]

    remaining_x = combined[int(split_frac*total_reviews):]
    remaining_y = combined_labels[int(split_frac*total_reviews):]

    valid_x = remaining_x[0:int(len(remaining_x)*0.5)]
    valid_y = remaining_y[0:int(len(remaining_y)*0.5)]

    test_x = remaining_x[int(len(remaining_x)*0.5):]
    test_y = remaining_y[int(len(remaining_y)*0.5):]
    print('\tTrain:', len(train_x))
    print('\tTest:', len(test_x))
    print('\tValidate:', len(valid_x))

    data = {
        'train_x': train_x,
        'train_y': train_y,
        'valid_x': valid_x,
        'valid_y': valid_y,
        'test_x': test_x,
        'test_y': test_y,
        'vocab_to_int': vocab_to_int,
        'int_to_vocab': int_to_vocab
    }

    if write_to_cache:
        save_to_cache(data, _cache_dir=_cache_dir)

    return data