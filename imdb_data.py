import os
import numpy as np
from tqdm import tqdm
from string import punctuation
from collections import Counter
import torch
from torch.utils.data import DataLoader, TensorDataset

DIRNAME = os.path.dirname(__file__)
SEQ_LENGTH = 400

def load_raw_classes(data_dir):
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

def saved_data_exists(cache_dir='data/cached_data_load'):
    check_files = (
        f'{cache_dir}/train_x.npy',
        f'{cache_dir}/train_y.npy',
        f'{cache_dir}/test_x.npy',
        f'{cache_dir}/test_y.npy',
        f'{cache_dir}/validate_x.npy',
        f'{cache_dir}/validate_y.npy',
        f'{cache_dir}/vocab_to_int.npy',
        f'{cache_dir}/int_to_vocab.npy'
    )

    for path in check_files:
        if not os.path.isfile(path):
            return False
    return True

def preprocessing(vector):
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
        elif review_len > pad_to:
            review = review[0:pad_to]
        
        tokenized_features[i,:] = np.array(review)
    
    return tokenized_features


def load_data(positive=None, negative=None, preprocess_func=preprocessing,
             from_cache=False, cache_dir='data/cached_data_load', disable_caching=False,
             pad_to=SEQ_LENGTH):
    if saved_data_exists(cache_dir=cache_dir):
        if not from_cache:
            while True:
                print('Found saved cached data load!')
                c = input('Load data from cache [Y/n]:')
                c = c.lower()
                if c in ('yes', 'y', ''):
                    from_cache = True
                    break
                elif c in ('no', 'n'):
                    from_cache = False
                    break
    elif from_cache:
        raise AssertionError('Cached data load either partial or non-existant')

    if from_cache:
        return {
            'train_x': np.load(f'{cache_dir}/train_x.npy'),
            'train_y': np.load(f'{cache_dir}/train_y.npy'),
            'test_x': np.load(f'{cache_dir}/test_x.npy'),
            'test_y': np.load(f'{cache_dir}/test_y.npy'),
            'valid_x': np.load(f'{cache_dir}/validate_x.npy'),
            'valid_y': np.load(f'{cache_dir}/validate_y.npy'),
            'vocab_to_int': np.load(f'{cache_dir}/vocab_to_int.npy', allow_pickle=True).item(),
            'int_to_vocab': np.load(f'{cache_dir}/int_to_vocab.npy', allow_pickle=True).item()
        }

    if positive is None or negative is None:
        print('Loading classes data from file...')
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

    # Cache data through numpy saves
    if not disable_caching:
        print(f'Caching current load in {cache_dir}...')
        os.makedirs(cache_dir)

        np.save(f'{cache_dir}/train_x.npy', train_x)
        np.save(f'{cache_dir}/train_y.npy', train_y),
        np.save(f'{cache_dir}/test_x.npy', test_x),
        np.save(f'{cache_dir}/test_y.npy', test_y),
        np.save(f'{cache_dir}/validate_x.npy', valid_x),
        np.save(f'{cache_dir}/validate_y.npy', valid_y),
        np.save(f'{cache_dir}/vocab_to_int.npy', vocab_to_int)
        np.save(f'{cache_dir}/int_to_vocab.npy', int_to_vocab)

    return {
        'train_x': train_x,
        'train_y': train_y,
        'valid_x': valid_x,
        'valid_y': valid_y,
        'test_x': test_x,
        'test_y': test_y,
        'vocab_to_int': vocab_to_int,
        'int_to_vocab': int_to_vocab
    }
