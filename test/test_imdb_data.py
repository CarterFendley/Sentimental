
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import re
import torch
import numpy as np
import imdb_data
import unittest
MOCK_POSITIVE = [f'positive review mock {i}' for i in range(100)]
MOCK_NEGATIVE = [f'negative review mock {i}' for i in range(100)]

MOCK_DATA = {
    'train_x': np.zeros(4),
    'train_y': np.zeros(4),
    'test_x': np.zeros(4),
    'test_y': np.zeros(4),
    'valid_x': np.zeros(4),
    'valid_y': np.zeros(4),
    'vocab_to_int': {'mock': 0},
    'int_to_vocab': {0: 'mock'}
}

class TestLoading(unittest.TestCase):

    def test_raw_load(self):
        positive, negative = imdb_data.load_raw_classes('test/mock_data')
        
        self.assertEqual(positive[0], 'positive test1')
        self.assertEqual(positive[1], 'positive train1')
        self.assertEqual(negative[0], 'negative test1')
        self.assertEqual(negative[1], 'negative train1')
    
    def test_load_data_basic(self):
        out = imdb_data.load_data(
            positive=['hello'],
            negative=['bye'],
            write_to_cache=False)

    def _test_output(self, out, verbose=False):
        # Pull out similar parts for ease of use later
        data_subsets = (
            (out['train_x'], out['train_y']),
            (out['test_x'], out['test_y']),
            (out['valid_x'], out['valid_y'])
        )

        int_to_vocab = out['int_to_vocab']
        vocab_to_int = out['vocab_to_int']

        if verbose: print('Testing data split...')
        self.assertEqual(len(out['train_x']), 160) # 200 * 0.8 = 160
        self.assertEqual(len(out['train_y']), 160)
        self.assertEqual(len(out['test_x']), 20)
        self.assertEqual(len(out['test_y']), 20)
        self.assertEqual(len(out['valid_x']), 20)
        self.assertEqual(len(out['valid_y']), 20)

        if verbose: print('Testing vocab size...')
        # 4 unique words
        # 100 unique numbers
        # 1 padding char
        self.assertEqual(len(out['vocab_to_int']), 105)

        if verbose: print('Testing data types...')
        self.assertTrue(isinstance(int_to_vocab, dict))
        self.assertTrue(isinstance(vocab_to_int, dict))
        for reviews, labels in data_subsets:
            self.assertTrue(isinstance(reviews, np.ndarray))
            self.assertTrue(isinstance(labels, np.ndarray))

            for r, l in zip(reviews, labels):
                self.assertTrue(isinstance(r, np.ndarray))
                self.assertTrue(isinstance(l, np.int64))
        

        if verbose: print('Testing tokenization and labels...')
        pos_pattern = 'positive review mock [0-9]{1,2}' # 1,2 numbers
        neg_pattern = 'negative review mock [0-9]{1,2}' 

        for reviews, labels in data_subsets:
            for r, l in zip(reviews, labels):
                # Decode the tokenization
                plain_text = ' '.join([int_to_vocab[int(w)] for w in r])
                l = int(l)

                if l == 1:
                    self.assertTrue(re.fullmatch(pos_pattern, plain_text))
                elif l == 0:
                    self.assertTrue(re.fullmatch(neg_pattern, plain_text))
                else:
                    raise TypeError('Non binary label!')
                
                # Test that vocab_to_int produce same encoding
                encoded = [vocab_to_int[word] for word in plain_text.split()]
                self.assertTrue(np.array_equal(r,np.array(encoded)))

        if verbose: print('Testing split')
        count = 0
        for label in out['valid_y']:
            if int(label) == 1:
                count += 1
        self.assertTrue(count >= 3) #TODO: Shitty probalistic test

    def test_load_data_from_input(self):
        out = imdb_data.load_data(
            positive=MOCK_POSITIVE,
            negative=MOCK_NEGATIVE,
            write_to_cache=False,
            pad_to=4)
        
        self._test_output(out)

    def _rm_cache_path(self, cache_path):
        # If is link to cache just unlink
        if os.path.islink(cache_path):
            os.unlink(cache_path)
            return

        remove_files = (
            f'train_x.npy',
            f'train_y.npy',
            f'test_x.npy',
            f'test_y.npy',
            f'validate_x.npy',
            f'validate_y.npy',
            f'vocab_to_int.npy',
            f'int_to_vocab.npy',
            f'hash.txt'
        )

        for f in remove_files:
            # Remove the files
            path = os.path.join(cache_path, f)
            if os.path.isfile(path):
                os.remove(path)
        # Remove cache dir
        os.rmdir(cache_path)

    def _clean_testing_cache(self, cache_dir):
        if not os.path.exists(cache_dir):
            return
        
        os.unlink(os.path.join(cache_dir, 'latest'))

        for d in os.listdir(cache_dir):
            dir_path = os.path.join(cache_dir, d)
            self._rm_cache_path(dir_path)
        # Remove the cache
        if os.path.exists(cache_dir):
            os.rmdir(cache_dir)

    def _is_valid_cache(self, cache_path):
        check_files = (
            f'train_x.npy',
            f'train_y.npy',
            f'test_x.npy',
            f'test_y.npy',
            f'validate_x.npy',
            f'validate_y.npy',
            f'vocab_to_int.npy',
            f'int_to_vocab.npy',
            f'hash.txt'
        )

        for f in check_files:
            self.assertTrue(os.path.isfile(os.path.join(cache_path, f)))
        return True

    def _check_testing_cache(self, cache_dir):
        # Assert that cache exists
        self.assertTrue(os.path.isdir(cache_dir))

        # Check caches for files
        dir_count = 0
        for d in os.listdir(cache_dir):
            dir_count += 1
            dir_path = os.path.join(cache_dir, d)
            self.assertTrue(os.path.isdir(dir_path))
            self.assertTrue(self._is_valid_cache(dir_path))

        self.assertEqual(dir_count, 2) # Latest & one created

        # Check for latests symlink
        latest_path = os.path.join(cache_dir, 'latest')
        self.assertTrue(os.path.islink(latest_path))

    def test_cache_hashing(self):
        v1_pattern = 'v1-[a-z0-9]+'

        hash_label = imdb_data.create_hash_string_v1(MOCK_DATA)

        self.assertTrue(re.fullmatch(v1_pattern, hash_label))

    def test_saving_data(self):
        current_dir = os.path.dirname(__file__)
        fake_cache_dir = os.path.join(current_dir, '_test_cache')
        fake_save_dir = os.path.join(current_dir, '_test_save')

        self._clean_testing_cache(fake_cache_dir)
        out = imdb_data.load_data(
            positive=MOCK_POSITIVE,
            negative=MOCK_NEGATIVE,
            _cache_dir=fake_cache_dir,
            pad_to = 4 
        )

        self._check_testing_cache(fake_cache_dir)
        # Validate output
        self._test_output(out)

        # Test saving
        imdb_data.save_processed_data(out, fake_save_dir, _cache_dir=fake_cache_dir)
        self._check_testing_cache(fake_cache_dir)
        self._is_valid_cache(fake_save_dir)
        self._rm_cache_path(fake_save_dir)

        # Test loading
        load = imdb_data.load_processed_data(imdb_data.get_latest_cache_path(_cache_dir=fake_cache_dir))
        self._test_output(load)

        # Make sure saving uses old cache
        imdb_data.save_processed_data(load, fake_save_dir, _cache_dir=fake_cache_dir)
        self._check_testing_cache(fake_cache_dir)
        self._is_valid_cache(fake_save_dir)
        self._rm_cache_path(fake_save_dir)

        self._clean_testing_cache(fake_cache_dir)

    # check model dir
    # Model post fix addition
