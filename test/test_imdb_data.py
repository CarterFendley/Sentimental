
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import re
import torch
import numpy as np
import imdb_data
import unittest

class TestLoading(unittest.TestCase):

    def test_raw_load(self):
        positive, negative = imdb_data.load_raw_classes('test/mock_data')
        
        self.assertEqual(positive[0], 'positive test1')
        self.assertEqual(positive[1], 'positive train1')
        self.assertEqual(negative[0], 'negative test1')
        self.assertEqual(negative[1], 'negative train1')
    
    def test_load_data_basic(self):
        out = imdb_data.load_data(positive=['hello'], negative=['bye'], disable_caching=True)

    mock_positive = [f'positive review mock {i}' for i in range(100)]
    mock_negative = [f'negative review mock {i}' for i in range(100)]

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

                print(plain_text)
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
            positive=self.mock_positive,
            negative=self.mock_negative,
            disable_caching=True,
            pad_to=4)
        
        self._test_output(out)

    def _clean_testing_cache(self, cache_dir):
        remove_files = (
            f'{cache_dir}/train_x.npy',
            f'{cache_dir}/train_y.npy',
            f'{cache_dir}/test_x.npy',
            f'{cache_dir}/test_y.npy',
            f'{cache_dir}/validate_x.npy',
            f'{cache_dir}/validate_y.npy',
            f'{cache_dir}/vocab_to_int.npy',
            f'{cache_dir}/int_to_vocab.npy'
        )

        for f in remove_files:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(cache_dir):
            os.rmdir(cache_dir)

    def test_load_data_caching(self):
        test_dir = os.path.dirname(__file__)
        fake_cached_dir = os.path.join(test_dir, 'mock_cache')
        fake_caching_dir = os.path.join(test_dir, '_testing_cache')

        # Test that things properly cache
        self._clean_testing_cache(fake_caching_dir)
        out = imdb_data.load_data(
            positive=self.mock_positive,
            negative=self.mock_negative,
            cache_dir = fake_caching_dir,
            pad_to=4,
        )

        check_files = (
            f'{fake_caching_dir}/train_x.npy',
            f'{fake_caching_dir}/train_y.npy',
            f'{fake_caching_dir}/test_x.npy',
            f'{fake_caching_dir}/test_y.npy',
            f'{fake_caching_dir}/validate_x.npy',
            f'{fake_caching_dir}/validate_y.npy',
            f'{fake_caching_dir}/vocab_to_int.npy',
            f'{fake_caching_dir}/int_to_vocab.npy'
        )

        for f in check_files:
            self.assertTrue(os.path.exists(f)) 
        
        self._test_output(out) # Make sure caching output is correct
        self._clean_testing_cache(fake_caching_dir)

        # Test loading from cacke
        out = imdb_data.load_data(
            positive=self.mock_positive,
            negative=self.mock_negative,
            cache_dir = fake_cached_dir,
            from_cache = True,
            disable_caching = True,
            pad_to=4,
        )

        self._test_output(out)

    # 
    # TODO: Padding
