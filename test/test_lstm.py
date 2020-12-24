
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import re
import torch
import numpy as np
import lstm
from imdb_data import load_processed_data
import unittest

DIRNAME = os.path.dirname(__file__)
MOCK_DATA_HASH = 'v1-fa3eb3a31d632c8c9ddf034ad5819bb7fd8f98ebda1daa3950b14f0987c9cadd'
MOCK_DATA_PATH = os.path.join(DIRNAME, f'mock_cache/{MOCK_DATA_HASH}')

MOCK_DATA = load_processed_data(MOCK_DATA_PATH)

MOCK_CONFIG = dict(
    epochs = 3,
    batch_size = 50,
    lr = 0.001,
    seq_len = 4,
    output_size = 1,
    embedding_dim = 400,
    hiddem_dim = 500,
    n_layers = 2,
    clip_grad = 5,
    dataset='IMDB',
    architecture='LSTM'
)

'''
TODO: 
- Assert gradients dont change with scoring helpers 
'''

class TestLSTM(unittest.TestCase):

    def test_make_model(self):
        model = lstm.make_model(MOCK_CONFIG, MOCK_DATA)

        self.assertTrue(isinstance(model, lstm.SentimentLSTM))
        self.assertTrue(isinstance(model, torch.nn.Module))

        #self._test_model(model)

    def _test_model(self, model):
        pass

    def _compare_data_hash(self, save_path, hash):
        hash_label = None
        with open(f'{save_path}/data_cache/hash.txt', 'r') as f:
            hash_label = f.read()
        
        self.assertEqual(hash_label, hash)

    def _validate_model_save(self, save_path):
        self.assertTrue(os.path.isfile(f'{save_path}/model_trained.pt'))
        self.assertTrue(os.path.isfile(f'{save_path}/model_config.json'))

        self.assertTrue(os.path.islink(f'{save_path}/data_cache'))
        # Data cache should be covered by other tests
    
    def _validate_saves_dir(self, saves_dir):
        for f in os.listdir(saves_dir):
            f_path = os.path.join(saves_dir, f)
            self._validate_model_save(f_path)

    def _rm_model_save(self, save_path):
        os.remove(f'{save_path}/model_trained.pt')
        os.remove(f'{save_path}/model_config.json')

        os.unlink(f'{save_path}/data_cache')
        os.rmdir(save_path)

    def _clean_test_model_saves(self, saves_dir):
        if os.path.exists(saves_dir):
            for f in os.listdir(saves_dir):
                f_path = os.path.join(saves_dir, f)
                self._rm_model_save(f_path)
        
            os.rmdir(saves_dir)


    def test_saving_model(self):
        fake_save_dir = os.path.join(DIRNAME, '_fake_model_saves')
        fake_cache_dir = os.path.join(DIRNAME, 'mock_cache')

        self._clean_test_model_saves(fake_save_dir)

        # Build model and save
        model = lstm.make_model(MOCK_CONFIG, MOCK_DATA)
        lstm.save_model(
            model,
            MOCK_CONFIG,
            MOCK_DATA,
            save_dir=fake_save_dir,
            _cache_dir=fake_cache_dir,
            name='test_save',
            time_stamp=False
        )

        save_path = os.path.join(fake_save_dir, 'test_save')

        self._validate_saves_dir(fake_save_dir) # Save dir is valid
        self._validate_model_save(save_path) # Test save is valid
        self._compare_data_hash(save_path, MOCK_DATA_HASH) # Compare data hash

        # Test that loading works 
        model_loaded, config, data = lstm.load_model(save_path)

        self.assertEqual(MOCK_CONFIG, config)
        for key in MOCK_DATA.keys():
            self.assertTrue(np.array_equal(MOCK_DATA[key], data[key]))
        
        self._clean_test_model_saves(fake_save_dir)

    