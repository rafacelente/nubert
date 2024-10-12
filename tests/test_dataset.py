import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path
from nugpt import NuDataset
import torch

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'AgencyNumber': ['A001', 'A001', 'A002', 'A002'],
        'Timestamp': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02'],
        'Amount': [100.0, 200.0, 150.0, 250.0],
        'MCC': ['TEST1', 'TEST2', 'TEST3', 'TEST4']
    })


@pytest.fixture
def mock_csv(tmp_path, sample_data):
    csv_file = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_file, index=False)
    return csv_file


def test_init(mock_csv):
    dataset = NuDataset(root=str(mock_csv.parent), fname="test_data")
    assert isinstance(dataset, NuDataset)
    assert dataset.tokenizer is not None

def test_len(mock_csv):
    dataset = NuDataset(root=str(mock_csv.parent), fname="test_data", seq_len=2, stride=1)
    assert len(dataset) > 0

def test_getitem(mock_csv):
    dataset = NuDataset(root=str(mock_csv.parent), fname="test_data", seq_len=2, stride=1)
    item = dataset[0]
    assert isinstance(item, torch.Tensor)
    assert item.shape[0] == 2  # seq_len

def test_encode_data(mock_csv):
    dataset = NuDataset(root=str(mock_csv.parent), fname="test_data")
    assert dataset.trans_table is not None
    assert 'Amount_bins' in dataset.encoder_fit
    assert 'Timestamp_bins' in dataset.encoder_fit

def test_init_tokenizer(mock_csv):
    dataset = NuDataset(root=str(mock_csv.parent), fname="test_data")
    assert dataset.tokenizer.get_vocab_size() > 0

def test_prepare_samples(mock_csv):
    dataset = NuDataset(root=str(mock_csv.parent), fname="test_data", seq_len=2, stride=1)
    assert len(dataset.data) > 0

def test_format_trans(mock_csv):
    dataset = NuDataset(root=str(mock_csv.parent), fname="test_data")
    trans_data, _, columns_names = dataset.user_level_data()
    user_token_ids = dataset.format_trans(trans_data[0], columns_names)
    assert len(user_token_ids) > 0