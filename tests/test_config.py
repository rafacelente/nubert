import pytest

def test_can_create_config():
    from nubert.config import NubertPreTrainConfig

    config = NubertPreTrainConfig()
    assert config is not None

def test_run_name_is_shared():
    from nubert.config import NubertPreTrainConfig

    config = NubertPreTrainConfig(
        run_name="test_run_name"
    )
    assert config.trainer.run_name == config.run_name

def can_create_trainer_arguments_from_config():
    from transformers import TrainingArguments
    from nubert.config import NubertPreTrainConfig
    import os

    config = NubertPreTrainConfig()
    config.trainer.output_dir = "/tmp"
    args = TrainingArguments(**config.trainer.model_dump())

    output_dir = os.path.join(config.trainer.output_dir, config.run_name)
    
    assert args is not None
    assert config.trainer.output_dir == output_dir
    assert config.trainer.output_dir == args.output_dir
    assert config.run_name == f"nubert-distil-transactions-{config.num_transactions}-stride-{config.stride}-randomize-{config.randomize_column_order}-bins-{config.num_bins}"
    assert args.run_name == config.run_name == config.trainer.run_name
    assert args.learning_rate == config.trainer.learning_rate

def test_can_create_nudataset_from_cleaned_data():
    from nubert.datasets import NuDataset
    import os

    file_path = os.path.join(os.path.dirname(__file__), 'artifacts', 'test_config.yaml')

    dataset = NuDataset.from_config_file(
        config_path=file_path,
    )

    assert dataset.max_seq_len == 512
    assert dataset.num_transaction_sequences == 5
    assert dataset.stride == 1
    assert dataset.trans_table is not None
