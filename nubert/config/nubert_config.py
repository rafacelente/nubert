from typing import Dict, List, Optional
from typing_extensions import Self
import os
from pydantic import BaseModel, model_validator, ConfigDict


class TrainerConfig(BaseModel):
    learning_rate: float = 5e-5
    num_train_epochs: float = 1.0
    seed: int = 42
    gradient_accumulation_steps: int = 1
    bf16: bool = True
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    output_dir: str = "/notebooks/nubank/"
    overwrite_output_dir: bool = True
    save_total_limit: int = 1
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    report_to: str = "wandb"
    remove_unused_columns: bool = False
    load_best_model_at_end: bool = True
    logging_steps: int = 5
    run_name: str = "nubert-distil-transactions-5-stride-1-randomize-False-bins-20"



class NubertPreTrainConfig(BaseModel):
    model_config  = ConfigDict(protected_namespaces=())
    model_name: str = "distilbert/distilbert-base-uncased"
    file_name: str = "nubank_raw"
    run_name: str = " "
    dataset_path: str = "/notebooks/nubank/"
    max_length: int = 512
    num_transactions: int = 5
    stride: int = 1
    num_bins: int = 20
    randomize_column_order: bool = False
    from_cleaned_data: bool = False
    columns_to_drop: List[str] = ["Posted Date", "Year-Month", "Cardholder Last Name", "Cardholder First Initial", "Agency Number"]
    filter_list: Optional[list[str]] = None
    agency_names_to_remove: List[str] = ["EMPLOYEE BENEFITS"]
    nrows: Optional[int] = None
    trainer: TrainerConfig = TrainerConfig()


    @classmethod
    def from_dict(cls, data: Dict) -> Self:
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> Self:
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @model_validator(mode="after")
    def set_run_name(self) -> Self:
        self.run_name = f"{self.run_name}-nubert-distil-transactions-{self.num_transactions}-stride-{self.stride}-randomize-{self.randomize_column_order}-bins-{self.num_bins}"
        return self

    @model_validator(mode="after")
    def share_run_name_with_trainer(self) -> Self:
        self.trainer.run_name = self.run_name
        return self
    
    @model_validator(mode="after")
    def change_output_dir_name(self) -> Self:
        self.trainer.output_dir = os.path.join(self.trainer.output_dir, self.run_name)
        return self