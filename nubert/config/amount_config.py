from typing import Dict, List, Optional
from typing_extensions import Self
import os
from pydantic import BaseModel, model_validator, ConfigDict
from .nubert_config import TrainerConfig


class AmountConfig(BaseModel):
    model_config  = ConfigDict(protected_namespaces=())
    model_name: str
    file_name: str = "amount_raw"
    run_name: str = "amount-transactions-5-stride-1-randomize-False-bins-20"
    dataset_path: str = "/notebooks/nubank/nubert/analyses/amount-2012-2013"
    max_length: int = 512
    num_transactions: int = 5
    stride: int = 1
    num_bins: int = 20
    from_cleaned_data: bool = False
    randomize_column_order: bool = False
    columns_to_drop: List[str] = ["Posted Date", "Year-Month", "Cardholder Last Name", "Cardholder First Initial", "Agency Number"]
    filter_list: Optional[list[str]] = None
    agency_names_to_remove: List[str] = []
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
        self.run_name = f"amount-transactions-{self.num_transactions}-stride-{self.stride}-randomize-{self.randomize_column_order}-bins-{self.num_bins}"
        return self

    @model_validator(mode="after")
    def share_run_name_with_trainer(self) -> Self:
        self.trainer.run_name = self.run_name
        return self
    
    @model_validator(mode="after")
    def change_output_dir_name(self) -> Self:
        self.trainer.output_dir = os.path.join(self.trainer.output_dir, self.run_name)
        return self