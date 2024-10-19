from .nubert_config import NubertPreTrainConfig
import os

NUBERT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
NUBERT_DEFAULT_CONFIG_FILE = os.path.join(NUBERT_ROOT_DIR, 'config', 'default_config.yaml')
