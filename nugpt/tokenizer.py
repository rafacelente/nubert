from transformers.tokenization_utils import PreTrainedTokenizer
from .vocabulary import Vocabulary

class TabFormerTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab: Vocabulary,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
    ):
        self.vocab = vocab
        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token)

    def get_vocab(self):
        return self.vocab.id2token