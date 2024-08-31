# import os
# import sys
# print(sys.path)
# sys.path.append(os.getcwd())
# sys.path.append(sys.path[0] + "/utils")

from LMScorer import *

import math
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class GPT2Scorer(LMScorer):
    def __init__(self, model_name="gpt2-large", cuda_id=0):
        super().__init__(model_name)

        self.device = f"cuda:{cuda_id}"
        self.model_name = model_name
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.BOS_TOKEN = self.tokenizer.bos_token
        self.EOS_TOKEN = self.tokenizer.eos_token

    def __name__(self):
        return self.model_name

    def get_input_perplexity(self, input_sequence):
        """
            Helper function to get the perplexity of a sequence from GPT3
        """
        formatted_input_sequence = self.BOS_TOKEN + '"' + input_sequence[0].upper() + input_sequence[1:] + '"' + self.EOS_TOKEN
        return self._get_input_perplexity(formatted_input_sequence)

    def _get_input_perplexity(self, formatted_input_sequence):
        full_input_ids = self.tokenizer.encode(formatted_input_sequence, return_tensors='pt').to(self.device)
        full_loss = self.model(full_input_ids, labels=full_input_ids)[0] * (full_input_ids.shape[1] - 1)
        loss = full_loss / full_input_ids.shape[1]
        return math.exp(loss.item())


    def get_input_perplexity_combo(self, input_sequence, return_all_ppl=False):
        """
            Helper function to get the perplexity of a sequence from GPT3
        """
        formatted_input_sequences = [
            self.BOS_TOKEN + '"' + input_sequence[0].upper() + input_sequence[1:] + '."' + self.EOS_TOKEN
        ]
        return self._get_input_perplexity_combo(formatted_input_sequences, return_all_ppl)


if __name__ == "__main__":
    gpt2_scorer = GPT2Scorer()

    premise = "killing a bear to save your child."
    print(gpt2_scorer.get_input_perplexity(premise))

    premise = "killing."
    print(gpt2_scorer.get_input_perplexity(premise))

    premise = "kiling."
    print(gpt2_scorer.get_input_perplexity(premise))

