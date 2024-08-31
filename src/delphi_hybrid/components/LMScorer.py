import math
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class LMScorer:
    def __init__(self, model_name):
        print("Initialize model:",model_name)

    def get_input_perplexity(self, input_sequence):
        return

    def _get_input_perplexity(self, formatted_input_sequence):
        return

    def _get_input_perplexity_combo(self, formatted_input_sequences, return_all_ppl=False):
        all_ppl = []
        for formatted_input_sequence in formatted_input_sequences:
            ppl = self._get_input_perplexity(formatted_input_sequence)
            all_ppl.append(ppl)

        if return_all_ppl:
            return formatted_input_sequences, (all_ppl + [(sum(all_ppl) / len(all_ppl))])
        else:
            return (sum(all_ppl) / len(all_ppl))

    def get_input_perplexity_combo(self, input_sequence, return_all_ppl=False):
        return

