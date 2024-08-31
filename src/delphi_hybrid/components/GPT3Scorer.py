from scripts.sub_event_extraction.utils.LMScorer import *


import math
import openai


class GPT3Scorer(LMScorer):
    def __init__(self, model_name="text-davinci-003"): # "text-davinci-002"
        super().__init__(model_name)

        self.MODEL_NAME = model_name
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.CONDITIONED_GEN_TOKEN = "<|endoftext|>"

    def correct_grammatical_error(self, input_sequence):
        response = openai.Completion.create(
            model=self.MODEL_NAME,
            prompt="Correct this to standard English:\n\n" + input_sequence,
            temperature=0,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        corrected_sequence = response["choices"][0]["text"].split("\n\n")[-1]
        return corrected_sequence

    def get_input_perplexity(self, input_sequence):
        """
            Helper function to get the perplexity of a sequence from GPT3
        """
        # formatted_input_sequence = CONDITIONED_GEN_TOKEN + input_sequence[0].upper() + input_sequence[1:]
        # formatted_input_sequence = CONDITIONED_GEN_TOKEN + input_sequence[0].upper() + input_sequence[1:] + "."
        # formatted_input_sequence = CONDITIONED_GEN_TOKEN + "'" + input_sequence[0].upper() + input_sequence[1:] + ".'"
        formatted_input_sequence = self.CONDITIONED_GEN_TOKEN + input_sequence
        return self._get_input_perplexity(formatted_input_sequence)

    def _get_input_perplexity(self, formatted_input_sequence):
        """
            Helper function to get the perplexity of a sequence from GPT3
        """
        response = openai.Completion.create(
            model=self.MODEL_NAME,
            prompt=formatted_input_sequence,
            # prompt=formatted_input_sequence
            max_tokens=0,
            logprobs=1,
            echo=True
        )
        # print(formatted_input_sequence)

        tokens_logprobs = response["choices"][0]["logprobs"]["token_logprobs"][1:]
        num_tokens = len(tokens_logprobs)
        sequence_cross_entropy = -sum(tokens_logprobs) / num_tokens
        perplexity = math.exp(sequence_cross_entropy)
        return perplexity

    def get_input_perplexity_combo(self, input_sequence, return_all_ppl=False):
        """
            Get a combined perplexity averaged across multiple input sequence formats.
        """
        formatted_input_sequences = [
            self.CONDITIONED_GEN_TOKEN + input_sequence[0].upper() + input_sequence[1:],
            self.CONDITIONED_GEN_TOKEN + input_sequence[0].upper() + input_sequence[1:] + ".",
            self.CONDITIONED_GEN_TOKEN + '"' + input_sequence[0].upper() + input_sequence[1:] + '"',
        ]

        return self._get_input_perplexity_combo(formatted_input_sequences, return_all_ppl)

    def get_input_logprob_gpt3(self, input_sequence):
        """
            Helper function to get the perplexity of a sequence from GPT3
            """
        formatted_input_sequence = self.CONDITIONED_GEN_TOKEN + '"' + input_sequence[0].upper() + input_sequence[1:] + '"'
        response = openai.Completion.create(
            model=self.MODEL_NAME,
            prompt=formatted_input_sequence,
            max_tokens=0,
            logprobs=1,
            echo=True
        )

        tokens_logprobs = response["choices"][0]["logprobs"]["token_logprobs"][1:]
        return sum(tokens_logprobs)

if __name__ == "__main__":
    gpt3_scorer = GPT3Scorer()

    # premise = "killing a bear to save your child."
    # print(gpt3_scorer.get_input_perplexity_combo(premise))

    # premise = "killing a stupid bear to save your child"
    # print(gpt3_scorer.get_paraphrases(premise))
