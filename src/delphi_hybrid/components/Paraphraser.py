import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
import re
import random
from scripts.utils.utils import *
from scripts.utils.WANLIScorer import WANLIScorer

# S: Putting my mother in a retirement home.
# A: Placing my mother in a retirement home.
# A: Killing someone because you are being kind.

PROMPT_SOURCE = """
Paraphrase the given sentence. 
S: Torturing a cat if other people cannot see it.
A: {}
S: Not caring for Christmas.
A: Being indifferent towards Christmas.
S: Killing someone with kindness.
A: Killing someone for showing your kindness.
S: {}
A:
"""

PARAPHRASE_OPTIONS = [
    "Torturing a cat assuming that other people cannot see it.",
    "Torturing a cat on the assumption that other people cannot see it.",
    "Torturing a cat in case other people cannot see it.",
    "Torturing a cat, given that other people cannot see it.",
    "Torturing a cat if nobody can see it."
]


class Paraphraser():
    def __init__(self, model_name="text-davinci-003"): # "text-davinci-003", "text-curie-001"
        self.MODEL_NAME = model_name
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.CONDITIONED_GEN_TOKEN = "<|endoftext|>"
        self.nli_scorer = WANLIScorer()

    def fix_sentence(self, input_str):
        if input_str == "":                    # Don't change empty strings.
            return input_str
        if input_str[-1] in ["?", ".", "!"]:   # Don't change if already okay.
            return input_str
        if input_str[-1] == ",":               # Change trailing ',' to '.'.
            return input_str[:-1] + "."
        return input_str + "."

    def _qualify_paraphrase(self, action_doc, paraphrased_action_doc):
        action = action_doc.text.lower()
        paraphrased_action = paraphrased_action_doc.text.lower()

        num_tokens_action = len(action_doc)
        num_tokens_paraphrased_action = len(paraphrased_action_doc)

        action_lemma = " ".join([token.lemma_ for token in action_doc]).lower()
        paraphrased_action_lemma = " ".join([token.lemma_ for token in paraphrased_action_doc]).lower()

        if action_lemma in paraphrased_action_lemma and action_lemma != paraphrased_action_lemma:
            return False

        if any(paraphrased_action.startswith(prefix) for prefix in ["it's ", "it is "]) \
            and not any(action.startswith(prefix) for prefix in ["it's ", "it is "]):
            return False

        if ":" not in action and ":" in paraphrased_action:
            return False

        if not re.search('[a-zA-Z]', paraphrased_action):
            return False

        if paraphrased_action.lower() in [action.lower(), "n/a"]:
            return False

        return abs(num_tokens_action - num_tokens_paraphrased_action) / num_tokens_action < 1

    def qualify_paraphrase(self, action, paraphrased_action):
        action_doc = nlp(action)
        paraphrased_action_doc = nlp(paraphrased_action)
        return self._qualify_paraphrase(action_doc, paraphrased_action_doc)

    def generate_paraphrases(self, action, num_paraphrases=1, max_iteration_count=20):
        fixed_action = self.fix_sentence(action.capitalize())
        action_doc = nlp(action)

        paraphrases = []
        iteration_count = 0
        while len(paraphrases) < num_paraphrases and iteration_count < max_iteration_count:
            _po = random.sample(PARAPHRASE_OPTIONS, 1)[0]
            _p = PROMPT_SOURCE.format(_po, fixed_action)

            # try:
            #     response = gpt3_completion(_p, model_name="text-davinci-003", max_tokens=100, temperature=0.9,
            #                                logprobs=0, echo=False, num_outputs=10, top_p=1.0, best_of=1)
            #     iteration_count += 1
            # except Exception as e:
            #     continue

            response = gen_gpt3(_p, n=5, model=self.MODEL_NAME, temperature=0.8, max_tokens=400)
            iteration_count += 1
            paraphrased_action = [c['text'].strip() for c in response['choices']]

            for paraphrase in paraphrased_action:
                paraphrase_doc = nlp(paraphrase)
                if self._qualify_paraphrase(action_doc, paraphrase_doc):
                    paraphrases.append(paraphrase)
            paraphrases = list(set(paraphrases))
            print(len(paraphrases))

        return {
            "action": action,
            "fixed_action": fixed_action,
            "paraphrases": paraphrases
        }

if __name__ == "__main__":
    # paraphraser = Paraphraser()

    # action = "genocide if it creates job"
    # action = "Having sex with your best friend"
    # action = "Putting my mother in a retirement home"
    # action = "Killing a baby because you really want to"
    # action = "Stabbing someone who opened your mail without permission"
    # action = "Killing someone with kindness"
    # action = "Robbing a bank because it is my job"
    # action = "Let people play games and allow them to win lots of money"
    # action = "My husband doesn't know that our daughter is not his"
    # action = "Making money instead of saving someone"
    # print(paraphraser.generate_paraphrases(action, num_paraphrases=6))

    # premise = "killing a bear to save your child."
    # hypothesis = "killing a bear."
    # prediction = wanli_scorer.get_scores(premise, hypothesis)
    # print(premise, "|", hypothesis, ":", prediction)
    openai.api_key = os.environ["OPENAI_API_KEY"]
    print(gen_gpt3("_p", n=20, model="text-davinci-003", temperature=0.7, max_tokens=400))




