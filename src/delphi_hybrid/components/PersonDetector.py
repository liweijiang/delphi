import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
import random
from scripts.utils.utils import *

PROMPT_SOURCE = "Q: Is your mom's best friend a person? A:Yes\
                 Q: Is your dog a person? A:No\
                 Q: Is your boss a person? A:Yes\
                 Q: Is lesbian a person? A:Yes\
                 Q: Is Dave a person? A:Yes\
                 Q: Is Chinese a person? A:Yes\
                 Q: Is servant a person? A:Yes\
                 Q: Is the alarm a person? A:No\
                 Q: Is {obj} a person? A:"

class PersonDetector():
    def __init__(self, model_name="text-davinci-002"):
        self.MODEL_NAME = model_name
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.CONDITIONED_GEN_TOKEN = "<|endoftext|>"

    def _convert_singular(self, obj):
        obj_doc = nlp(obj)
        obj_lemma = [t.lemma_ for t in obj_doc]
        return " ".join(obj_lemma)


    def is_person(self, obj, max_iter=10):
        obj_lemma = self._convert_singular(obj)
        _p = PROMPT_SOURCE.format(obj=obj_lemma)

        iter = 0
        # while answer.lower() not in ["yes", "no"] and max_iter < 10:
        answers = []
        while len(answers) < 5 and iter < max_iter:
            iter += 1
            response = gpt3_completion(_p, model_name="text-davinci-002", max_tokens=64, temperature=0.9, logprobs=0, echo=False, num_outputs=1, top_p=1.0, best_of=1)
            answer = response["choices"][0]["text"]
            if answer.lower() in ["yes", "no"]:
                answers.append(answer)
        answer = max(answers, key=answers.count)
        return answer.lower() == "yes"

if __name__ == "__main__":
    person_detector = PersonDetector()

    # all_objs = ["marking"]

    all_objs = ["fictional character","teacher", "service dog", "zoo keeper", "midget", "chino", "jew", "paraplegic", "rapist",
    "my moms", "Israeli", "gay", "homosexuals", "women", "Emily", "cuban", "Raymond", "slave"
    "paint mark", "Yes", "Smurf", "Goblin", "olive oil", "god", "unborn baby", "putin",
    "candy bars", "dirty sock", "tissues", "tiger", "elephants", "books", "pencil", "eraser"]

    # "teacher", "service dog", "zoo keeper", "midget", "chino", "jew", "paraplegic", "rapist",
    # "my moms", "Israeli", "gay", "homosexuals", "women", "Emily", "cuban", "Raymond", "slave"
    # "paint mark", "Yes", "Smurf", "Goblin", "olive oil", "god", "unborn baby", "putin",
    # "candy bars", "dirty sock", "tissues", "tiger", "elephants", "books", "pencil", "eraser"

    for obj in all_objs:
        answer = person_detector.is_person(obj)
        print(obj, "|", answer)
