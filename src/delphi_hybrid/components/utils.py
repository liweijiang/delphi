import pandas as pd
import spacy
import json
import os
import sys
import time
import openai

sys.path.append(os.getcwd())

from src.delphi_hybrid.components.constants import *
from src.delphi_hybrid.components.bank import *

# sys.path.append("/Users/liweijiang/Desktop/delphi_algo/scripts/utils/")
# from constants import *
# from bank import *

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")

def get_is_correct(original_class_label, original_is_correct, class_label):
    if original_is_correct:
        if original_class_label in [-1] and class_label in [-1]:
            return True
        elif original_class_label in [0, 1] and class_label in [0, 1]:
            return True
        else:
            return False
    else:
        if original_class_label in [-1] and class_label in [0, 1]:
            return True
        elif original_class_label in [0, 1] and class_label in [-1]:
            return True
        else:
            return False


def get_class_label_from_probs(prob_minus_1, prob_0, prob_1):
    if all(prob_minus_1 > p for p in [prob_0, prob_1]):
        return -1
    elif all(prob_1 > p for p in [prob_minus_1, prob_0]):
        return 1
    else:
        return 0


def gen_gpt3(p, n=20, model="text-davinci-003", temperature=0.7, max_tokens=1000):
    response = None
    while response is None:
        try:
            response = openai.Completion.create(
                model=model,
                prompt=p,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                n=n
            )
        except:
            time.sleep(0.3)
            print(f"[Try Again!] {sys.exc_info()[0]}")
    return response


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def gpt3_completion(prompt, model_name, max_tokens, temperature, logprobs, echo, num_outputs, top_p, best_of):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    # prevent over 600 requests per minute

    while not received:
        try:
            response = openai.Completion.create(
                engine=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=logprobs,
                echo=echo,
                stop=["."],
                n=num_outputs,
                top_p=top_p,
                best_of=best_of)
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError: # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False
            print("API error:", error)
            time.sleep(0.3)
    return response


########################## get ##########################
def get_delphi_scores(event, delphi_scorer, delphi_scores_cache, filename=None, is_save=True):
    if event not in delphi_scores_cache:
        class_label, probs, text_label = delphi_scorer.generate_with_score(event)
        delphi_scores_cache = add_delphi_scores_to_cache(delphi_scores_cache,
                                                         event,
                                                         class_label,
                                                         probs,
                                                         text_label,
                                                         filename=filename,
                                                         is_save=is_save)
    else:
        print(">> Delphi scores exist in cache!")
        class_label = delphi_scores_cache[event]["class_label"]
        probs = [delphi_scores_cache[event]["prob_1"],
                 delphi_scores_cache[event]["prob_0"],
                 delphi_scores_cache[event]["prob_minus_1"]]
        text_label = delphi_scores_cache[event]["text_label"]
    return class_label, probs, text_label, delphi_scores_cache


########################## add to cache ##########################
def add_paraphrases_to_cache(paraphrases_cache, event, paraphrases, filename=None, is_save=True):
    paraphrases_cache[event] = list(set(paraphrases))
    if is_save:
        cache_file_path = get_cache_path("paraphrases", filename)
        print(">> Save paraphrases to", cache_file_path)
        with open(cache_file_path, 'w') as f:
            json.dump(paraphrases_cache, f)
    return paraphrases_cache

# def add_compositions_to_cache(compositions_cache, event, compositions, filename=None, is_save=True):
#     compositions_cache[event] = compositions
#     if is_save:
#         cache_file_path = data_base_path + "cache/compositions.json"
#         print(">> Save compositions to", cache_file_path)
#         with open(cache_file_path, 'w') as f:
#             json.dump(compositions_cache, f)
#     return compositions_cache


# def add_delphi_scores_to_cache(delphi_scores_cache, event, class_label, probs, text_label, filename=None, is_save=True):
#     delphi_scores_cache[event] = {"class_label": class_label,
#                                   "prob_1": probs[0],
#                                   "prob_0": probs[1],
#                                   "prob_minus_1": probs[2],
#                                   "text_label": text_label,}
#     if is_save:
#         cache_file_path = get_cache_path("delphi_scores", filename)
#         print("Save delphi score to", cache_file_path)
#         with open(cache_file_path, 'w') as f:
#             json.dump(delphi_scores_cache, f)
#     return delphi_scores_cache
#
#
# def add_comet_inferences_to_cache(comet_cache, event, comet_inferences, filename=None, is_save=True):
#     comet_cache[event] = comet_inferences
#     if is_save:
#         cache_file_path = get_cache_path("comet", filename)
#         print("Save comet inferences to", cache_file_path)
#         with open(cache_file_path, 'w') as f:
#             json.dump(comet_cache, f)
#     return comet_cache




########################## parse sequence ##########################
def parse_sequence(input_sequence, is_dependency_parse=False):
    doc = nlp(input_sequence)

    noun_phrases = [chunk.text for chunk in doc.noun_chunks]

    tokens_list = [str(token) for (i, token) in enumerate(doc)]
    verbs = [(i, token) for (i, token) in enumerate(doc) if token.pos_ == "VERB"]
    conjunctions = [(i, token) for (i, token) in enumerate(doc) if token.pos_[-4:] == "CONJ"]

    if is_dependency_parse:
        tokens = [(i, token, token.pos_, token.dep_) for (i, token) in enumerate(doc)] # , token.head.text, token.head.pos_, [child for child in token.children]
    else:
        tokens = [(i, token, token.pos_) for (i, token) in enumerate(doc)]

    lemmatized_tokens_list = [str(token.lemma_) for (i, token) in enumerate(doc)]
    lemmatized_verbs = [(i, token.lemma_) for (i, token) in enumerate(doc) if token.pos_ == "VERB"]
    lemmatized_conjunctions = [(i, token.lemma_) for (i, token) in enumerate(doc) if
                               (token.pos_[-4:] == "CONJ" or str(token) in ["to", "for", "when"])]

    if is_dependency_parse:
        lemmatized_tokens = [(i, token.lemma_, token.pos_, token.dep_) for (i, token) in enumerate(doc)]
    else:
        lemmatized_tokens = [(i, token.lemma_, token.pos_) for (i, token) in enumerate(doc)]



    return {"tokens": {"tokens_list": tokens_list,
                       "tokens_dict": tokens,
                       "verbs_dict": verbs,
                       "conjunctions_dict": conjunctions},

            "lemmatized_tokens": {"tokens_list": lemmatized_tokens_list,
                                  "tokens_dict": lemmatized_tokens,
                                  "verbs_dict": lemmatized_verbs,
                                  "conjunctions_dict": lemmatized_conjunctions},

            "noun_phrases": noun_phrases}


def segment_string_by_verb(parsed_input_sequence):
    lemmatized_verbs = parsed_input_sequence["lemmatized_tokens"]["verbs_dict"]
    tokens_list = parsed_input_sequence["tokens"]["tokens_list"]
    lemmatized_tokens_list = parsed_input_sequence["lemmatized_tokens"]["tokens_list"]

    verbs_indices = [v[0] for v in lemmatized_verbs]
    verbs_indices = [0] + verbs_indices + [len(tokens_list)]

    segment_tokens = []
    segment_lemmatized_tokens = []
    segment_original_indices = []
    segment_indices = []
    for i in range(1, len(verbs_indices) - 1):
        start_idx = verbs_indices[i - 1]
        end_idx = verbs_indices[i + 1]
        anchor_idx = verbs_indices[i]
        if i != 1:
            start_idx = start_idx + 1

        seg_toks = tokens_list[start_idx: end_idx]
        seg_lemmatized_toks = lemmatized_tokens_list[start_idx: end_idx]

        end_idx = end_idx - 1

        segment_tokens.append(seg_toks)
        segment_lemmatized_tokens.append(seg_lemmatized_toks)
        segment_original_indices.append([start_idx, anchor_idx, end_idx])
        segment_indices.append([0, anchor_idx - start_idx, end_idx - start_idx])

    if len(lemmatized_verbs) == 0:
        print("*** Warning! ***\nNo verbs in sequence:", tokens_list)

    return {
        # "segment_sequences": [" ".join(s) for s in segment_tokens],
        # "segment_lemmatized_sequences": [" ".join(s) for s in segment_lemmatized_tokens],
        "segment_tokens": segment_tokens,
        "segment_lemmatized_tokens": segment_lemmatized_tokens,
        "segment_indices": segment_indices,
        "segment_original_indices": segment_original_indices,
        "tokens": tokens_list,
        "lemmatized_tokens": lemmatized_tokens_list,
    }


def segment_string_by_conj(parsed_input_sequence):
    lemmatized_conjs = parsed_input_sequence["lemmatized_tokens"]["conjunctions_dict"]
    tokens_list = parsed_input_sequence["tokens"]["tokens_list"]
    lemmatized_tokens_list = parsed_input_sequence["lemmatized_tokens"]["tokens_list"]
    # for c in parsed_input_sequence["tokens"]["tokens_dict"]:
    #     print(c)

    # print(tokens_list)
    # print(lemmatized_conjs)

    conjs_indices = [c[0] for c in lemmatized_conjs]
    conjs_indices = [0] + conjs_indices + [len(tokens_list)]
    # print(conjs_indices)

    # print("!" * 20)

    segment_tokens = []
    segment_lemmatized_tokens = []
    segment_original_indices = []
    segment_indices = []
    for i in range(1, len(conjs_indices)):
        start_idx = conjs_indices[i - 1]
        end_idx = conjs_indices[i] - 1
        # anchor_idx = conjs_indices[i]

        if i != 1:
            start_idx = start_idx + 1

        # print(start_idx, end_idx) #, anchor_idx)

        seg_toks = tokens_list[start_idx: end_idx + 1]
        seg_lemmatized_toks = lemmatized_tokens_list[start_idx: end_idx + 1]

        #
        #     end_idx = end_idx - 1
        #     print(seg_toks)
        #     print(seg_lemmatized_toks)
        #     print(start_idx, end_idx)
        #     print(0, end_idx - start_idx)
        #
        #     # print(" ".join(tokens_list[start_ind: end_ind]))
        segment_tokens.append(seg_toks)
        segment_lemmatized_tokens.append(seg_lemmatized_toks)
        segment_original_indices.append([start_idx, end_idx])
        segment_indices.append([0, end_idx - start_idx])

    if len(lemmatized_conjs) == 0:
        print("*** Warning! ***\nNo conjs in sequence:", tokens_list)

    return {
        "segment_tokens": segment_tokens,
        "segment_lemmatized_tokens": segment_lemmatized_tokens,
        "segment_indices": segment_indices,
        "segment_original_indices": segment_original_indices,
        "tokens": tokens_list,
        "lemmatized_tokens": lemmatized_tokens_list,
    }


def get_string_from_tokens(tokens, sep=" "):
    return sep.join(tokens)


def get_list_strings_from_list_tokens(list_tokens):
    return [get_string_from_tokens(t) for t in list_tokens]


def _get_sub_sequences(tokens, sub_seq_set):
    # Iterate over the entire string
    for i in range(len(tokens)):

        # Iterate from the end of the string
        # to generate substrings
        for j in range(len(tokens), i, -1):
            sub_tokens = tokens[i: i + j]
            sub_seq_set.add(" ".join(sub_tokens))

            # Drop kth character in the substring
            # and if its not in the set then recur
            for k in range(1, len(sub_tokens)):
                sb = sub_tokens[:]
                sb.pop(k)
                _get_sub_sequences(sb, sub_seq_set)


def get_sub_sequences(tokens):
    sub_seq_set = set()
    _get_sub_sequences(tokens, sub_seq_set)
    return list(sub_seq_set)



def read_json(json_file_path, is_load=False):
    with open(json_file_path, 'r') as j:
        if is_load:
            data = json.load(j)
        else:
            data = json.loads(j.read())
    return data


def save_json(json_file_path, data_to_save):
    with open(json_file_path, 'w') as f:
        json.dump(data_to_save, f)


def normalize_label(label):
    if type(label) != type(""):
        return ""
    label = label.lower()

    while len(label) > 0 and label[-1] == " ":
        label = label[:-1]
    while len(label) > 0 and label[-1] == ".":
        label = label[:-1]

    if label.startswith("its"):
        label_tokens = label.split(" ")
        label_tokens[0] = "it's"
        label = " ".join(label_tokens)
    if label.startswith("no "):
        label_tokens = label.split(" ")
        label_tokens[0] = "no,"
        label = " ".join(label_tokens)
    if label.startswith("yes "):
        label_tokens = label.split(" ")
        label_tokens[0] = "yes,"
        label = " ".join(label_tokens)
    if label.startswith("it's"):
        label_tokens = label.split(" ")
        label_tokens[0] = "it is"
        label = " ".join(label_tokens)
    if label.startswith("it’s"):
        label_tokens = label.split(" ")
        label_tokens[0] = "it is"
        label = " ".join(label_tokens)
    if label.startswith("it is considered"):
        label = label.replace("it is considered", "it is")
    if label.startswith("is "):
        label_tokens = label.split(" ")
        label_tokens[0] = "it is"
        label = " ".join(label_tokens)
    if label.startswith("isn't "):
        label_tokens = label.split(" ")
        label_tokens[0] = "it is not"
        label = " ".join(label_tokens)
    if label.startswith("it is generally"):
        label = label.replace("it is generally", "it is")
    if label.startswith("it can be"):
        label = label.replace("it can be", "it is")
    if label.startswith("it is often"):
        label = label.replace("it is often", "it is")
    if "can't" in label:
        label = label.replace("can't", "cannot")
    if label.startswith("so it is"):
        label = label.replace("so it is", "it is")
    if label.startswith("so it's"):
        label = label.replace("so it's", "it is")
    if label.startswith("it was"):
        label = label.replace("it was", "it is")
    if label.startswith("it is never"):
        label = label.replace("it is never", "it is not")
    if label.startswith("it would be"):
        label = label.replace("it would be", "it is")
    if label.startswith("we should"):
        label = label.replace("we should", "you should")
    if label.startswith("i should"):
        label = label.replace("i should", "you should")
    if label.startswith("they are"):
        label = label.replace("they are", "it is")
    if label.startswith("they're"):
        label = label.replace("they're", "it is")
    if label.startswith("it isn't"):
        label = label.replace("it isn't", "it is not")
    if label.startswith("it is so"):
        label = label.replace("it is so", "it is")
    if label.startswith("it is very"):
        label = label.replace("it is very", "it is")
    if label.startswith("it is still"):
        label = label.replace("it is still", "it is")
    if "you're" in label:
        label = label.replace("you're", "you are")
    if "shouldn't" in label:
        label = label.replace("shouldn't", "should not")
    if "shouldnt" in label:
        label = label.replace("shouldnt", "should not")
    if "don't" in label:
        label = label.replace("don't", "do not")
    if "doesn't" in label:
        label = label.replace("doesn't", "does not")
    if label.startswith("murder is"):
        label = label.replace("murder is", "it is")
    if label.startswith("killing is"):
        label = label.replace("killing is", "it is")
    if label.startswith("that's "):
        label = label.replace("that's ", "it is ")

    if label.startswith("it is "):
        label = label.replace("it is ", "")
    return label