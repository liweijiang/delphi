import os
import sys
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.delphi_hybrid.components.utils import *


class CompositionalityParser():
    def __init__(self):
        self.to_exclude_list = ["able", "want", "refuse", "try", "due", "go",
                                "supposed", "claim", "pretend",
                                # https://grammar.collinsdictionary.com/us/easy-learning/which-verbs-are-followed-by-the-to-infinitive-in-english
                                "agree", "arrange", "attempt", "choose", "decide", "fail", "hope", "learn", "manage", "offer", "plan", "seem", "come",
                                "how", "when", "which", "what", "struggle", "remember"]  # "lie",

        self.phrases_to_replace = {"in order to": "to", "so that": "so"}

        self.relative_pronounce_list = [
            "that", "which", "who", "whom", "where"]

        self.for_exclude_list = ["pay"]  # "lie",

        self.adj_exclude_list = ["nuclear"]
        self.conj_additional_list = [
            "otherwise", "to", "for", "so", "because", "by", "or"]  # , "if", "when"

    def fix_event(self, event):
        for pr in self.phrases_to_replace:
            if pr in event:
                event = event.replace(pr, self.phrases_to_replace[pr])
        return event

    def organize_subevents(self, tokens, lemmatized_tokens, poss, deps, idxs, idxs_to_segment, original_tokens):
        all_segments = []
        last_idx = 0
        for current_idx in idxs_to_segment:
            if last_idx in idxs_to_segment:
                all_segments.append([tokens[last_idx],
                                     [tokens[last_idx]],
                                     [lemmatized_tokens[last_idx]],
                                     [poss[last_idx]],
                                     [deps[last_idx]],
                                     [idxs[last_idx]],
                                     [original_tokens[last_idx]],
                                     "conjunction"])
                last_idx += 1
            all_segments.append([" ".join(tokens[last_idx:current_idx]),
                                 tokens[last_idx:current_idx],
                                 lemmatized_tokens[last_idx:current_idx],
                                 poss[last_idx:current_idx],
                                 deps[last_idx:current_idx],
                                 idxs[last_idx:current_idx],
                                 original_tokens[last_idx:current_idx],
                                 "content"])
            last_idx = current_idx

        if last_idx in idxs_to_segment:
            all_segments.append([tokens[last_idx],
                                 [tokens[last_idx]],
                                 [lemmatized_tokens[last_idx]],
                                 [poss[last_idx]],
                                 [deps[last_idx]],
                                 [idxs[last_idx]],
                                 [original_tokens[last_idx]],
                                 "conjunction"])
            last_idx += 1
        all_segments.append([" ".join(tokens[last_idx:]),
                             tokens[last_idx:],
                             lemmatized_tokens[last_idx:],
                             poss[last_idx:],
                             deps[last_idx:],
                             idxs[last_idx:],
                             original_tokens[last_idx:],
                             "content"])
        segments = [segment for segment in all_segments if segment[0] != ""]
        return segments

    def get_subevents(self, tokens, lemmatized_tokens, poss, deps):
        original_tokens = [None for _ in range(len(tokens))]
        conjs_idxs_global = [i for i, pos in enumerate(poss) if (
            (len(pos) > 3 and pos[-4:] == "CONJ") or lemmatized_tokens[i] in self.conj_additional_list)]
        conjs_idxs_global += [len(original_tokens) - 1]

        idxs = [i for i in range(len(tokens))]
        idxs_to_segment = []
        for i, token in enumerate(tokens):
            pos = poss[i]

            if (len(pos) > 3 and pos[-4:] == "CONJ") or token in self.conj_additional_list:
                if i != 0 and lemmatized_tokens[i - 1] not in self.to_exclude_list \
                        and poss[i - 1] not in ["ADJ"]:
                    # print(pos, token)
                    conj_idx_local = conjs_idxs_global.index(i)
                    # print(conj_idx_local)

                    if conj_idx_local < (len(conjs_idxs_global) - 1):
                        next_conj_idx_global = conjs_idxs_global[conj_idx_local + 1]

                        # if there's no verb in the next sequence, then don't segment
                        if "VERB" in poss[i: next_conj_idx_global + 1] \
                                or "be" in lemmatized_tokens[i: next_conj_idx_global + 1]:
                            # print(pos, token)
                            idxs_to_segment.append(i)

        subevents = self.organize_subevents(tokens, lemmatized_tokens, poss, deps,
                                            idxs, idxs_to_segment, original_tokens)
        return subevents

    def _get_relative_clause(self, subevent):
        tokens = subevent[1]
        poss = subevent[3]

        for i, token in enumerate(tokens):
            if token in self.relative_pronounce_list and i != 0 and poss[i - 1] in ["NOUN"] \
                    or token in ["who", "whom"]:
                return [
                    [" ".join(tokens[:i])] + [l[:i]
                                              for l in subevent[1:-1]] + ["content"],
                    [" ".join(tokens[i:i+1])] + [l[i:i+1]
                                                 for l in subevent[1:-1]] + ["relative pronoun"],
                    [" ".join(tokens[i+1:])] + [l[i+1:]
                                                for l in subevent[1:-1]] + ["relative clause"]
                ]

    def _get_all_subevents(self, event):
        event = self.fix_event(event)
        parsed_event = parse_sequence(event, is_dependency_parse=True)

        tokens = parsed_event["tokens"]["tokens_list"]
        lemmatized_tokens = parsed_event["lemmatized_tokens"]["tokens_list"]
        poss = [_token[2] for _token in parsed_event["tokens"]["tokens_dict"]]
        deps = [_token[3] for _token in parsed_event["tokens"]["tokens_dict"]]

        return self.get_subevents(tokens, lemmatized_tokens, poss, deps)

    def glue_subevents(self, subevents):
        glued_subevent = subevents[0]

        for e in subevents[1:]:
            for i, c in enumerate(e[1:-1]):
                glued_subevent[i + 1] += c

        glued_subevent[0] = " ".join(glued_subevent[1])
        glued_subevent[-1] = "content"

        return glued_subevent

    def _get_parsed_event(self, event, is_simple=True):
        subevents = self._get_all_subevents(event)
        parsed_event = {}

        relative_clause = self._get_relative_clause(subevents[0])
        if relative_clause != None:
            parsed_event["main_action"] = {"main_clause": relative_clause[0],
                                           "relative_pronoun": relative_clause[1], "relative_clause": relative_clause[2]}
        else:
            parsed_event["main_action"] = {"main_clause": subevents[0]}

        if len(subevents) > 1:
            parsed_event["connector"] = subevents[1]
            parsed_event["other_action"] = self.glue_subevents(subevents[2:])

        if not is_simple:
            return parsed_event
        else:
            main_action = parsed_event["main_action"]["main_clause"][0]
            relative_pronoun = None
            relative_clause = None
            connector = None
            other_action = None
            if "relative_pronoun" in parsed_event["main_action"]:
                relative_pronoun = parsed_event["main_action"]["relative_pronoun"][0]
                relative_clause = parsed_event["main_action"]["relative_clause"][0]

            if "connector" in parsed_event:
                connector = parsed_event["connector"][0]
                other_action = parsed_event["other_action"][0]

            return {"main_action": main_action,
                    "relative_pronoun": relative_pronoun,
                    "relative_clause": relative_clause,
                    "connector": connector,
                    "other_action": other_action}

    def get_parsed_event(self, event):
        parsed_events = self._get_parsed_event(event, is_simple=True)

        main_event = parsed_events["main_action"]
        main_event_main_clause = parsed_events["main_action"]
        main_event_relative_pronoun = parsed_events["relative_pronoun"]
        main_event_relative_clause = parsed_events["relative_clause"]
        adjunct_event = parsed_events["other_action"]

        if main_event_main_clause != None and main_event_relative_clause != None:
            main_event = main_event_main_clause + " " + \
                main_event_relative_pronoun + " " + main_event_relative_clause

        return {"main_event": main_event,
                "main_event_main_clause": main_event_main_clause,
                "main_event_relative_clause": main_event_relative_clause,
                "adjunct_event": adjunct_event, }


if __name__ == "__main__":
    compositionality_parser = CompositionalityParser()

    events = []
    for split in ["test", "validation"]:
        input_file = data_base_path + \
            f"cache_norm_bank/events/clean_{split}.moral_acceptability.tsv"
        df_data = pd.read_csv(input_file, sep="\t")
        events += df_data["clean_event"].tolist()

    data_to_save = {}
    for event in tqdm(events):
        parsed_event = compositionality_parser.get_parsed_event(event)
        data_to_save[event] = parsed_event

    save_json(data_base_path + f"cache_norm_bank/constituents.json", data_to_save)
