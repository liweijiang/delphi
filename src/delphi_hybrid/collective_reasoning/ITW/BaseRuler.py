import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.delphi_hybrid.components.PersonDetector import *
from src.delphi_hybrid.components.CompositionalityParser import *
from src.delphi_hybrid.components.MoralSaliencyKeywordIdentifier import *
from src.delphi_hybrid.components.bank import *
from src.delphi_hybrid.components.constants import *
from src.delphi_hybrid.components.utils import *


class BaseRuler():
    def __init__(self):
        self.all_event_types = None
        self.events = None
        self.events_by_agreement_rate = {}
        self.events_by_split = {}

        self.paraphrases_cache = None
        self.comet_cache = None
        self.delphi_cache = None

        self.saliency_identifier = None
        self.person_detector = None
        self.compose_parser = None

        self.paraphrase_to_event_map = {}
        self.top_level_keywords_map = None
        self.gold_data_map = None
        self.top_level_keywords_counts_map = {}
        self.constituents_map = {}
        self.raw_keywords_counts_map = {}
        self.binary_keywords_counts_map = {}
        self.average_paraphrases_map = {}

        self._init_constants()
        self._init_cache_data()
        self._init_gold_data()
        self._init_utils()

        self._init_constituents_map()
        self._init_raw_keywords_counts_map()
        self._init_top_level_keywords_counts_maps()
        self._init_binary_keywords_counts_map()
        self._init_average_paraphrases_map()

    def _init_constants(self):
        """
        Initialize constants
        """
        self.all_event_types = [
            "event", "main_event", "main_event_main_clause", "main_event_relative_clause", "adjunct_event"]

        self.top_level_keywords_map = {"kill": ["kill", "lose life", "die", "death", "dead", "suicide", "murder",
                                                "assassinate", "killed"],

                                       "pain physical": ["harm", "danger", "torture", "attack", "hit", "violent",
                                                         "hurt", "cruel", "injure", "hit", "destructive", "threaten",
                                                         "aggressive", "pain", "rape", "revenge", "destroy", "weapon",
                                                         "gun shot",
                                                         "injured", "attacked", "threatened", "destroyed", "get hit",
                                                         "get hurt", "raped"],

                                       "pain mental": ["anxious", "disgusted", "displeased", "hunger", "hungry",
                                                       "shame", "remorse", "fear", "anger", "angry", "outrage",
                                                       "guilt", "depress", "mad", "scare", "mental illness", "scare",
                                                       "sad", "terrified", "terror", "scared"],

                                       "disable": ["blind", "deaf", "disable", "phobia", "addiction", "unconscious",
                                                   "harm", "danger", "attack", "hit", "violent", "hurt", "destroy",
                                                   "destructive", "attacked", "get hurt", "get hit"],

                                       "no freedom": ["coercion", "no consent", "restrain", "not freedom", "not privacy",
                                                      "rape", "raped"],

                                       "no pleasure": ["deprive"],

                                       "deceive": ["lie", "deceive"],

                                       "no promise": ["break promise", "renege", "betray",
                                                      "betrayed"],

                                       "no cheat": ["adultery", "dishonest", "lose trust", "cheat", "unfair", "greedy",
                                                    "cheated"],  # , "no sex"

                                       "sex": ["sex"],

                                       "not sex": ["not sex"],

                                       "no law": ["illegal", "jail", "crime", "prison", "arrest", "guilty", "violate",
                                                  "violated", "arrested"],

                                       "duty": ["duty"],

                                       "conscious": ["conscious", "save life", "survive", "live life"],

                                       "able": ["ability", "intelligence", "talent", "protect", "safe",
                                                "welfare", "benefit", "take care of"],

                                       "freedom": ["freedom", "privacy", "gain trust", "fair", "show respect",
                                                   "gain respect"],

                                       "pleasure": ["pleasure", "enjoy", "smile", "joy", "happiness", "happy",
                                                    "satisfaction", "satisfy"],

                                       "derivative good": ["health", "wealth", "knowledge", "love", "friendship",
                                                           "peace", "education", "medicare", "have money",
                                                           "have job"],  # "spend money",

                                       "derivative evil": ["war", "poverty", "slum", "lose money", "lose job"],

                                       "unjust": ["discrimination", "bias", "favoritism", "hypocracy", "hypocrite",
                                                  "racist", "sexist", "unjust", "unfair"],

                                       "just": ["justice", "fair", "honest", "upright", "equitable", "reasonable",
                                                "impartial", "unbiased"],

                                       "moral": ["moral", "ethical"],

                                       "immoral": ["immoral", "unethical", "evil", "inhumane"],
                                       }

        self.all_keywords = self._init_all_keywords()

        self.events_to_exclude = ["playing a video game with tamim",
                                  "You use the word pog and poggers",
                                  "Beitar Jerusalem football club",
                                  "Buying a lootbox in my favourite online game",
                                  "Lolong is going to beat Ohma",
                                  "Up tilting Falcon 4 times in a row",
                                  "Playing new world with janus and cake",
                                  "drawing marianne and amita as sumo wrestlers",
                                  "Up tilting Falcon 4 times in a row",
                                  "Jail people who put ananas on pizzas",
                                  "say I love you to kenza sebti",
                                  "Someone is rubbing his hands intensly",
                                  "Anything to do with Rachel",
                                  "Solo open world pvp like a chad",
                                  "Allen institute for artificial intelligence",
                                  "Why is veganism not morally obligatory",
                                  "asking which moron decided this machines moral compass",
                                  "Make a robot to evaluate human morals",
                                  "tickling jimins balls backstage at the concert for 7 hours",
                                  "Triggering a Friend by killing his imagenary waifu",
                                  "Telling everyone that Andrej got peged with a cone",
                                  "Giving a nerd a swirlie because he enjoys Heidegger",
                                  "playing fire emblem heroes",
                                  "Being a bot that evaluates the morality of statements",
                                  "Smoking weed because you are snoop dogg",
                                  "Sending a kid directly to hell for losing a Yugioh duel",
                                  "Committing targeted genocide towards children in Fortnite",
                                  "let my mom do the dish when  im not working",
                                  "Stealing a road sign for decorating my uni flat",
                                  "Contributing to a mural for Watsuki",
                                  "Watching Jojo's bizarre adventure"
                                  ]

        self.raw_keywords_binary_map = {"kill": -1, "lose life": -1, "die": -1, "death": -1, "dead": -1, "killed": -1,
                                        "suicide": -1, "murder": -1, "assassinate": -1, "harm": -1, "danger": -1,
                                        "torture": -1, "attack": -1, "hit": -1, "violent": -1, "unethical": -1,
                                        "hurt": -1, "cruel": -1, "injure": -1, "destructive": -1, "threaten": -1,
                                        "aggressive": -1, "pain": -1, "rape": -1, "revenge": -1, "destroy": -1,
                                        "weapon": -1, "gun shot": -1, "injured": -1, "attacked": -1, "threatened": -1,
                                        "destroyed": -1, "get hit": -1, "get hurt": -1, "raped": -1, "immoral": -1,
                                        "anxious": -1, "disgusted": -1, "displeased": -1, "hypocrite": -1,
                                        "shame": -1, "remorse": -1, "fear": -1, "anger": -1, "angry": -1, "outrage": -1,
                                        "guilt": -1, "mad": -1, "scare": -1, "mental illness": -1, "depress": -1,
                                        "terrified": -1, "terror": -1, "scared": -1, "greedy": -1, "cheated": -1,
                                        "blind": -1, "deaf": -1, "disable": -1, "phobia": -1, "addiction": -1,
                                        "unconscious": -1, "no privacy": -1, "unfair": -1, "betray": -1, "betrayed": -1,
                                        "coercion": -1, "no consent": -1, "restrain": -1, "no freedom": -1,
                                        "deprive": -1, "lie": -1, "deceive": -1, "break promise": -1, "renege": -1,
                                        "adultery": -1, "sex": -1, "dishonest": -1, "lose trust": -1, "cheat": -1,
                                        "illegal": -1, "jail": -1, "crime": -1, "prison": -1, "arrest": -1, "guilty": -1,
                                        "violate": -1, "violated": -1, "arrested": -1, "racist": -1, "sexist": -1,
                                        "war": -1, "poverty": -1, "slum": -1, "lose money": -1, "lose job": -1,
                                        "discrimination": -1, "bias": -1, "favoritism": -1, "hypocracy": -1, "evil": -1,
                                        "inhumane": -1, "not sex": -1, "unjust": -1,

                                        "conscious": 1, "save life": 1, "survive": 1, "live life": 1, "equitable": 1,
                                        "ability": 1, "intelligence": 1, "talent": 1, "protect": 1, "safe": 1,
                                        "welfare": 1, "benefit": 1, "take care of": 1, "gain respect": 1,
                                        "freedom": 1, "privacy": 1, "gain trust": 1, "fair": 1, "show respect": 1,
                                        "pleasure": 1, "enjoy": 1, "smile": 1, "joy": 1, "happiness": 1, "happy": 1,
                                        "satisfaction": 1, "satisfy": 1, "moral": 1, "ethical": 1, "honest": 1,
                                        "health": 1, "wealth": 1, "knowledge": 1, "love": 1, "friendship": 1,
                                        "peace": 1, "education": 1, "medicare": 1, "have money": 1, "have job": 1,
                                        "reasonable": 1, "impartial": 1,

                                        "duty": 0, "hunger": 0, "hungry": 0, "sad": 0, "not unbiased": 0,
                                        "not kill": 0, "not die": 0, "not death": 0, "not dead": 0, "not suicide": 0,
                                        "not murder": 0, "not assassinate": 0, "not killed": 0, "not harm": 0,
                                        "not danger": 0, "not torture": 0, "not attack": 0, "not hit": 0,
                                        "not violent": 0, "not hurt": 0, "not cruel": 0, "not injure": 0,
                                        "not destructive": 0, "not threaten": 0, "not aggressive": 0, "not pain": 0,
                                        "not rape": 0, "not revenge": 0, "not destroy": 0, "not weapon": 0,
                                        "not injured": 0, "not attacked": 0, "not threatened": 0, "not destroyed": 0,
                                        "not raped": 0, "not anxious": 0, "not disgusted": 0, "not displeased": 0,
                                        "not hunger": 0, "not hungry": 0, "not shame": 0, "not remorse": 0,
                                        "not fear": 0, "not anger": 0, "not angry": 0, "not outrage": 0, "not guilt": 0,
                                        "not depress": 0, "not mad": 0, "not sad": 0, "not terrified": 0, "not terror": 0,
                                        "not scared": 0, "not blind": 0, "not deaf": 0, "not disable": 0, "not phobia": 0,
                                        "not restrain": 0, "not deprive": 0, "not lie": 0, "not deceive": 0,
                                        "not renege": 0, "not betray": 0, "not betrayed": 0, "not adultery": 0,
                                        "not dishonest": 0, "not cheat": 0, "not addiction": 0, "not unconscious": 0,
                                        "not unfair": 0, "not greedy": 0, "not cheated": 0, "not illegal": 0,
                                        "not jail": 0, "not crime": 0, "not prison": 0, "not arrest": 0,
                                        "not guilty": 0, "not violate": 0, "not violated": 0, "not arrested": 0,
                                        "not war": 0, "not poverty": 0, "not slum": 0, "not favoritism": 0,
                                        "not discrimination": 0, "not bias": 0, "not hypocracy": 0, "not hypocrite": 0,
                                        "not racist": 0, "not sexist": 0, "not unjust": 0, "not pleasure": 0,
                                        "not enjoy": 0, "not smile": 0, "not joy": 0, "not happiness": 0, "not happy": 0,
                                        "not satisfaction": 0, "not satisfy": 0, "not immoral": 0, "not unethical": 0,
                                        "not evil": 0, "not inhumane": 0, "not coercion": 0,

                                        "not duty": -1, "not conscious": -1, "not survive": -1, "not ability": -1,
                                        "not intelligence": -1, "not talent": -1, "not protect": -1, "not safe": -1,
                                        "not welfare": -1, "not benefit": -1, "not freedom": -1, "not privacy": -1,
                                        "not health": -1, "not wealth": -1, "not knowledge": -1, "not love": -1,
                                        "not friendship": -1, "not peace": -1, "not education": -1, "not medicare": -1,
                                        "not justice": -1, "not honest": -1, "not upright": -1, "not equitable": -1,
                                        "not reasonable": -1, "not impartial": -1, "not moral": -1, "not fair": -1,
                                        "not ethical": -1,
                                        }

    def _init_cache_data(self):
        self.paraphrases_cache = read_json(
            data_base_path + f"cache/paraphrases_filtered_by_nli.json")
        print(f"* Paraphrases cache loaded! ({len(self.paraphrases_cache)})")

        self.comet_cache = read_json(
            data_base_path + f"cache/comet_subset.json")
        print(f"* Comet cache loaded! ({len(self.comet_cache)})")

        self.delphi_cache = read_json(
            data_base_path + f"cache/delphi_subset.json")
        print(f"* Delphi cache loaded! ({len(self.delphi_cache)})")

    def _init_utils(self):
        self.person_detector = PersonDetector()
        print(f"* {self.person_detector.__class__.__name__} loaded!")

        self.compose_parser = CompositionalityParser()
        print(f"* {self.compose_parser.__class__.__name__} loaded!")

    def _init_gold_data(self, gold_data_path=data_base_path + "cache/data_gold_labels.csv"):
        """
        Initialize gold data map and event maps
        """
        try:
            df_data_gold_label = pd.read_csv(
                gold_data_path, sep=",", low_memory=False)
        except:
            df_data_gold_label = pd.read_csv(
                gold_data_path, sep="\t", low_memory=False)

        df_data_gold_label = df_data_gold_label[~df_data_gold_label["event"].isin(
            self.events_to_exclude)]
        df_data_gold_label = df_data_gold_label.drop_duplicates(subset=[
                                                                "event"])

        gold_data_list = df_data_gold_label.to_dict(orient="records")
        self.gold_data_map = {el["event"]: el for el in gold_data_list}
        self.events = list(self.gold_data_map.keys())

        for event in self.events:
            self.paraphrase_to_event_map[event] = event
            for paraphrase in self.paraphrases_cache[event]:
                self.paraphrase_to_event_map[paraphrase] = event

        for event in self.events:
            for paraphrase in self.paraphrases_cache[event]:
                self.gold_data_map[paraphrase] = self.gold_data_map[event]

        print(f"* Events loaded! ({len(self.events)})")

        all_splits = ["train", "test", "dev"]
        all_agreement_rates = [1, 0.8, 0.6, "all", "certain", "ambiguous"]

        self.events_by_split = {split: {agreement_rate: [] for agreement_rate in all_agreement_rates} for split in
                                all_splits}
        self.events_by_agreement_rate = {agreement_rate: {split: [] for split in all_splits} for agreement_rate in
                                         all_agreement_rates}

        for split in all_splits:
            for agreement_rate in [1, 0.8, 0.6]:
                df_data_split = df_data_gold_label[df_data_gold_label["split"] == split]
                df_data_split = df_data_split[df_data_split["agreement_rate"]
                                              == agreement_rate]
                split_event = list(df_data_split["event"])
                self.events_by_split[split][agreement_rate] = split_event
                self.events_by_agreement_rate[agreement_rate][split] = split_event

            self.events_by_split[split]["all"] = self.events_by_split[split][1] + \
                self.events_by_split[split][0.6] + \
                self.events_by_split[split][0.8]
            self.events_by_split[split]["certain"] = self.events_by_split[split][1]
            self.events_by_split[split]["ambiguous"] = self.events_by_split[split][0.6] + \
                self.events_by_split[split][0.8]

            self.events_by_agreement_rate["all"][split] = self.events_by_agreement_rate[1][split] + \
                self.events_by_agreement_rate[0.6][split] + \
                self.events_by_agreement_rate[0.8][split]
            self.events_by_agreement_rate["certain"][split] = self.events_by_agreement_rate[1][split]
            self.events_by_agreement_rate["ambiguous"][split] = self.events_by_agreement_rate[0.6][split] + \
                self.events_by_agreement_rate[0.8][split]

    def _init_raw_keywords_counts_map(self, is_save=False):
        """
        Create morally salient keywords cache file.
        """
        data_path = data_base_path + f"cache/keywords.json"
        if is_save:
            self.saliency_identifier = MoralSaliencyKeywordIdentifier()
            print(f"* {self.saliency_identifier.__class__.__name__} loaded!")

            def add_instance(e, keywords_cache, all_sequences):
                e_comet_inferences = self.comet_cache[e]
                moral_saliency_keywords_count_dict = self.saliency_identifier.identify_moral_saliency_keywords(
                    e_comet_inferences)
                keywords_cache[e] = moral_saliency_keywords_count_dict
                all_sequences.append(e)
                return keywords_cache, all_sequences

            self.all_sequences = []

            for e in tqdm(self.events):
                self.raw_keywords_counts_map, self.all_sequences = add_instance(
                    e, self.raw_keywords_counts_map, self.all_sequences)

                e_compositions_raw = self.constituents_map[e]
                e_compositions = [e_compositions_raw[t] for t in self.all_event_types[1:] if
                                  e_compositions_raw[t] != None]

                for ce in e_compositions:
                    self.raw_keywords_counts_map, self.all_sequences = add_instance(
                        ce, self.raw_keywords_counts_map, self.all_sequences)

                for p in self.paraphrases_cache[e]:
                    self.raw_keywords_counts_map, self.all_sequences = add_instance(
                        p, self.raw_keywords_counts_map, self.all_sequences)

                    p_compositions_raw = self.compose_parser.get_parsed_event(
                        p)
                    p_compositions = [p_compositions_raw[t]
                                      for t in self.all_event_types[1:] if p_compositions_raw[t] != None]
                    for cp in p_compositions:
                        self.raw_keywords_counts_map, self.all_sequences = add_instance(
                            cp, self.raw_keywords_counts_map, self.all_sequences)

            save_json(data_path, self.raw_keywords_counts_map)
            print(
                f"* Moral Saliency Keywords map loaded! ({len(self.raw_keywords_counts_map)})")
        else:
            self.raw_keywords_counts_map = read_json(data_path)
            self.all_sequences = list(self.raw_keywords_counts_map.keys())

            print(
                f"* Moral Saliency Keywords map loaded! ({len(self.raw_keywords_counts_map)})")

    def _init_all_keywords(self):
        _all_keywords = []
        for k in self.top_level_keywords_map:
            _all_keywords += self.top_level_keywords_map[k]

        all_keywords = []
        for k in _all_keywords:
            if " " not in k:
                all_keywords.append("not " + k)

        return _all_keywords + all_keywords

    def _get_event_top_level_keyword_counts(self, event, top_level_keyword):
        event_keywords_counts = self.raw_keywords_counts_map[event]
        selected_keywords = self.top_level_keywords_map[top_level_keyword]
        selected_keywords_counts = {k: event_keywords_counts[k] if k in event_keywords_counts else 0
                                    for k in selected_keywords}
        return selected_keywords_counts

    def _get_top_level_keyword_counts_map(self, top_level_keyword):
        top_level_keywords_counts_map = {}
        for event in self.raw_keywords_counts_map.keys():
            selected_keywords_counts = self._get_event_top_level_keyword_counts(
                event, top_level_keyword)
            top_level_keywords_counts_map[event] = sum(
                [selected_keywords_counts[k] for k in selected_keywords_counts])
        return top_level_keywords_counts_map

    def _init_top_level_keywords_counts_maps(self):
        for k in self.top_level_keywords_map:
            self.top_level_keywords_counts_map[k] = self._get_top_level_keyword_counts_map(
                k)
        print(
            f"* Top Level Keywords Counts Map Loaded! ({len(self.top_level_keywords_counts_map)})")

    def _init_binary_keywords_counts_map(self):
        for event in self.raw_keywords_counts_map:
            self.binary_keywords_counts_map[event] = {}
            self.binary_keywords_counts_map[event][1] = sum([self.raw_keywords_counts_map[event][k]
                                                             for k in self.raw_keywords_counts_map[event]
                                                             if (k in self.all_keywords and
                                                                 self.raw_keywords_binary_map[k] == 1)])
            self.binary_keywords_counts_map[event][-1] = sum([self.raw_keywords_counts_map[event][k]
                                                             for k in self.raw_keywords_counts_map[event]
                                                             if (k in self.all_keywords and
                                                                 self.raw_keywords_binary_map[k] == -1)])

        print(
            f"* Binary Keywords Counts Map Loaded! ({len(self.binary_keywords_counts_map)})")

    def _init_constituents_map(self, is_save=False):
        data_path = data_base_path + f"cache/constituents.json"
        if is_save:
            for event in tqdm(self.events):
                self.constituents_map[event] = self.compose_parser.get_parsed_event(
                    event)
                for event_paraphrase in self.paraphrases_cache[event]:
                    self.constituents_map[event_paraphrase] = self.compose_parser.get_parsed_event(
                        event_paraphrase)
            save_json(data_path, self.constituents_map)
            print(f"* Constituents map loaded! ({len(self.constituents_map)})")
        else:
            self.constituents_map = read_json(data_path)
            print(f"* Constituents map loaded! ({len(self.constituents_map)})")

    def _init_average_paraphrases_map(self, is_save=False):
        """
        Compile average paraphrases data
        """
        data_path = data_base_path + f"cache/average_paraphrases_filtered_by_nli.json"
        if is_save:
            compiled_data = {"root_event": [], "type": [],
                             "agreement_rate": [], "class_label_3_way": []}

            for e_t in self.all_event_types:
                compiled_data[e_t] = []
                compiled_data[e_t + "_prob_0"] = []
                compiled_data[e_t + "_prob_1"] = []
                compiled_data[e_t + "_prob_minus_1"] = []
                compiled_data[e_t + "_class_pred_3_way"] = []
                compiled_data[e_t + "_text_pred"] = []
                compiled_data[e_t + "_kill_count"] = []
                compiled_data[e_t + "_save_life_count"] = []

            for e in tqdm(self.events):
                e_compositions = self.constituents_map[e]
                e_compositions["event"] = e

                compiled_data["root_event"].append(e)
                compiled_data["type"].append("original")
                compiled_data["agreement_rate"].append(
                    self.gold_data_map[e]["agreement_rate"])
                compiled_data["class_label_3_way"].append(
                    self.gold_data_map[e]["class_label"])

                for e_t in self.all_event_types:
                    e_e = e_compositions[e_t]
                    if e_e != None:
                        compiled_data[e_t].append(e_e)
                        compiled_data[e_t +
                                      "_prob_0"].append(self.delphi_cache[e_e]["prob_0"])
                        compiled_data[e_t +
                                      "_prob_1"].append(self.delphi_cache[e_e]["prob_1"])
                        compiled_data[e_t + "_prob_minus_1"].append(
                            self.delphi_cache[e_e]["prob_minus_1"])
                        compiled_data[e_t + "_class_pred_3_way"].append(
                            self.delphi_cache[e_e]["class_label"])
                        compiled_data[e_t + "_text_pred"].append(
                            self.delphi_cache[e_e]["text_label"])
                        compiled_data[e_t + "_kill_count"].append(
                            self.top_level_keywords_counts_map["kill"][e_e])
                        compiled_data[e_t + "_save_life_count"].append(
                            self.top_level_keywords_counts_map["save life"][e_e])
                    else:
                        compiled_data[e_t].append("")
                        compiled_data[e_t + "_prob_0"].append(None)
                        compiled_data[e_t + "_prob_1"].append(None)
                        compiled_data[e_t + "_prob_minus_1"].append(None)
                        compiled_data[e_t + "_class_pred_3_way"].append(None)
                        compiled_data[e_t + "_text_pred"].append(None)
                        compiled_data[e_t + "_kill_count"].append(None)
                        compiled_data[e_t + "_save_life_count"].append(None)

                for p in self.paraphrases_cache[e]:
                    p_compositions = self.constituents_map[p]
                    p_compositions["event"] = p

                    compiled_data["root_event"].append(e)
                    compiled_data["type"].append("paraphrase")
                    compiled_data["agreement_rate"].append(
                        self.gold_data_map[e]["agreement_rate"])
                    compiled_data["class_label_3_way"].append(
                        self.gold_data_map[e]["class_label"])

                    for e_t in self.all_event_types:
                        e_e = p_compositions[e_t]
                        if e_e != None:
                            compiled_data[e_t].append(e_e)
                            compiled_data[e_t + "_prob_0"].append(
                                self.delphi_cache[e_e]["prob_0"])
                            compiled_data[e_t + "_prob_1"].append(
                                self.delphi_cache[e_e]["prob_1"])
                            compiled_data[e_t + "_prob_minus_1"].append(
                                self.delphi_cache[e_e]["prob_minus_1"])
                            compiled_data[e_t + "_class_pred_3_way"].append(
                                self.delphi_cache[e_e]["class_label"])
                            compiled_data[e_t + "_text_pred"].append(
                                self.delphi_cache[e_e]["text_label"])
                            compiled_data[e_t + "_kill_count"].append(
                                self.top_level_keywords_counts_map["kill"][e_e])
                            compiled_data[e_t + "_save_life_count"].append(
                                self.top_level_keywords_counts_map["save life"][e_e])
                        else:
                            compiled_data[e_t].append("")
                            compiled_data[e_t + "_prob_0"].append(None)
                            compiled_data[e_t + "_prob_1"].append(None)
                            compiled_data[e_t + "_prob_minus_1"].append(None)
                            compiled_data[e_t +
                                          "_class_pred_3_way"].append(None)
                            compiled_data[e_t + "_text_pred"].append(None)
                            compiled_data[e_t + "_kill_count"].append(None)
                            compiled_data[e_t +
                                          "_save_life_count"].append(None)

            df_data = pd.DataFrame(compiled_data)

            """
            handle average paraphrases
            """
            for e in tqdm(self.events):
                df_data_e_paraphrase = df_data[df_data["type"] == "paraphrase"]
                df_data_e_paraphrase = df_data_e_paraphrase[df_data_e_paraphrase["root_event"] == e]

                df_data_e_paraphrase = df_data_e_paraphrase[
                    ~df_data_e_paraphrase["event_text_pred"].str.lower().str.startswith("yes,")]
                df_data_e_paraphrase = df_data_e_paraphrase[
                    ~df_data_e_paraphrase["event_text_pred"].str.lower().str.startswith("no,")]

                # Get average scores cross paraphrases
                avg_prob_minus_1 = df_data_e_paraphrase["event_prob_minus_1"].mean(
                )
                avg_prob_0 = df_data_e_paraphrase["event_prob_0"].mean()
                avg_prob_1 = df_data_e_paraphrase["event_prob_1"].mean()
                avg_class_label = get_class_label_from_probs(
                    avg_prob_minus_1, avg_prob_0, avg_prob_1)

                self.average_paraphrases_map[e] = {"prob_minus_1": avg_prob_minus_1,
                                                   "prob_0": avg_prob_0,
                                                   "prob_1": avg_prob_1,
                                                   "class_label": avg_class_label}

            save_json(data_path, self.average_paraphrases_map)
            print(
                f"* Average paraphrases map loaded! ({len(self.average_paraphrases_map)})")

        else:
            self.average_paraphrases_map = read_json(data_path)
            print(
                f"* Average paraphrases map loaded! ({len(self.average_paraphrases_map)})")

    def print_accuracy(self, function, class_label_preds, class_label_targets, num_way):
        is_correct = self.get_class_label_correct(
            class_label_preds, class_label_targets, num_way)
        acc = self.get_accuracy(is_correct)
        print("{}: {:5.2f}%".format(function.__name__, acc * 100))

    def get_incorrect_events(self, is_correct, events):
        return [events[i] for i in range(len(is_correct)) if is_correct[i] == 0]

    def get_accuracy(self, is_correct):
        acc = sum(is_correct) / len(is_correct)
        return acc

    def get_class_label_correct(self, class_label_preds, class_label_targets, num_way):
        if num_way == 2:
            class_label_preds = [self._3_way_to_2_way_class_label(
                pred) for pred in class_label_preds]
            class_label_targets = [self._3_way_to_2_way_class_label(
                target) for target in class_label_targets]
        return [int(class_label_preds[i] == class_label_targets[i]) for i in range(len(class_label_preds))]

    def _3_way_to_2_way_class_label(self, class_label):
        return 1 if class_label in [0, 1] else -1


if __name__ == "__main__":
    delphi_ruler = BaseRuler()

    for e in delphi_ruler.events[:2]:
        print("-" * 20)
        print(e)
        e_paraphrases = delphi_ruler.paraphrases_cache[e]
        for i, e_p in enumerate(e_paraphrases):
            print(e_p)
            e_p_c = delphi_ruler.constituents_map[e_p]
            print(e_p_c)
