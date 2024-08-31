import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools as it
from scipy.stats import ttest_ind, chisquare

sys.path.append(os.getcwd())

from scripts.collective_reasoning.ITW.Ruler import *

rng = np.random.default_rng()


class EnsembleRuler(Ruler):
    def __init__(self, mode="hybrid"):
        super().__init__()

        # self.vertex_weights = {"delphi": 4,
        #                        "rule": 3,
        #                        "constituent": 1}  # up-weight delphi predictions on event and paraphrases

        # self.vertex_weights = {"delphi": 3,
        #                        "rule": 1,
        #                        "constituent": 1}  # up-weight delphi predictions on event and paraphrases

        self.vertex_weights = {"delphi": 8,
                               "rule": 3,
                               "constituent": 1}  # up-weight delphi predictions on event and paraphrases

        print(self.vertex_weights)

        self.kill_entailed_moral_concept_pairs = [["do_not_kill", "do_not_cause_pain_physical"],
                                                  ["do_not_kill", "do_not_disable"],
                                                  ["do_not_kill", "obey_the_law"],
                                                  ["do_not_disable", "do_not_cause_pain_physical"]]

        self.moral_saliency_entailed_moral_concepts = ["moral_valency", "moral_impartiality", "do_not_kill",
                                                       "do_not_cause_pain_physical", "do_not_cause_pain_mental",
                                                       "do_not_disable", "do_not_deprive_of_freedom",
                                                       "do_not_deprive_of_pleasure", "do_not_deceive",
                                                       "keep_your_promises", "do_not_cheat", "obey_the_law",
                                                       "do_your_duty"]

        self.rule_list = []

        if mode == "bottom":
            self.rule_list += self.rule_sets["base"]
        elif mode == "top":
            for moral_concept in self.rule_sets:
                if moral_concept != "base":
                    self.rule_list += self.rule_sets[moral_concept]
        else:
            for moral_concept in self.rule_sets:
                self.rule_list += self.rule_sets[moral_concept]

    """HELPERS begin"""

    def _get_maj_vote_class_label_preds(self, events, class_label_preds, column_selected):
        """
        Get the majority vote label among the indicated list of class labels
        """
        maj_vote_class_label_preds = []
        for event in events:
            class_labels_preds = class_label_preds[event][column_selected]
            if len(class_labels_preds) == 0: # if the judgment pool has no nodes, assign 0 to the label
                maj_vote_class_label_pred = 0
            else:
                maj_vote_class_label_pred = max(set(class_labels_preds), key=class_labels_preds.count)
            class_label_preds[event][column_selected + "_maj_vote"] = maj_vote_class_label_pred
            maj_vote_class_label_preds.append(maj_vote_class_label_pred)
        return maj_vote_class_label_preds, class_label_preds

    def _get_combinatory_pairs(self, list_of_elements):
        combinatory_pairs = []
        for a, b in it.combinations(list_of_elements, 2):
            combinatory_pairs.append([a, b])
        return combinatory_pairs

    def _get_class_label_pred(self, class_label_preds, event, idx):
        event_data = class_label_preds[event]
        if not event_data["is_affected"][idx]:
            class_label_pred = None
        else:
            class_label_pred = event_data["class_label_preds"][idx]
        return class_label_pred

    def wcnf_header(self, v, c, w):
        return "p wcnf " + str(v) + " " + str(c) + " " + str(w)

    def solve_wcnf(self, wcnf_file, output_file):
        with open(output_file, 'w') as f:
            subprocess.run([base_path + 'open-wbo/open-wbo', '-cpu-lim=3600',
                            os.path.abspath(wcnf_file)], stdout=f)

    """HELPERS end"""

    def optimize(self, vertices, edges):
        """
        Use defined vertices and edges to define a constrained optimization problem
        """
        v = len(vertices)
        c = len(vertices) + len(edges)
        max_w = len(vertices) * 100
        header = self.wcnf_header(v, c, max_w)

        output_folder = "results/collective_reasoning/outputs/itw/"
        wcnf_file = output_folder + "wcnf.txt"
        with open(wcnf_file, 'w') as f:
            f.write(header)
            f.write('\n')

            # Soft constraints: include (or not) a given rule (weight w)
            for v in vertices:
                f.write(str(v["w"]) + " " + str(v["v"]) + " 0")
                f.write('\n')

            # Soft constraints: do not include two contradicting rules (weight 1)
            for e in edges:
                f.write(str(1) + " -" + str(e["v0"]) + " -" + str(e["v1"]) + " 0")
                f.write('\n')

        res_file = output_folder + "wcnf_output.txt"
        self.solve_wcnf(wcnf_file, res_file)
        return res_file

    def optimize_results(self, event, class_label_preds, res_file, vertex_id2idx):
        var_assignment = []
        with open(res_file, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    for e in line[2:].split():
                        var_assignment.append(int(e))

        # print(len(vertices), vertices, len(var_assignment), var_assignment, len(class_label_preds[event]["rule_id"]),
        #       len([i for i in class_label_preds[event]["is_affected"] if i]))
        selected_class_label_preds = []
        max_sat_raw_vertex_id_assignment = []
        max_sat_selected_idx = []
        for a in var_assignment:
            vertex_id = abs(a)
            idx = vertex_id2idx[vertex_id]
            if a < 0:
                max_sat_raw_vertex_id_assignment.append(vertex_id * -1)
            else:
                selected_class_label_preds.append(class_label_preds[event]["class_label_preds"][idx])
                max_sat_raw_vertex_id_assignment.append(vertex_id)
                max_sat_selected_idx.append(idx)

        class_label_preds[event]["max_sat_class_label_preds"] = selected_class_label_preds
        class_label_preds[event]["max_sat_raw_vertex_id_assignment"] = max_sat_raw_vertex_id_assignment
        class_label_preds[event]["max_sat_selected_idx"] = max_sat_selected_idx

        return class_label_preds

    def is_identity_constraint_violated(self, rule_0_class_label_pred, rule_1_class_label_pred):
        """
        Check if the identity constraint is violated between two rules under the same moral concept.
        e.g., if two rules both fall under the moral concept "kill", then they should not contradict with each other
        """
        if rule_0_class_label_pred in [-1] and rule_1_class_label_pred in [0, 1]:
            return True
        elif rule_0_class_label_pred in [0, 1] and rule_1_class_label_pred in [-1]:
            return True
        return False

    def is_moral_saliency_entailment_constraint_violated(self, rule_0_class_label_pred, rule_1_class_label_pred):
        """
        rule_0 entails moral_saliency,
        e.g., if a situation has killing implication, it must have moral saliency as well
        """
        if rule_0_class_label_pred in [1, -1] and rule_1_class_label_pred in [0]:
            return True
        return False

    def is_moral_valency_entailment_constraint_violated(self, rule_0_class_label_pred, rule_1_class_label_pred):
        """
        rule_0 entails moral_valency,
        e.g., if a situation has killing implication, it must have moral saliency as well
        Cases:
        if rule_0 is 0, moral_valency can be anything (0, 1, -1, because other rules may lead to moral valency)
        if rule_0 is 1, moral_valency should be 1 or 0 (-1 is violation)
        if rule_0 is -1, moral_valency should be -1 (0, 1 are violations)
        """
        if rule_0_class_label_pred in [-1] and rule_1_class_label_pred in [0, 1]:
            return True
        elif rule_0_class_label_pred in [1] and rule_1_class_label_pred in [-1]:
            return True
        return False

    def is_kill_entailment_constraint_violated(self, rule_0_class_label_pred, rule_1_class_label_pred):
        """
        rule_0 entails rule_1,
        e.g., kill entails harm, if rule_0 indicates "kill", rule_1 must indicates "harm"
        """
        if rule_0_class_label_pred in [-1] and rule_1_class_label_pred in [0, 1]:
            return True
        return False

    def get_vertices(self, event, class_label_preds):
        event_data = class_label_preds[event]
        vertices = []
        vertex_id2idx = {}
        vertex_id = 1  # the vertex_id when feed into wcnf solver, has to be greater than 0
        for idx, rule_id in enumerate(event_data["rule_id"]):
            if event_data["is_affected"][idx]:
                if rule_id in ["delphi"]:
                    weight = self.vertex_weights["delphi"]
                elif "delphi_" in rule_id:
                    weight = self.vertex_weights["constituent"]
                else:
                    weight = self.vertex_weights["rule"]

                v = {"v": vertex_id, "v_idx": idx, "v_rule": rule_id, "w": weight}
                vertices.append(v)
                vertex_id2idx[vertex_id] = idx
                vertex_id += 1
        return vertices, vertex_id2idx, {v: k for k, v in vertex_id2idx.items()}

    def get_identity_violation_edges(self, event, class_label_preds, moral_concept2idxs, idx2vertex_id, idx2rule_id,
                                     edges):
        """
        <TOP-DOWN> add identity constraints
        <BOTTOM-UP> add identity constraints within delphi preds
        """
        for moral_concept in self.rule_sets:
            idxs_with_moral_concept = moral_concept2idxs[moral_concept]
            idx_pairs = self._get_combinatory_pairs(idxs_with_moral_concept)

            for idx0, idx1 in idx_pairs:
                rule_0_class_label_pred = self._get_class_label_pred(class_label_preds, event, idx0)
                rule_1_class_label_pred = self._get_class_label_pred(class_label_preds, event, idx1)
                is_identity_constraint_violated = self.is_identity_constraint_violated(rule_0_class_label_pred,
                                                                                       rule_1_class_label_pred)
                if is_identity_constraint_violated:
                    edges.append({"v0": idx2vertex_id[idx0], "v0_idx": idx0, "v0_rule_id": idx2rule_id[idx0],
                                  "v1": idx2vertex_id[idx1], "v1_idx": idx1, "v1_rule_id": idx2rule_id[idx1],
                                  "type": "identity"})
        return edges

    def get_moral_saliency_violation_edges(self, event, class_label_preds, moral_concept2idxs, idx2vertex_id,
                                           idx2rule_id, edges):
        """
        <TOP-DOWN> (moral saliency) entailment constraints
        """
        for c0 in self.moral_saliency_entailed_moral_concepts:
            c0_idxs = moral_concept2idxs[c0]
            c1_idxs = moral_concept2idxs["moral_saliency"]

            for idx0 in c0_idxs:
                for idx1 in c1_idxs:
                    rule_0_class_label_pred = self._get_class_label_pred(class_label_preds, event, idx0)
                    rule_1_class_label_pred = self._get_class_label_pred(class_label_preds, event, idx1)
                    is_moral_saliency_entailment_constraint_violated = \
                        self.is_moral_saliency_entailment_constraint_violated(rule_0_class_label_pred,
                                                                              rule_1_class_label_pred)
                    if is_moral_saliency_entailment_constraint_violated:
                        edges.append({"v0": idx2vertex_id[idx0], "v0_idx": idx0, "v0_rule_id": idx2rule_id[idx0],
                                      "v1": idx2vertex_id[idx1], "v1_idx": idx1, "v1_rule_id": idx2rule_id[idx1],
                                      "type": "moral_saliency"})
        return edges

    def get_moral_valency_violation_edges(self, event, class_label_preds, moral_concept2idxs, idx2vertex_id,
                                          idx2rule_id, edges):
        """
        <TOP-DOWN> (moral valency) entailment constraints
        """
        for c0 in self.moral_saliency_entailed_moral_concepts[1:]:
            c0_idxs = moral_concept2idxs[c0]
            c1_idxs = moral_concept2idxs["moral_valency"]

            for idx0 in c0_idxs:
                for idx1 in c1_idxs:
                    rule_0_class_label_pred = self._get_class_label_pred(class_label_preds, event, idx0)
                    rule_1_class_label_pred = self._get_class_label_pred(class_label_preds, event, idx1)
                    is_moral_valency_entailment_constraint_violated = \
                        self.is_moral_valency_entailment_constraint_violated(rule_0_class_label_pred,
                                                                             rule_1_class_label_pred)
                    if is_moral_valency_entailment_constraint_violated:
                        edges.append({"v0": idx2vertex_id[idx0], "v0_idx": idx0, "v0_rule_id": idx2rule_id[idx0],
                                      "v1": idx2vertex_id[idx1], "v1_idx": idx1, "v1_rule_id": idx2rule_id[idx1],
                                      "type": "moral_valency"})
        return edges

    def get_kill_violation_edges(self, event, class_label_preds, moral_concept2idxs, idx2vertex_id, idx2rule_id, edges):
        """
        <TOP-DOWN> (kill) entailment constraints
        """
        for c0, c1 in self.kill_entailed_moral_concept_pairs:
            c0_idxs = moral_concept2idxs[c0]
            c1_idxs = moral_concept2idxs[c1]

            for idx0 in c0_idxs:
                for idx1 in c1_idxs:
                    rule_0_class_label_pred = self._get_class_label_pred(class_label_preds, event, idx0)
                    rule_1_class_label_pred = self._get_class_label_pred(class_label_preds, event, idx1)
                    is_kill_entailment_constraint_violated = self.is_kill_entailment_constraint_violated(
                        rule_0_class_label_pred, rule_1_class_label_pred)
                    if is_kill_entailment_constraint_violated:
                        edges.append({"v0": idx2vertex_id[idx0], "v0_idx": idx0, "v0_rule_id": idx2rule_id[idx0],
                                      "v1": idx2vertex_id[idx1], "v1_idx": idx1, "v1_rule_id": idx2rule_id[idx1],
                                      "type": "kill"})
        return edges

    def get_bottom_up_top_down_violation_edges(self, event, class_label_preds, moral_concept2idxs, idx2vertex_id,
                                               idx2rule_id, edges):
        """
        <BOTTOM-UP> delphi preds vs. top-down rules
        """
        all_bottom_up_moral_concepts = [c for c in self.rule_sets if c not in ["base"]]
        c0_idxs = moral_concept2idxs["base"]
        for c1 in all_bottom_up_moral_concepts:
            c1_idxs = moral_concept2idxs[c1]
            for idx0 in c0_idxs:
                for idx1 in c1_idxs:
                    rule_0_class_label_pred = self._get_class_label_pred(class_label_preds, event, idx0)
                    rule_1_class_label_pred = self._get_class_label_pred(class_label_preds, event, idx1)
                    is_identity_constraint_violated = self.is_identity_constraint_violated(rule_0_class_label_pred,
                                                                                           rule_1_class_label_pred)
                    if is_identity_constraint_violated:
                        edges.append({"v0": idx2vertex_id[idx0], "v0_idx": idx0, "v0_rule_id": idx2rule_id[idx0],
                                      "v1": idx2vertex_id[idx1], "v1_idx": idx1, "v1_rule_id": idx2rule_id[idx1],
                                      "type": "bottom_up_top_down"})
        return edges

    def preds2wcnf(self, event, class_label_preds):
        """
        idx: index in the list of preds
        rule_id: rule_name, e.g., delphi, s_moral_saliency
        moral_concept: top-level keywords of "rule_sets", e.g., moral_saliency
        """
        idx2rule_id = {idx: rule_id for idx, rule_id in enumerate(class_label_preds[event]["rule_id"])}
        # rule_id2idx = {v: k for k, v in idx2rule_id.items()}

        moral_concept2idxs = {moral_concept: [idx for idx, rule_id in enumerate(class_label_preds[event]["rule_id"])
                                              if rule_id in self.rule_sets[moral_concept] and class_label_preds[event][
                                                  "is_affected"][idx]] for moral_concept in self.rule_sets}

        """***** ADD VERTICES *****"""
        vertices, vertex_id2idx, idx2vertex_id = self.get_vertices(event, class_label_preds)
        class_label_preds[event]["vertices"] = vertices

        """***** ADD EDGES *****"""
        edges = self.get_identity_violation_edges(event, class_label_preds, moral_concept2idxs, idx2vertex_id,
                                                  idx2rule_id, [])
        edges = self.get_moral_saliency_violation_edges(event, class_label_preds, moral_concept2idxs, idx2vertex_id,
                                                        idx2rule_id, edges)
        edges = self.get_moral_valency_violation_edges(event, class_label_preds, moral_concept2idxs, idx2vertex_id,
                                                       idx2rule_id, edges)
        edges = self.get_kill_violation_edges(event, class_label_preds, moral_concept2idxs, idx2vertex_id, idx2rule_id,
                                              edges)
        edges = self.get_bottom_up_top_down_violation_edges(event, class_label_preds, moral_concept2idxs, idx2vertex_id,
                                                            idx2rule_id, edges)
        class_label_preds[event]["edges"] = edges

        """***** Constrained Optimization *****"""
        res_file = self.optimize(vertices, edges)

        """***** Consolidate Optimized Results *****"""
        class_label_preds = self.optimize_results(event, class_label_preds, res_file, vertex_id2idx)

        return class_label_preds

    def _print_results(self, class_label_preds, splits):
        for split in splits:
            print("-" * 20, split, "-" * 20)
            for num_way in [2, 3]:
                print("*" * 10, f"{num_way}-way classification", "*" * 10)
                for agreement_rate in ["all", 1, "ambiguous"]:  # [1, 0.8, 0.6, "ambiguous", "all"]
                    # print("* agreement_rate:", agreement_rate, "*")
                    events_split = self.events_by_agreement_rate[agreement_rate][split]

                    maj_vote_class_label_preds, class_label_preds = \
                        self._get_maj_vote_class_label_preds(events_split, class_label_preds, "affected_class_label_preds")

                    max_sat_maj_vote_class_label_preds, class_label_preds = \
                        self._get_maj_vote_class_label_preds(events_split, class_label_preds, "max_sat_class_label_preds")

                    delphi_class_label_preds = [self.delphi_cache[event]["class_label"] for event in events_split]
                    class_label_targets = [class_label_preds[event]["class_label_targets"][0] for event in events_split]

                    maj_vote_is_correct = self.get_class_label_correct(maj_vote_class_label_preds, class_label_targets, num_way)
                    maj_vote_acc = self.get_accuracy(maj_vote_is_correct)

                    max_sat_maj_vote_is_correct = self.get_class_label_correct(max_sat_maj_vote_class_label_preds,
                                                                               class_label_targets, num_way)
                    max_sat_acc = self.get_accuracy(max_sat_maj_vote_is_correct)

                    delphi_is_correct = self.get_class_label_correct(delphi_class_label_preds, class_label_targets, num_way)
                    delphi_acc = self.get_accuracy(delphi_is_correct)
                    # print(f"({num_way}-way) Raw Maj vote: {:5.1f} | MAX-SAT: {:5.1f} | Delphi: {:5.1f}".format(maj_vote_acc * 100,
                    #                                                                                   max_sat_acc * 100,
                    #                                                                                   delphi_acc * 100))
                    print("{:5.1f}\t{:5.1f}\t{:5.1f}".format(
                        delphi_acc * 100,
                        maj_vote_acc * 100,
                        max_sat_acc * 100))

                    # res = ttest_ind(delphi_is_correct,
                    #                 max_sat_maj_vote_is_correct,
                    #                 permutations=1000,
                    #                 random_state=rng)
                    # print(f"({num_way}-way) MAX-SAT p-value w/ Delphi:", "%0.10f" % res[1])


    def get_rule_labels(self, splits, agreement_rate=1, num_way=2, is_print=False):
        # events = self.events
        events = []
        for split in splits:
            events += self.events_by_split[split]["all"]
        class_label_targets = [self.gold_data_map[event]["class_label"] for event in events]
        class_label_preds = {event: {"class_label_preds": [], "affected_class_label_preds": [], "is_affected": [],
                                     "rule_id": [], "paraphrase": [], "class_label_targets": []} for event in events}

        rule_list = self.rule_list
        for i, event in enumerate(events):
            self._reset_existing_constituents_cache()
            # Apply rules to the original event
            for rule_func_name in rule_list:
                class_label_preds = self.apply_rule(class_label_preds, rule_func_name, event,
                                                    class_label_target=class_label_targets[i],
                                                    paraphrase=None)
                # Apply rules to paraphrases
                for paraphrase in self.paraphrases_cache[event]:
                    class_label_preds = self.apply_rule(class_label_preds, rule_func_name, event,
                                                        class_label_target=class_label_targets[i],
                                                        paraphrase=paraphrase)

        if is_print:
            _, class_label_preds = self._get_maj_vote_class_label_preds(events, class_label_preds,
                                                                        "affected_class_label_preds")

            ######## Get maj vote accuracy ########
            print("-" * 20, "maj vote", "-" * 20)
            for split in ["train", "dev"]:
                events_split = self.events_by_agreement_rate[agreement_rate][split]

                class_label_preds_split = []
                class_label_targets_split = []
                for event in events_split:
                    class_label_preds_split.append(class_label_preds[event]["affected_class_label_preds_maj_vote"])
                    class_label_targets_split.append(class_label_preds[event]["class_label_targets"][0])

                is_correct = self.get_class_label_correct(class_label_preds_split, class_label_targets_split, num_way)
                acc = self.get_accuracy(is_correct)
                print(split, ": {:5.2f}%".format(acc * 100))

            ######## Get accuracy per each rule (for affected events) ########
            for rule_id_selected in rule_list:
                print("<" * 10, rule_id_selected, ">" * 10)
                for split in ["train", "dev"]:
                    events_split = self.events_by_agreement_rate[agreement_rate][split]

                    class_label_preds_split = []
                    class_label_targets_split = []
                    for event in events_split:
                        for i, rule_id in enumerate(class_label_preds[event]["rule_id"]):
                            if rule_id == rule_id_selected:
                                if class_label_preds[event]["is_affected"][i]:
                                    class_label_preds_split.append(class_label_preds[event]["class_label_preds"][i])
                                    class_label_targets_split.append(class_label_preds[event]["class_label_targets"][i])

                    is_correct = self.get_class_label_correct(class_label_preds_split, class_label_targets_split,
                                                              num_way)
                    if len(is_correct) != 0:
                        acc = self.get_accuracy(is_correct)
                        print(split, ": {:5.2f}%".format(acc * 100), len(class_label_preds_split))
        return class_label_preds

    def main_preds2wcnf(self, class_label_preds, splits, is_print=True):
        w = self.vertex_weights["delphi"]

        for event in tqdm(class_label_preds):
            class_label_preds = self.preds2wcnf(event, class_label_preds)
        # save_json(f"results/collective_reasoning/outputs/class_label_preds_w_{w}_remove_dup_downweight_constituent.json", class_label_preds)

        # class_label_preds = read_json(f"results/collective_reasoning/outputs/class_label_preds_w_{w}.json")

        if is_print:
            self._print_results(class_label_preds, splits)


if __name__ == "__main__":

    # for k in class_label_preds[event]:
    #     print(k, ":", len(class_label_preds[event][k]))
    #
    # print("class_label_pred selected", ":", len([i for i in class_label_preds[event]["class_label_preds"] if i != None]))
    # print("is_affected selected", ":", len([i for i in class_label_preds[event]["is_affected"] if i]))

    # print("-" * 20)
    # print([i for i, is_affected in enumerate(class_label_preds[event]["is_affected"]) if is_affected])
    # print(max_sat_selected_idx)
    # print(max_sat_raw_vertex_id_assignment)

    # for agreement_rate in ["certain", "ambiguous", "all"]:
    # class_label_preds = delphi_ruler.get_rule_labels(agreement_rate=1)
    # print(delphi_ruler.main.__name__)
    # print(delphi_ruler.comet_cache["silently farting in a crowded room"])

    # delphi_ruler.main_top_down_only(class_label_preds)
    print("has valency constraint")
    for mode in ["hybrid"]: # hybrid", "top", "bottom", "top", "bottom"
        delphi_ruler = EnsembleRuler(mode)
        print("-" * 50)
        print(mode)
        splits = ["train", "dev"]  # "train",
        class_label_preds = delphi_ruler.get_rule_labels(splits=splits)
        delphi_ruler.main_preds2wcnf(class_label_preds, splits=splits)

