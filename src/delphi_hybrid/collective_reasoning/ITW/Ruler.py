import os
import sys

sys.path.append(os.getcwd())
from src.delphi_hybrid.collective_reasoning.ITW.BaseRuler import *


class Ruler(BaseRuler):
    """
    Class Ruler that implements different instantiations of rules under theoretically motivated moral concepts
    """

    def __init__(self):
        super().__init__()

        self.rule_sets = {"base": ["delphi", "delphi_main_event", "delphi_main_event_main_clause",
                                   "delphi_main_event_relative_clause", "delphi_adjunct_event"],

                          "moral_saliency": ["e_moral_saliency", "s_moral_saliency"],

                          "moral_valency": ["e_moral_valency_0", "e_moral_valency_1", "e_moral_valency_2",
                                            "s_moral_valency_0", "s_moral_valency_1", "s_moral_valency_2",
                                            "s_moral_valency_3"],

                          "moral_impartiality": ["e_moral_impartiality_0", "e_moral_impartiality_1",
                                                 "e_moral_impartiality_2", "s_moral_impartiality_0",
                                                 "s_moral_impartiality_1", "s_moral_impartiality_2"],

                          "do_not_kill": ["e_kill_vs_conscious", "s_kill_vs_conscious_0", "s_kill_vs_conscious_1",
                                          "s_kill_vs_conscious_2", "s_kill_vs_conscious_3"],

                          "do_not_cause_pain_physical": ["e_do_not_cause_pain_physical",
                                                         "s_do_not_cause_pain_physical_0",
                                                         "s_do_not_cause_pain_physical_1"],

                          "do_not_cause_pain_mental": ["e_do_not_cause_pain_mental",
                                                       "s_do_not_cause_pain_mental_0",
                                                       "s_do_not_cause_pain_mental_1"],

                          "do_not_disable": ["e_do_not_disable_0",
                                             "e_do_not_disable_1",
                                             "s_do_not_disable_0",
                                             "s_do_not_disable_1",
                                             "s_do_not_disable_2",
                                             "s_do_not_disable_3"],

                          "do_not_deprive_of_freedom": ["e_do_not_deprive_of_freedom_0",
                                                        "e_do_not_deprive_of_freedom_1",
                                                        "s_do_not_deprive_of_freedom_0",
                                                        "s_do_not_deprive_of_freedom_1"],

                          "do_not_deprive_of_pleasure": ["e_do_not_deprive_of_pleasure_0",
                                                         "e_do_not_deprive_of_pleasure_1",
                                                         "s_do_not_deprive_of_pleasure_0",
                                                         "s_do_not_deprive_of_pleasure_1"],

                          "do_not_deceive": ["e_do_not_deceive", "s_do_not_deceive"],

                          "keep_your_promises": ["e_keep_your_promises", "s_keep_your_promises"],

                          "do_not_cheat": ["e_sex", "s_sex_0", "s_sex_1", "e_no_sex", "s_no_sex",
                                           "e_do_not_cheat", "s_do_not_cheat"],

                          "obey_the_law": ["e_illegal", "s_illegal"],

                          "do_your_duty": ["e_do_your_duty", "s_do_your_duty"],
                          }

        # if a constituents is already added, then skip
        self.existing_constituents_cache = []

    def _reset_existing_constituents_cache(self):
        self.existing_constituents_cache = []

    def apply_rule(self, class_label_preds, rule_func_name, event, class_label_target, paraphrase):
        """
        Apply the given rule function to the given event
        """
        rule_func = getattr(self, rule_func_name)
        if paraphrase != None:
            class_label, is_affected = rule_func(paraphrase)
        else:
            class_label, is_affected = rule_func(event)

        class_label_preds[event]["class_label_preds"].append(class_label)
        if is_affected:
            class_label_preds[event]["affected_class_label_preds"].append(
                class_label)
        class_label_preds[event]["is_affected"].append(is_affected)
        class_label_preds[event]["rule_id"].append(rule_func.__name__)
        class_label_preds[event]["class_label_targets"].append(
            class_label_target)
        class_label_preds[event]["paraphrase"].append(paraphrase)
        return class_label_preds

    def get_averaged_paraphrases_class_label(self, event):
        return self.average_paraphrases_map[event]["class_label"]

    """
    ################################# delphi #################################
    """

    def delphi(self, event):
        if event in self.existing_constituents_cache:
            return None, False
        else:
            self.existing_constituents_cache.append(event)
            return self.delphi_cache[event]["class_label"], True

    def _constituents(self, event, c_name):
        c = self.constituents_map[event][c_name]
        is_affected = False
        class_label_pred = None

        if c != None and c not in self.existing_constituents_cache:
            self.existing_constituents_cache.append(c)
            is_affected = True
            class_label_pred = self.delphi_cache[c]["class_label"]

        return class_label_pred, is_affected

    def delphi_main_event(self, event):
        return self._constituents(event, "main_event")

    def delphi_main_event_main_clause(self, event):
        return self._constituents(event, "main_event_main_clause")

    def delphi_main_event_relative_clause(self, event):
        return self._constituents(event, "main_event_relative_clause")

    def delphi_adjunct_event(self, event):
        return self._constituents(event, "adjunct_event")

    """
    ################################# moral_saliency #################################
    """

    def e_moral_saliency(self, event):
        is_affected = False
        class_label_pred = None

        # If there's no morally salient keywords, then it mostly means the situation is morally neutral
        if self.binary_keywords_counts_map[event][1] == 0 and self.binary_keywords_counts_map[event][-1] == 0:
            class_label_pred = 0
            is_affected = True

        # If there are few number of morally salient keywords (<4), if not all keywords are negative,
        # then the situation is most likely to be morally neutral
        elif (self.binary_keywords_counts_map[event][1] + self.binary_keywords_counts_map[event][-1]) < 6 \
                and self.binary_keywords_counts_map[event][1] > 0:
            class_label_pred = 0
            is_affected = True
        return class_label_pred, is_affected

    def s_moral_saliency(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        # if both the main event and the adjunct event do not have many morally salient keywords
        # then the overall situation is most likely to be morally neutral
        if adjunct_event != None:
            if self.binary_keywords_counts_map[main_event][1] \
                    + self.binary_keywords_counts_map[main_event][-1] == 0 \
                    or (self.binary_keywords_counts_map[main_event][1]
                        + self.binary_keywords_counts_map[main_event][-1] < 4
                        and self.binary_keywords_counts_map[main_event][1] > 0):

                if self.binary_keywords_counts_map[adjunct_event][1] \
                        + self.binary_keywords_counts_map[adjunct_event][-1] < 4:
                    class_label_pred = 0
                    is_affected = True
        return class_label_pred, is_affected

    """
    ################################# moral_valency #################################
    """

    def e_moral_valency_0(self, event):
        is_affected = False
        class_label_pred = None

        # If all keywords are of the same valency, and if the total count of the keywords are within
        # reasonable range, assign label based on the valency of the keywords (negative)
        if self.binary_keywords_counts_map[event][1] == 0 \
                and self.binary_keywords_counts_map[event][-1] > 10 \
            and self.binary_keywords_counts_map[event][-1] < 35:
            class_label_pred = -1
            is_affected = True

        # If all keywords are of the same valency, and if the total count of the keywords are within
        # reasonable range, assign label based on the valency of the keywords (positive)
        elif self.binary_keywords_counts_map[event][-1] == 0 \
                and self.binary_keywords_counts_map[event][1] > 10 \
                and self.binary_keywords_counts_map[event][1] < 35:
            class_label_pred = 0
            is_affected = True
        return class_label_pred, is_affected

    def e_moral_valency_1(self, event):
        is_affected = False
        class_label_pred = None

        if self.top_level_keywords_counts_map["moral"][event] > 0 and self.binary_keywords_counts_map[event][-1] < 6:
            class_label_pred = 0
            is_affected = True
        return class_label_pred, is_affected

    def e_moral_valency_2(self, event):
        is_affected = False
        class_label_pred = None

        if self.top_level_keywords_counts_map["immoral"][event] > 0 and self.binary_keywords_counts_map[event][1] < 6:
            class_label_pred = -1
            is_affected = True
        return class_label_pred, is_affected

    def s_moral_valency_0(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            if self.binary_keywords_counts_map[adjunct_event][1] + self.binary_keywords_counts_map[adjunct_event][-1] < 2:
                # if the main_event is very negative, and if the adjunct_event doesn't have many morally salient
                # keywords, then the overall event is most likely to be morally negative
                if self.binary_keywords_counts_map[main_event][-1] - self.binary_keywords_counts_map[main_event][1] > 6:
                    class_label_pred = -1
                    is_affected = True

                # if the main_event is very positive, and if the adjunct_event doesn't have many morally salient
                # keywords, then the overall event is most likely to be morally positive
                elif self.binary_keywords_counts_map[main_event][1] - self.binary_keywords_counts_map[main_event][-1] > 6:
                    class_label_pred = 1
                    is_affected = True
        return class_label_pred, is_affected

    def s_moral_valency_1(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            # if both the main_event and adjunct_event have more morally negative keywords than morally positive
            # keywords, then the overall event is more likely to be morally negative
            if self.binary_keywords_counts_map[main_event][-1] - self.binary_keywords_counts_map[main_event][1] > 3:
                if self.binary_keywords_counts_map[adjunct_event][-1] - \
                        self.binary_keywords_counts_map[adjunct_event][1] > 3:
                    class_label_pred = -1
                    is_affected = True

            # if the main_event has much more morally positive keywords than morally negative keywords, and if the
            # adjunct_event is not overly negative, then the overall event is more likely to be morally positive/neutral
            elif self.binary_keywords_counts_map[main_event][1] - self.binary_keywords_counts_map[main_event][-1] > 6:
                if self.binary_keywords_counts_map[adjunct_event][-1] - \
                        self.binary_keywords_counts_map[adjunct_event][1] < 4:
                    class_label_pred = 0
                    is_affected = True
        return class_label_pred, is_affected

    def s_moral_valency_2(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            if self.top_level_keywords_counts_map["immoral"][main_event] > 0 and self.binary_keywords_counts_map[adjunct_event][1] < 6:
                class_label_pred = -1
                is_affected = True
        return class_label_pred, is_affected

    def s_moral_valency_3(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            if self.top_level_keywords_counts_map["moral"][main_event] > 0 and self.binary_keywords_counts_map[adjunct_event][-1] < 6:
                class_label_pred = 0
                is_affected = True
        return class_label_pred, is_affected

    """
    ################################# moral_impartiality #################################
    """

    def e_moral_impartiality_0(self, event):
        is_affected = False
        class_label_pred = None

        if self.top_level_keywords_counts_map["unjust"][event] > 0 and self.binary_keywords_counts_map[event][1] < 5:
            class_label_pred = -1
            is_affected = True
        return class_label_pred, is_affected

    def e_moral_impartiality_1(self, event):
        is_affected = False
        class_label_pred = None

        if self.top_level_keywords_counts_map["just"][event] > 0 and self.binary_keywords_counts_map[event][-1] < 5:
            class_label_pred = 0
            is_affected = True
        return class_label_pred, is_affected

    def e_moral_impartiality_2(self, event):
        is_affected = False
        class_label_pred = None

        if self.top_level_keywords_counts_map["unjust"][event] - self.top_level_keywords_counts_map["just"][event] > 1:
            class_label_pred = -1
            is_affected = True
        return class_label_pred, is_affected

    def s_moral_impartiality_0(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            if self.top_level_keywords_counts_map["unjust"][main_event] > 0 and self.binary_keywords_counts_map[main_event][1] < 5:
                class_label_pred = -1
                is_affected = True
        return class_label_pred, is_affected

    def s_moral_impartiality_1(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            if self.top_level_keywords_counts_map["unjust"][main_event] - self.top_level_keywords_counts_map["just"][main_event] > 0:
                class_label_pred = -1
                is_affected = True
        return class_label_pred, is_affected

    def s_moral_impartiality_2(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            if self.top_level_keywords_counts_map["unjust"][main_event] > 0 and self.binary_keywords_counts_map[adjunct_event][1] < 5:
                class_label_pred = -1
                is_affected = True
        return class_label_pred, is_affected

    """
    ################################# do_not_kill #################################
    """

    def e_kill_vs_conscious(self, event):
        is_affected = False
        class_label_pred = None

        # If there are many keywords about kill and less keywords about conscious,
        # then the situation is most likely to be negative
        if self.top_level_keywords_counts_map["kill"][event] > 8 \
                and self.top_level_keywords_counts_map["conscious"][event] < 8:
            class_label_pred = -1
            is_affected = True

        # If there are many keywords about conscious and less keywords about kill,
        # then the situation is most likely to be positive
        elif self.top_level_keywords_counts_map["conscious"][event] > 8 \
                and self.top_level_keywords_counts_map["kill"][event] < 8:
            class_label_pred = 0
            is_affected = True
        return class_label_pred, is_affected

    def s_kill_vs_conscious_0(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            # if the "killing" implication of the main_event is much stronger than the "conscious" implication of
            # the adjunct_event, then the overall event is most likely to be negative
            if self.top_level_keywords_counts_map["kill"][main_event] - \
                    self.top_level_keywords_counts_map["conscious"][adjunct_event] > 6:
                class_label_pred = -1
                is_affected = True

            # if the "conscious" implication of the main_event is much stronger than the "killing" implication of
            # the adjunct_event, then the overall event is most likely to be neutral/positive
            elif self.top_level_keywords_counts_map["conscious"][main_event] - \
                    self.top_level_keywords_counts_map["kill"][adjunct_event] > 6:
                class_label_pred = 0
                is_affected = True
        return class_label_pred, is_affected

    def s_kill_vs_conscious_1(self, event):
        main_clause = self.constituents_map[event]["main_event_main_clause"]
        relative_clause = self.constituents_map[event]["main_event_relative_clause"]
        is_affected = False
        class_label_pred = None

        # if the main event has killing implication, and if the main event has a relative clause,
        # and the relative clause has strong indication of killing, then the overall event is most
        # likely to be neutral, as the main action killing is likely to act on an object that is
        # likely to harm the subject
        if self.constituents_map[event]["main_event_relative_clause"] != None:
            if self.top_level_keywords_counts_map["kill"][main_clause] > 5 \
                    and self.top_level_keywords_counts_map["kill"][relative_clause] > 11:
                class_label_pred = 0
                is_affected = True
        return class_label_pred, is_affected

    def s_kill_vs_conscious_2(self, event):
        main_clause = self.constituents_map[event]["main_event_main_clause"]
        relative_clause = self.constituents_map[event]["main_event_relative_clause"]
        is_affected = False
        class_label_pred = None

        # if the main event has killing implication, and if the main event has a relative clause,
        # and the relative clause has strong indication of killing/harming, then the overall event is most
        # likely to be neutral, as the main action killing is likely to act on an object that is
        # likely to harm the subject
        if self.constituents_map[event]["main_event_relative_clause"] != None:
            if self.top_level_keywords_counts_map["kill"][main_clause] > 5 \
                    and self.top_level_keywords_counts_map["kill"][relative_clause] > 0 \
                    and self.top_level_keywords_counts_map["pain physical"][relative_clause] > 5:
                class_label_pred = 0
                is_affected = True
        return class_label_pred, is_affected

    def s_kill_vs_conscious_3(self, event):
        main_clause = self.constituents_map[event]["main_event_main_clause"]
        relative_clause = self.constituents_map[event]["main_event_relative_clause"]
        is_affected = False
        class_label_pred = None

        # if the relative clause has no implication of killing, then the main action killing
        # may not be justified
        if self.constituents_map[event]["main_event_relative_clause"] != None:
            if self.top_level_keywords_counts_map["kill"][main_clause] > 0 \
                    and self.top_level_keywords_counts_map["kill"][relative_clause] == 0:
                class_label_pred = -1
                is_affected = True
        return class_label_pred, is_affected

    """
    ################################# do_not_cause_pain_physical #################################
    """

    def e_do_not_cause_pain_physical(self, event):
        is_affected = False
        class_label_pred = None

        # If there are many keywords about harm, and there are less positive keywords,
        # then the situation is most likely to be negative
        if self.top_level_keywords_counts_map["pain physical"][event] > 10 \
                and self.binary_keywords_counts_map[event][1] < 5:
            class_label_pred = -1
            is_affected = True
        return class_label_pred, is_affected

    def s_do_not_cause_pain_physical_0(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            # if the "harm" implication of the main_event is strong, and the positive implication fo the
            # adjunct_event is weak, then the overall event is likely to be negative
            if self.top_level_keywords_counts_map["pain physical"][main_event] > 8 \
                    and self.binary_keywords_counts_map[adjunct_event][1] < 2:
                class_label_pred = -1
                is_affected = True
        return class_label_pred, is_affected

    def s_do_not_cause_pain_physical_1(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            # if the "harm" implication of the main_event is stronger than the positive implication of the
            # adjunct_event, then the overall event is likely to be negative
            if self.top_level_keywords_counts_map["pain physical"][main_event] - \
                    self.binary_keywords_counts_map[adjunct_event][1] > 5:
                class_label_pred = -1
                is_affected = True
        return class_label_pred, is_affected

    """
    ################################# do_not_cause_pain_mental #################################
    """

    def e_do_not_cause_pain_mental(self, event):
        is_affected = False
        class_label_pred = None

        if self.top_level_keywords_counts_map["pain mental"][event] > 15 \
                and self.binary_keywords_counts_map[event][1] < 5:
            class_label_pred = -1
            is_affected = True
        return class_label_pred, is_affected

    def s_do_not_cause_pain_mental_0(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            if self.top_level_keywords_counts_map["pain mental"][main_event] > 12 \
                    and self.binary_keywords_counts_map[adjunct_event][1] < 4:
                class_label_pred = -1
                is_affected = True
        return class_label_pred, is_affected

    def s_do_not_cause_pain_mental_1(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            if self.top_level_keywords_counts_map["pain mental"][main_event] - \
                    self.binary_keywords_counts_map[adjunct_event][1] > 8:
                class_label_pred = -1
                is_affected = True
        return class_label_pred, is_affected

    """
    ################################# do_not_disable #################################
    """

    def e_do_not_disable_0(self, event):
        is_affected = False
        class_label_pred = None

        if self.top_level_keywords_counts_map["disable"][event] > 15 \
                and self.binary_keywords_counts_map[event][1] < 4:
            class_label_pred = -1
            is_affected = True
        return class_label_pred, is_affected

    def e_do_not_disable_1(self, event):
        is_affected = False
        class_label_pred = None

        if self.top_level_keywords_counts_map["able"][event] > 15 \
                and self.binary_keywords_counts_map[event][-1] < 2:
            class_label_pred = 0
            is_affected = True
        return class_label_pred, is_affected

    def s_do_not_disable_0(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            if self.top_level_keywords_counts_map["disable"][main_event] > 10 \
                    and self.binary_keywords_counts_map[adjunct_event][1] < 2:
                class_label_pred = -1
                is_affected = True
        return class_label_pred, is_affected

    def s_do_not_disable_1(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            if self.top_level_keywords_counts_map["able"][main_event] > 15 \
                    and self.binary_keywords_counts_map[adjunct_event][-1] < 2:
                class_label_pred = 0
                is_affected = True
        return class_label_pred, is_affected

    def s_do_not_disable_2(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            if self.top_level_keywords_counts_map["disable"][main_event] - \
                    self.binary_keywords_counts_map[adjunct_event][1] > 5:
                class_label_pred = -1
                is_affected = True
        return class_label_pred, is_affected

    def s_do_not_disable_3(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            if self.top_level_keywords_counts_map["able"][main_event] - \
                    self.binary_keywords_counts_map[adjunct_event][-1] > 6:
                class_label_pred = 0
                is_affected = True
        return class_label_pred, is_affected

    """
    ################################# do_not_deprive_of_freedom #################################
    """

    def e_do_not_deprive_of_freedom_0(self, event):
        is_affected = False
        class_label_pred = None

        if self.top_level_keywords_counts_map["no freedom"][event] > 5 \
                and self.binary_keywords_counts_map[event][1] < 4:
            class_label_pred = -1
            is_affected = True
        return class_label_pred, is_affected

    def e_do_not_deprive_of_freedom_1(self, event):
        is_affected = False
        class_label_pred = None

        if self.top_level_keywords_counts_map["freedom"][event] > 5 \
                and self.binary_keywords_counts_map[event][-1] < 4:
            class_label_pred = 1
            is_affected = True
        return class_label_pred, is_affected

    def s_do_not_deprive_of_freedom_0(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            if self.top_level_keywords_counts_map["no freedom"][main_event] > 4 \
                    and self.binary_keywords_counts_map[adjunct_event][1] < 4:
                class_label_pred = -1
                is_affected = True
        return class_label_pred, is_affected

    def s_do_not_deprive_of_freedom_1(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            if self.top_level_keywords_counts_map["freedom"][main_event] > 4 \
                    and self.binary_keywords_counts_map[adjunct_event][-1] < 4:
                class_label_pred = 1
                is_affected = True
        return class_label_pred, is_affected

    """
    ################################# do_not_deprive_of_pleasure #################################
    """

    def e_do_not_deprive_of_pleasure_0(self, event):
        is_affected = False
        class_label_pred = None

        if self.top_level_keywords_counts_map["no pleasure"][event] > 2 \
                and self.binary_keywords_counts_map[event][1] < 4:
            class_label_pred = -1
            is_affected = True
        return class_label_pred, is_affected

    def e_do_not_deprive_of_pleasure_1(self, event):
        is_affected = False
        class_label_pred = None

        if self.top_level_keywords_counts_map["pleasure"][event] > 4 \
                and self.binary_keywords_counts_map[event][-1] < 4:
            class_label_pred = 0
            is_affected = True
        return class_label_pred, is_affected

    def s_do_not_deprive_of_pleasure_0(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            if self.top_level_keywords_counts_map["no pleasure"][main_event] > 2 \
                    and self.binary_keywords_counts_map[adjunct_event][1] < 4:
                class_label_pred = -1
                is_affected = True
        return class_label_pred, is_affected

    def s_do_not_deprive_of_pleasure_1(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            if self.top_level_keywords_counts_map["pleasure"][main_event] > 4 \
                    and self.binary_keywords_counts_map[adjunct_event][-1] < 4:
                class_label_pred = 0
                is_affected = True
        return class_label_pred, is_affected

    """
    ################################# do_not_deceive #################################
    """

    def e_do_not_deceive(self, event):
        is_affected = False
        class_label_pred = None

        if self.top_level_keywords_counts_map["deceive"][event] > 2 \
                and self.binary_keywords_counts_map[event][1] < 4:
            class_label_pred = -1
            is_affected = True
        return class_label_pred, is_affected

    def s_do_not_deceive(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            if self.top_level_keywords_counts_map["deceive"][main_event] > 2 \
                    and self.binary_keywords_counts_map[adjunct_event][1] < 4:
                class_label_pred = -1
                is_affected = True
        return class_label_pred, is_affected

    ################################# keep_your_promises #################################
    def e_keep_your_promises(self, event):
        is_affected = False
        class_label_pred = None

        if self.top_level_keywords_counts_map["no promise"][event] > 1 \
                and self.binary_keywords_counts_map[event][1] < 4:
            class_label_pred = -1
            is_affected = True
        return class_label_pred, is_affected

    def s_keep_your_promises(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            if self.top_level_keywords_counts_map["no promise"][main_event] > 1 \
                    and self.binary_keywords_counts_map[adjunct_event][1] < 4:
                class_label_pred = -1
                is_affected = True
        return class_label_pred, is_affected

    """
    ################################# do_not_cheat #################################
    """

    def e_sex(self, event):
        is_affected = False
        class_label_pred = None

        # When the keyword "sex" appears reasonably infrequently, and if the other keywords are not overly
        # negative, it usually refers to ok actions
        if self.top_level_keywords_counts_map["sex"][event] > 0 \
                and self.top_level_keywords_counts_map["sex"][event] < 16:
            if self.binary_keywords_counts_map[event][-1] - self.top_level_keywords_counts_map["sex"][event] < 5:
                class_label_pred = 0
                is_affected = True

        # When the keyword "sex" appears very frequently, it usually refers to negative actions (e.g., raping)
        elif self.top_level_keywords_counts_map["sex"][event] > 16:
            class_label_pred = -1
            is_affected = True
        return class_label_pred, is_affected

    def e_no_sex(self, event):
        is_affected = False
        class_label_pred = None

        # When the keyword "no sex" appears, it usually refers to bad actions
        if self.top_level_keywords_counts_map["not sex"][event] > 0:
            class_label_pred = -1
            is_affected = True
        return class_label_pred, is_affected

    def s_sex_0(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            # if the main_action has very strong implication of "sex", it's likely the overall event is negative
            if self.top_level_keywords_counts_map["sex"][main_event] > 15 \
                    and self.binary_keywords_counts_map[adjunct_event][1] < 10:
                class_label_pred = -1
                is_affected = True
        return class_label_pred, is_affected

    def s_sex_1(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            # if both the main_event and the adjunct_event have relative strong implication of "sex",
            # then the overall event is most likely to be morally negative
            if self.top_level_keywords_counts_map["sex"][main_event] > 6 \
                    and self.top_level_keywords_counts_map["sex"][adjunct_event] > 6:
                class_label_pred = -1
                is_affected = True
        return class_label_pred, is_affected

    def s_no_sex(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            # if the main action has implication of "no sex", and if the adjunct_event is not overly positive
            # then the overall event is more likely to be morally negative
            if self.top_level_keywords_counts_map["not sex"][main_event] > 0 \
                    and self.binary_keywords_counts_map[adjunct_event][1] < 10:
                class_label_pred = -1
                is_affected = True
        return class_label_pred, is_affected

    def e_do_not_cheat(self, event):
        is_affected = False
        class_label_pred = None

        if self.top_level_keywords_counts_map["no cheat"][event] > 1 \
                and self.binary_keywords_counts_map[event][1] < 4:
            class_label_pred = -1
            is_affected = True
        return class_label_pred, is_affected

    def s_do_not_cheat(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            if self.top_level_keywords_counts_map["no cheat"][main_event] > 1 \
                    and self.binary_keywords_counts_map[adjunct_event][1] < 4:
                class_label_pred = -1
                is_affected = True
        return class_label_pred, is_affected

    """
    ################################# obey_the_law #################################
    """

    def e_illegal(self, event):
        is_affected = False
        class_label_pred = None

        # If there are many keywords about crime, and there are less positive keywords,
        # then the situation is mostly likely to be negative
        if self.top_level_keywords_counts_map["no law"][event] > 15 \
                and self.binary_keywords_counts_map[event][1] < 5:
            class_label_pred = -1
            is_affected = True
        return class_label_pred, is_affected

    def s_illegal(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            # if the main_event has implication of "crime", and the adjunct_event doesn't have overly
            # strong positive implication, then the overall event is likely to be negative
            if self.top_level_keywords_counts_map["no law"][main_event] > 12 \
                    and self.binary_keywords_counts_map[adjunct_event][1] < 5:
                class_label_pred = -1
                is_affected = True
        return class_label_pred, is_affected

    """
    ################################# do_your_duty #################################
    """

    def e_do_your_duty(self, event):
        is_affected = False
        class_label_pred = None

        if self.top_level_keywords_counts_map["duty"][event] > 1 \
                and self.binary_keywords_counts_map[event][1] < 4:
            class_label_pred = 0
            is_affected = True
        return class_label_pred, is_affected

    def s_do_your_duty(self, event):
        main_event = self.constituents_map[event]["main_event"]
        adjunct_event = self.constituents_map[event]["adjunct_event"]
        is_affected = False
        class_label_pred = None

        if adjunct_event != None:
            if self.top_level_keywords_counts_map["duty"][main_event] > 1 \
                    and self.binary_keywords_counts_map[adjunct_event][1] < 4:
                class_label_pred = 0
                is_affected = True
        return class_label_pred, is_affected


if __name__ == "__main__":
    delphi_ruler = Ruler()

    # for agreement_rate in ["certain", "ambiguous", "all"]:
    delphi_ruler.main(agreement_rate=1)
    # print(delphi_ruler.main.__name__)
    # print(delphi_ruler.comet_cache["silently farting in a crowded room"])
