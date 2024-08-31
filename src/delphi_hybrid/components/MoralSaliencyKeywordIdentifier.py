import os
import sys
import random

sys.path.append(os.getcwd())

from src.delphi_hybrid.components.utils import *
# from src.delphi_hybrid.components.bank import *

moral_saliency_keywords = ["loyal", "welfare", "benefit", "authority", "dignity", "assassinate",
                           "greedy", "weapon", "dishonest", "guilty", "sad", "destructive",
                           "terror", "torture", "violent", "aggressive", "terrible", "hunger",
                           "cruel", "inhumane", "immoral", "illegal", "caring", "disaster",
                           "hungry", "careful", "careless", "crime", "jail", "prison", "survive",
                           "protect", "evil", "pain", "deceive", "deaf", "blind", "addiction",
                           "phobia", "illegal", "adultery", "renege", "guilt", "shame", "joy",
                           "remorse", "outrage", "angry", "disgusted", "anxious", "displeased",
                           "poverty", "slum", "discrimination", "bias", "favoritism", "hypocrisy",
                           "wealth", "hypocrite", "knowledge", "peace", "education", "medicare",
                           "duty", "ability", "talent", "freedom", "privacy", "enjoy", "smile",
                           "happiness", "happy", "satisfaction", "satisfy", "restrain", "coercion",
                           "deprive", "unethical", "ethical", "unjust", "upright", "equitable",
                           "honest", "reasonable", "impartial", "unbiased"]

moral_saliency_keywords_w_people = []

passive_keywords = [("revenge", "revenged"),
                    ("violate", "violated"),
                    ("destroy", "destroyed"),
                    ("threaten", "threatened"),
                    ("arrest", "arrested"),
                    ("cheat", "cheated"),
                    ("betray", "betrayed"),
                    ("terrify", "terrified"),
                    ]

relative_people_keywords = ["mother", "father", "mom", "dad", "parent",
                            "grandmother", "grandfather", "grandma", "grandpa", "grandparent",
                            "aunt", "uncle", "cousin",
                            "sister", "brother", "sibling"
                            "father-in-law", "mother-in-law", "in-law",
                            "stepmom", "stepdad"]

young_people_keywords = ["kid", "child", "baby", "babies", "underage"]

people_keywords = ["someone", "people", "person", "mankind",
                   "women", "men", "woman", "man",
                   "girl", "boy",
                   "wife", "husband",
                   "I", "she", "he",
                   "me", "you", "her", "him", "them",
                   "victim",
                   "life", "lives",
                   "myself", "yourself", "themselves",
                   "soul"] + relative_people_keywords + young_people_keywords

negation_keywords = ["low", "lack", "not ", "no ", "lose", "losing", "loss", "lost",
                     "waste", "lack", "less", "prevent", "stop", "avoid", "relieve",
                     "end ", "ends ","ended ", "get rid of", "report", "investigate", "n't",
                     "get rid", "free from", "is gone", "get away with", "victim",
                     "free of", "out of", "rid of", "break"]

be_conjugates = ["be ", "being ", "is ", "are ", "was ", "were "]
have_conjugates = ["have", "has", "had", "having"]
get_conjugates = ["get", "got"]


class MoralSaliencyKeywordIdentifier:
    def __init__(self):
        self.all_moral_saliency_keywords_count_dict = {}

    def add_keywords_count(self, moral_saliency_keywords_count_dict, keyword):
        if keyword in moral_saliency_keywords_count_dict:
            moral_saliency_keywords_count_dict[keyword] += 1
        else:
            moral_saliency_keywords_count_dict[keyword] = 1

        if keyword in self.all_moral_saliency_keywords_count_dict:
            self.all_moral_saliency_keywords_count_dict[keyword] += 1
        else:
            self.all_moral_saliency_keywords_count_dict[keyword] = 1
        return moral_saliency_keywords_count_dict

    def search_keywords(self, te, moral_saliency_keywords_count_dict):
        for kw in moral_saliency_keywords:
            if kw in te:
                if self.any_keyword_in(te, negation_keywords):
                    moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "not " + kw)
                else:
                    moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, kw)
                return moral_saliency_keywords_count_dict, True
        return moral_saliency_keywords_count_dict, False

    def check_passive_voice_keywords(self, te, moral_saliency_keywords_count_dict):
        for (kw, passive_kw) in passive_keywords:
            if kw in te:
                if passive_kw in te:
                    moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, passive_kw)
                    return moral_saliency_keywords_count_dict, True
                elif not self.any_keyword_in(te, negation_keywords):
                    moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, kw)
                    return moral_saliency_keywords_count_dict, True
        return moral_saliency_keywords_count_dict, False

    def any_keyword_in(self, te, keywords):
        return any(kw in te for kw in keywords)

    def all_keyword_in(self, te, keywords):
        return all(kw in te for kw in keywords)

    def any_keyword_match(self, te, keywords):
        return (te in keywords)

    def identify_moral_saliency_keywords(self, comet_inferences):
        moral_saliency_keywords_count_dict = {}
        for r in comet_relations:
            tail_events = comet_inferences[r]
            for te in tail_events:
                te = te.lower()

                moral_saliency_keywords_count_dict, is_keyword_searched = self.search_keywords(te, moral_saliency_keywords_count_dict)
                moral_saliency_keywords_count_dict, is_passive_keyword_searched = self.check_passive_voice_keywords(te, moral_saliency_keywords_count_dict)

                if not is_keyword_searched and not is_passive_keyword_searched:
                    if "kill" in te:
                        kill_ignore_keywords = ["skill"]
                        killed_keywords_not_in = ["not kill", "killed", "killer"]
                        killed_keywords = ["get killed", "gets killed", "got killed",
                                           "is killed", "are killed", "being killed"]
                        if self.any_keyword_in(te, kill_ignore_keywords):
                            continue
                        elif not self.any_keyword_in(te, killed_keywords_not_in):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "kill")
                        elif self.any_keyword_in(te, killed_keywords) or "killed" == te:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "killed")

                    elif "murder" in te:
                        murder_keywords_not_in = ["murdered"] + negation_keywords
                        murdered_keywords = ["murdered"]
                        if not self.any_keyword_in(te, murder_keywords_not_in):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "murder")
                        elif self.any_keyword_in(te, murdered_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "murdered")

                    elif "save" in te:
                        save_money_keywords = ["save up", "money"]
                        save_time_keywords = ["time"]
                        save_life_keywords_not_in = ["environment"]
                        if self.any_keyword_in(te, negation_keywords):
                            continue
                        elif ("save for" in te and "day" not in te) or self.any_keyword_in(te, save_money_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "have money")
                        elif self.any_keyword_in(te, save_time_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "save time")
                        elif self.any_keyword_in(te, people_keywords) and not self.any_keyword_in(te, save_life_keywords_not_in):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "save life")

                    elif "death" in te:
                        if "death penalty" in te:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "death penalty")
                        else:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "death")

                    elif "die" in te:
                        die_keywords = ["die ", "dies ", "died ", " die", " dies", " died"]
                        die_keywords_match = ["die", "died", "dies"]
                        die_keywords_not_in = ["diet", "studied", "peacefully"] + negation_keywords
                        if (self.any_keyword_match(te, die_keywords_match) or self.any_keyword_in(te, die_keywords)) \
                            and not self.any_keyword_in(te, die_keywords_not_in):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "die")

                    elif "money" in te:
                        have_money_keywords = ["make", "earn", "gain", "receive", "raise", "win", "more"] + have_conjugates + get_conjugates
                        spend_money_keywords = ["spend", "spent", "cost"]
                        if self.any_keyword_in(te, have_money_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "have money")
                        elif self.any_keyword_in(te, spend_money_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "spend money")
                        elif self.any_keyword_in(te, negation_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "lose money")

                    elif "harm" in te:
                        harm_keywords = ["harm ", " harm", "harmed", "harmful"]
                        harm_keywords_not_in = ["harmony"]
                        if self.any_keyword_in(te, harm_keywords) and not self.any_keyword_in(te, harm_keywords_not_in):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "harm")

                    elif "job" in te:
                        have_job_keywords = ["find", "start"] + have_conjugates + get_conjugates
                        leave_job_keywords = ["leave", "leaving", "left", "quit"]
                        if self.any_keyword_in(te, have_job_keywords) or self.all_keyword_in(te, ["receive", "offer"]):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "have job")
                        elif self.any_keyword_in(te, leave_job_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "leave job")
                        elif self.any_keyword_in(te, negation_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "lose job")

                    elif "income" in te:
                        income_keywords = ["increase", "gain", "receive", "have", "has", "having", "had"]
                        if self.any_keyword_in(te, income_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "have money")
                        elif self.any_keyword_in(te, negation_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "lose money")

                    elif "trust" in te:
                        if "untrustworthy" in te:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "lose trust")
                        elif self.any_keyword_in(te, negation_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "lose trust")
                        else:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "gain trust")

                    elif "respect" in te:
                        gain_respect_keywords = ["receive", "gain", "earn", "respected", "respectful"] + be_conjugates + have_conjugates + get_conjugates
                        show_respect_keywords = ["show"]
                        if "disrespect" in te:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "lose respect")
                        elif self.any_keyword_in(te, gain_respect_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "gain respect")
                        elif "respect" == te or self.any_keyword_in(te, show_respect_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "show respect")
                        elif self.any_keyword_in(te, negation_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "lose respect")

                    elif "racis" in te:
                        not_racist_keywords = negation_keywords # ["victim"]

                        if self.any_keyword_in(te, not_racist_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "not racist")
                        else:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "racist")

                    elif "sexis" in te:
                        not_sexist_keywords = ["learn"] + negation_keywords # "victim",

                        if self.any_keyword_in(te, not_sexist_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "not sexist")
                        else:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "sexist")

                    elif "insurance" in te:
                        if self.any_keyword_in(te, negation_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "not insurance")
                        else:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "have insurance")

                    elif "health" in te:
                        not_health_keywords = ["unhealthy", "bad", "problem"] + negation_keywords
                        if self.any_keyword_in(te, not_health_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "not health")
                        else:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "health")

                    elif "sick" in te:
                        if self.any_keyword_in(te, negation_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "health")
                        else:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "not health")

                    elif "scar" in te:
                        scare_keywords_not_in = ["scarecrow", "movie", "oscar", "sportscar"] + negation_keywords
                        scared_keywords = ["scared"]
                        if self.any_keyword_in(te, scare_keywords_not_in):
                            continue
                        if self.any_keyword_in(te, scared_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "scared")
                        else:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "scare")

                    elif "care" in te:
                        care_ignore_keywords = ["career"] + negation_keywords
                        care_for_keywords = ["for", "about"]
                        if self.any_keyword_in(te, care_ignore_keywords):
                            continue
                        elif self.all_keyword_in(te, ["take", "of"]):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "take care of")
                        elif self.any_keyword_in(te, care_for_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "care for")

                    elif "fair" in te:
                        fair_ignore_keywords = ["fairy"] + negation_keywords
                        unfair_keywords = ["unfair"]
                        fair_keywords = ["fairness"]
                        if self.any_keyword_in(te, fair_ignore_keywords):
                            continue
                        elif self.any_keyword_in(te, unfair_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "unfair")
                        elif "fair" == te or self.any_keyword_in(te, fair_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "fair")

                    elif "safe" in te:
                        safe_ignore_keywords = ["unsafe"] + negation_keywords
                        if self.any_keyword_in(te, safe_ignore_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "danger")
                        else:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "safe")

                    elif "danger" in te:
                        if self.any_keyword_in(te, negation_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "safe")
                        else:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "danger")

                    elif "life" in te:
                        live_life_keywords = ["enjoy", "live", "good", "nice", "better", "succeed", "important",
                                              "new", "essential", "improve", "appreciate", "success", "fulfill"]
                        if self.any_keyword_in(te, live_life_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "live life")
                        elif self.any_keyword_in(te, negation_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "lose life")

                    elif "suicide" in te or "suicidal" in te:
                        moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "suicide")

                    elif "mental" in te:
                        mental_illness_keywords = ["ill", "disability", "disabled"]
                        if self.any_keyword_in(te, mental_illness_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "mental illness")

                    elif "hurt" in te:
                        if self.any_keyword_in(te, get_conjugates):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "get hurt")
                        elif self.any_keyword_in(te, negation_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "not hurt")
                        elif te == "hurt" or self.any_keyword_in(te, people_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "hurt")

                    elif "dead" in te:
                        dead_keywords = ["body", "bodies", "honor"] + people_keywords
                        dead_keywords_match = ["dead", "is dead"]
                        if self.any_keyword_in(te, dead_keywords) or self.any_keyword_match(te, dead_keywords_match):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "dead")

                    elif "shot" in te:
                        gun_shot_keywords = [w + " shot" for w in get_conjugates] + ["gun "]
                        if self.any_keyword_in(te, gun_shot_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "gun shot")

                    elif "war" in te:
                        war_keywords = ["go to war", "goes to war", "went to war", "start", "have war", "has war", "had war"]
                        if "war" == te or self.any_keyword_in(te, war_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "war")

                    elif "moral" in te:
                        unmoral_keywords = ["unmoral"] + negation_keywords
                        if self.any_keyword_in(te, unmoral_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "immoral")
                        else:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "moral")

                    elif "mean" in te:
                        mean_keywords = ["to be mean", "mean-spirited", "very mean", "heart"]
                        if "mean" == te or self.any_keyword_in(te, mean_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "mean")

                    elif "criminal" in te:
                        crime_keywords = ["become", "be ", "one kind of", "is a criminal"]
                        not_crime_keywords = ["stop", "scare away", "catch", "bait", "tie up", "punish",
                             "fine", "corner", "restrain", "rid of", "control", "fight",
                             "hold", "report", "capture", "lock up", "imprisonment"] + negation_keywords
                        if "criminal" == te or self.any_keyword_in(te, crime_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "crime")
                        elif self.any_keyword_in(te, not_crime_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "not crime")

                    elif "depress" in te:
                        not_depress_keywords = ["help with", "get out of"]
                        if self.any_keyword_in(te, not_depress_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "not depress")
                        else:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "depress")

                    elif "attack" in te:
                        health_attack_keywords = ["anxiety", "heart", "panic"]
                        if "attacked" in te:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "attacked")
                        elif self.any_keyword_in(te, health_attack_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "not health")
                        else:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "attacking")

                    elif "injur" in te:
                        if "injured" in te or ("injury" in te and "cause" not in te):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "injured")
                        elif self.any_keyword_in(te, negation_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "safe")
                        else:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "injure")

                    elif "gun" in te:
                        shoot_gun_keywords = ["shoot", "fire", "use"]
                        if self.any_keyword_in(te, shoot_gun_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "gun shot")

                    elif "sex" in te:
                        bad_sex_keywords = relative_people_keywords + young_people_keywords
                        if self.any_keyword_in(te, bad_sex_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "not sex")
                        else:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "sex")

                    elif "rape" in te:
                        rape_keywords_not_in = ["scrape", "grape", "drape", "crape"]
                        raped_keywords = ["victim"]
                        if self.any_keyword_in(te, rape_keywords_not_in):
                            continue
                        elif self.any_keyword_in(te, raped_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "raped")
                        else:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "rape")

                    elif "hit" in te:
                        hit_keywords_not_in = ["white", "hitler"]
                        get_hit_keywords = [kw + " " for kw in get_conjugates] + [kw + "hit" for kw in be_conjugates]
                        if self.any_keyword_in(te, hit_keywords_not_in):
                            continue
                        elif self.any_keyword_in(te, get_hit_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "get hit")
                        elif "hit" == te or self.any_keyword_in(te, people_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "hit")

                    elif "beat" in te:
                        if "beat" == te or self.any_keyword_in(te, people_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "hit")

                    elif "homeless" in te:
                        if "become" in te:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "homeless")

                    elif "lie" in te:
                        lie_keywords_not_in = ["relieve", "believe", "down", "relief"]
                        if self.any_keyword_in(te, lie_keywords_not_in):
                            continue
                        else:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "lie")

                    elif "conscious" in te:
                        conscious_keywords_not_in = ["self"]
                        unconscious_keywords = ["unconscious"] + negation_keywords
                        if self.any_keyword_in(te, conscious_keywords_not_in):
                            continue
                        elif self.any_keyword_in(te, unconscious_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "unconscious")
                        elif te == "conscious":
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "conscious")

                    elif "disable" in te or "disability" in te:
                        if self.any_keyword_in(te, negation_keywords):
                            continue
                        elif te == "disabled" or self.any_keyword_in(te, people_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "disable")

                    elif "promise" in te:
                        promise_not_in = ["compromise"]
                        if self.any_keyword_in(te, promise_not_in):
                            continue
                        elif self.any_keyword_in(te, negation_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "break promise")
                        else:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "promise")

                    elif "anger" in te:
                        anger_not_in = ["ranger"]

                        if self.any_keyword_in(te, anger_not_in):
                            continue
                        elif self.any_keyword_in(te, negation_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "no anger")
                        else:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "anger")

                    elif "mad" in te:
                        mad_not_in = ["made"]

                        if self.any_keyword_in(te, mad_not_in):
                            continue
                        elif self.any_keyword_in(te, negation_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "no mad")
                        else:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "mad")

                    elif "love" in te:
                        love_not_in = ["glove"]
                        make_love_keywords = ["make", "made"]
                        if self.any_keyword_in(te, love_not_in):
                            continue
                        elif self.any_keyword_in(te, make_love_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "sex")
                        elif self.any_keyword_in(te, negation_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "no love")
                        else:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "love")

                    elif "intelligen" in te:
                        intelligence_not_in = ["artificial"]

                        if self.any_keyword_in(te, intelligence_not_in):
                            continue
                        elif self.any_keyword_in(te, negation_keywords):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "no intelligence")
                        else:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "intelligence")

                    elif "pleasure" in te:
                        pleasure_not_in = ["dis"] + negation_keywords
                        if self.any_keyword_in(te, pleasure_not_in):
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "no pleasure")
                        else:
                            moral_saliency_keywords_count_dict = self.add_keywords_count(moral_saliency_keywords_count_dict, "pleasure")

                    # elif "reasonable" in te:
                    #     print(te)
                        # else:
                    # fear

                    #

                    # "equitable", "reasonable", "impartial", "unbiased"


        return moral_saliency_keywords_count_dict


if __name__ == '__main__':
    saliency_identifier = MoralSaliencyKeywordIdentifier()

    input_file = data_base_path + "cache/data_gold_labels.csv"
    df_data_ref = pd.read_csv(input_file, sep=",")
    df_data_ref = df_data_ref[df_data_ref["split"].isin(["train"])]
    events_ref = df_data_ref["event"].tolist()
    # print(len(events_ref))
    # print(df_data_ref)

    paraphrases_cache = read_json(data_base_path + f"cache/paraphrases_filtered_by_nli.json")
    print(f"* Paraphrases cache loaded! ({len(paraphrases_cache)})")

    # input_file = "data/demo/demo_examples_102721_delphi_full.csv"
    #     # data_base_path + "demo/demo_correct_examples.tsv"
    # df_data = pd.read_csv(input_file, sep="\t")
    # df_data = df_data.drop_duplicates(subset=['action1'])
    # # print(df_data.shape)
    # df_data = df_data[~df_data["action1"].isin(events_ref)]
    # # print(df_data.shape)
    # events = df_data["action1"].tolist()


    # print(df_data)
    #
    # random.shuffle(events)
    #
    # # print(len(events))
    #
    # # events = df_data["event"].tolist()
    # #
    #
    # comet_cache = read_json(data_base_path + f"cache/nov2022/comet.json")
    comet_cache = read_json(data_base_path + f"cache/comet_subset.json")
    print("COMET loaded!")

    for e in events_ref:
        if e in comet_cache:
            comet_inferences = comet_cache[e]
            moral_saliency_keywords_count_dict = saliency_identifier.identify_moral_saliency_keywords(comet_inferences)

        for p in paraphrases_cache[e]:
            if e in comet_cache:
                comet_inferences = comet_cache[e]
                moral_saliency_keywords_count_dict = saliency_identifier.identify_moral_saliency_keywords(comet_inferences)




