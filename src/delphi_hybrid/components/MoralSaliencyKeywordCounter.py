
import os
import sys

sys.path.append(os.getcwd())
from src.delphi_hybrid.components.DelphiScorer import *
from src.delphi_hybrid.components.CacheHandler.CacheHandler import *

class MoralSaliencyKeywordCounter():
    def __init__(self, filename=data_base_path + f"cache/keywords.json"):
        self.keywords_cache = read_json(filename)

    def get_events_keyword_counts(self, e, keyword):
        e_keywords = self.keywords_cache[e]

        all_keywords_selected = all_keywords_categorize[keyword]
        keywords_counts = {k: 0 for k in all_keywords_selected}

        for k in all_keywords_selected:
            if k in e_keywords:
                keywords_counts[k] = e_keywords[k]
        return keywords_counts

    def get_all_top_level_keyword_count(self, keywords_map):
        return sum([keywords_map[k] for k in keywords_map])

    def get_event_top_level_keyword_count(self, keywords_counts):
        return sum([keywords_counts[k] for k in keywords_counts])

    def get_top_level_keywords_counts_map(self, top_level_keyword):
        top_level_keywords_counts_map = {}
        for e in self.keywords_cache:
            keywords_counts = self.get_events_keyword_counts(e, top_level_keyword)
            keywords_counts_map[e] = self.get_all_top_level_keyword_count(keywords_counts)
        return keywords_counts_map

if __name__ == "__main__":
    keyword_counter = MoralSaliencyKeywordCounter()
    # instance = cache_handler.get_instance("Killing a bear")

