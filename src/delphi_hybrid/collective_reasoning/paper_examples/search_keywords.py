import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.getcwd())

from scripts.utils.utils import *
from scripts.utils.constants import *
from scripts.utils.bank import *
from scripts.utils.MoralSaliencyKeywordIdentifier import *

from scripts.utils.COMETGenerator import *

# from scripts.utils.CompositionalityParser import *
# from scripts.utils.PersonDetector import *

comet_generator = COMETGenerator(device_id=0)
saliency_identifier = MoralSaliencyKeywordIdentifier()

# head_events = ["performing genocide if it creates job",
# 			   "performing genocide",
# 			   "it creates job",
# 			   "killing an aggressive bear",
# 			   "killing an ugly bear"]


head_events = ["killing a bear to save your child",
			   "killing a bear",
			   "save your child",]

for head_event in head_events:
	tree = Tree()
	tree.create_node("\n" + head_event + "\n", head_event)

	comet_gen = {}
	for relation in comet_relations:
		tail_events = comet_generator.generate_beam(head_event, relation)
		# print(tail_events)

		tail_events_string = " | ".join(tail_events)
		tree.create_node("{:<12} >>> {:>11}".format(relation, tail_events_string), relation, parent=head_event)

		comet_gen[relation] = tail_events

	moral_saliency_keywords_count_dict = saliency_identifier.identify_moral_saliency_keywords(comet_gen)
	print(moral_saliency_keywords_count_dict)

	tree.show()






