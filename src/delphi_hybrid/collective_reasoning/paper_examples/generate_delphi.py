import os
import sys
from tqdm import tqdm
import argparse
import pandas as pd

sys.path.append(os.getcwd())

from scripts.utils.utils import *
from scripts.utils.DelphiScorer import *

events = read_json(f"/data/all_sequences")

parser = argparse.ArgumentParser(description="")
parser.add_argument("--total_section", type=int, default=20)
parser.add_argument("--section_id", type=int, default=0)
args = parser.parse_args()

num_section = args.total_section
section_id = args.section_id
num_event_section = int(len(events) / num_section)
print(f"num_section:{num_section}",
      f"section_id: {section_id}",
      f"num_event_section: {num_event_section}")

start_id = section_id * num_event_section
if section_id == num_section - 1:
    end_id = -1
else:
    end_id = (section_id + 1) * num_event_section

events_to_gen = events[start_id : end_id]
print(f"start_id:{start_id}, end_id: {end_id}")

model_name = "t5-11b-1239200"
device_id = 0
server = "beaker_batch"

delphi_generator = DelphiScorer(model=model_name, device_id=device_id, server=server)

data_to_save = {}
for i, event in enumerate(tqdm(events_to_gen)):
    class_label, probs, text_label = delphi_generator.generate_with_score(event)
    delphi_pred = {"class_label": class_label,
                   "prob_1": probs[0],
                   "prob_0": probs[1],
                   "prob_minus_1": probs[2],
                   "text_label": text_label}
    data_to_save[event] = delphi_pred

    output_path = f"/output/delphi_{section_id}.json"
    save_json(output_path, data_to_save)
