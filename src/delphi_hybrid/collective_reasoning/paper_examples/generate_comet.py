import os
import sys
from tqdm import tqdm
import argparse
import pandas as pd

sys.path.append(os.getcwd())

from scripts.utils.utils import *
from scripts.utils.COMETGenerator import *

events = read_json(f"/data/all_sequences")

parser = argparse.ArgumentParser(description="")
parser.add_argument("--section_id", type=int, default=0)
parser.add_argument("--total_section", type=int, default=10)
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

model_name = "gpt2-xl-atomic2020"
device_id = 0
server = "beaker_batch"

comet_generator = COMETGenerator(model_name=model_name, device_id=device_id, server=server)

all_comet_inferences = {}
for i, event in enumerate(tqdm(events_to_gen)):
    comet_inferences = {}
    for relation in comet_relations:
        tail_events = comet_generator.generate_beam(event, relation)
        comet_inferences[relation] = tail_events
    all_comet_inferences[event] = comet_inferences

    output_path = f"/output/comet_{section_id}.json"
    save_json(output_path, all_comet_inferences)
