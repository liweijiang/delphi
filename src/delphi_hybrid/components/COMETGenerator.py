import os
import sys
import torch
from treelib import Node, Tree
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

sys.path.append(os.getcwd())
from src.delphi_hybrid.components.bank import *


class COMETGenerator():
    def __init__(self, model_name="gpt2-xl-atomic2020", device_id=0, server="beaker"): # beaker_batch, local
        if model_name == "gpt2-xl-atomic2020":
            if server == "beaker_batch":
                base_path_atomic_2020 = "/model/atomic2020"
            else:
                base_path_atomic_2020 = "/net/nfs.cirrascale/mosaic/liweij/model/atomic2020"

            CUDA_DEVICE = f"cuda:{device_id}" if torch.cuda.is_available() else 'cpu'
            self.device = torch.device(CUDA_DEVICE)
            print(f"COMETGenerator device: {self.device}")
            self.model_name = model_name
            self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-xl")
            self.model = GPT2LMHeadModel.from_pretrained(base_path_atomic_2020,
                                                         pad_token_id=self.tokenizer.eos_token_id).to(self.device)
            self.add_special_tokens()

            self.BOS_TOKEN = self.tokenizer.bos_token
            self.EOS_TOKEN = self.tokenizer.eos_token
            self.GEN_TOKEN = "[GEN]"

    def add_special_tokens(self):
        self.tokenizer.add_special_tokens({
            'eos_token': '[EOS]',
            'additional_special_tokens': [
                'LocationOfAction',
                'HinderedBy',
                'HasFirstSubevent',
                'NotHasProperty',
                'NotHasA',
                'HasA',
                'AtLocation',
                'NotCapableOf',
                'CausesDesire',
                'HasPainCharacter',
                'NotDesires',
                'MadeUpOf',
                'InstanceOf',
                'SymbolOf',
                'xReason',
                'isAfter',
                'HasPrerequisite',
                'UsedFor',
                'MadeOf',
                'MotivatedByGoal',
                'Causes',
                'oEffect',
                'CreatedBy',
                'ReceivesAction',
                'NotMadeOf',
                'xWant',
                'PartOf',
                'DesireOf',
                'HasPainIntensity',
                'xAttr',
                'DefinedAs',
                'oReact',
                'xIntent',
                'HasSubevent',
                'oWant',
                'HasProperty',
                'IsA',
                'HasSubEvent',
                'LocatedNear',
                'Desires',
                'isFilledBy',
                'isBefore',
                'InheritsFrom',
                'xNeed',
                'xEffect',
                'xReact',
                'HasLastSubevent',
                'RelatedTo',
                'CapableOf',
                'NotIsA',
                'ObjectUse',
                '[GEN]'
            ]
        })
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))

    def __name__(self):
        return self.model_name

    def generate(self, head_event, relation):
        input_string = head_event + " " + relation + " [GEN]"
        input_ids = self.tokenizer(input_string, return_tensors='pt').to(self.device).input_ids
        outputs = self.model.generate(input_ids, max_length=200, output_scores=True, return_dict_in_generate=True)

        decoded_sequence = self.tokenizer.decode(outputs["sequences"][0])
        tail_event = decoded_sequence.split(self.GEN_TOKEN)[-1].split(self.EOS_TOKEN)[0]
        return tail_event

    def generate_beam(self, head_event, relation, num_beams=5, num_return_sequences=5, max_length=100):
        input_string = head_event + " " + relation + " [GEN]"
        tokenized_input = self.tokenizer(input_string,
                                         max_length=max_length,
                                         truncation=True,
                                         return_tensors='pt').to(self.device)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        outputs = self.model.generate(input_ids=tokenized_input.input_ids,
                                      attention_mask=tokenized_input.attention_mask,
                                      # output_scores=True,
                                      # return_dict_in_generate=True,
                                      num_beams=num_beams,
                                      max_length=max_length,
                                      num_return_sequences=num_return_sequences,)

        decoded_sequences = self.tokenizer.batch_decode(outputs)

        tail_events = [ds.split(self.GEN_TOKEN)[-1].split(self.EOS_TOKEN)[0] for ds in decoded_sequences]

        return tail_events

    def generate_all_relations(self, event):
        comet_inferences = {}
        for relation in comet_relations:
            tail_events = self.generate_beam(event, relation)
            comet_inferences[relation] = tail_events
        return comet_inferences

if __name__ == "__main__":
    comet_generator = COMETGenerator(device_id=0)

    head_events = ["a bear",
                   "being a stupid bear",
                   "performing genocide",
                   "a protected bear"]

    # for head_event in head_events:
    for relation in comet_relations:
        tail_events = comet_generator.generate_beam(head_events[0], relation)
        