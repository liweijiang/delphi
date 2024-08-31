import sys
sys.path.append(".")

import torch
from scipy.special import softmax
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config


class DelphiScorer:
    def __init__(self, device_id=1, model="t5-11b-1239200", server="beaker_batch"):
        CUDA_DEVICE = f"cuda:{device_id}" if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(CUDA_DEVICE)
        print(f"DelphiScorer device: {self.device}", model)

        if model == "t5-large":
            MODEL_BASE = "t5-large"
            MODEL_LOCATION = "/net/nfs.cirrascale/mosaic/liweij/model/large_commonsense_morality_hf"
            # MODEL_LOCATION = "/net/nfs2.corp/mosaic/home/ronanlb/t5_models/large_commonsense_morality_hf"
            self.class_token_pos = 4
            self.sep_tokens = ["<unk> /class> <unk> text>", " class>", "<unk> /text>"]

        elif model == "t5-11b":
            MODEL_BASE = "t5-11b"
            MODEL_LOCATION = "/net/nfs.cirrascale/mosaic/liweij/model/11b_commonsense_morality_hf"
            self.class_token_pos = 4
            self.sep_tokens = ["<unk> /class> <unk> text>", " class>", "<unk> /text>"]

        elif model == "t5-11b-1239200":
            MODEL_BASE = "t5-11b"
            # MODEL_LOCATION = "/net/nfs2.corp/mosaic/home/ronanlb/t5_models/11b_commonsense_morality_v4_hf"
            if server == "beaker_batch":
                MODEL_LOCATION = "/model/delphi11b"
            else:
                MODEL_LOCATION = "/net/nfs.cirrascale/mosaic/liweij/model/11b_commonsense_morality_1239200_hf"
            self.class_token_pos = 4
            self.sep_tokens = ["<unk> /class> <unk> text>", " class>", "<unk> /text>"]

        elif model == "v11_distribution":
            MODEL_BASE = "t5-11b"
            MODEL_LOCATION = "/net/nfs.cirrascale/mosaic/liweij/model/v11_distribution_hf"
            self.class_token_pos = 3
            self.sep_tokens = ["[/class] [text]", "[class]", "[/text]"]

        elif model == "11B_001":
            MODEL_BASE = "t5-11b"
            MODEL_LOCATION = "/net/nfs.cirrascale/mosaic/liweij/model/11B_001_hf"
            self.class_token_pos = 4
            self.sep_tokens = ["<unk> /class> <unk> text>", " class>", "<unk> /text>"]

        else:
            print("ERROR: model doesn't exist")
            return

        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_LOCATION)
        self.model.to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_BASE, model_max_length=512)

    def score(self, input_string, normalize=None):
        input_string = f"[moral_single]: {input_string}"
        input_ids = self.tokenizer(input_string, return_tensors='pt').to(self.device).input_ids
        outputs = self.model.generate(input_ids, max_length=200, output_scores=True, return_dict_in_generate=True)

        probs = [(self.tokenizer.decode(i), x) for (i, x) in enumerate(outputs['scores'][self.class_token_pos][0].softmax(0))]

        class1_prob = sum([v[1].item() for v in probs if v[0] == "1"])
        class0_prob = sum([v[1].item()  for v in probs if v[0] == "0"])
        classminus1_prob = sum([v[1].item()  for v in probs if v[0] == "-1"])

        probs = [class1_prob, class0_prob, classminus1_prob]
        probs_sum = sum(probs)

        if normalize == "regular":
            probs = [p / probs_sum for p in probs]
        elif normalize == "softmax":
            probs = softmax(probs)

        return probs

    def generate(self, input_string):
        input_string = f"[moral_single]: {input_string}"
        input_ids = self.tokenizer(input_string, return_tensors='pt').to(self.device).input_ids
        outputs = self.model.generate(input_ids, max_length=200, output_scores=True, return_dict_in_generate=True)

        decoded_sequence = self.tokenizer.decode(outputs["sequences"][0])
        class_label = int(decoded_sequence.split(self.sep_tokens[0])[0].split(self.sep_tokens[1])[-1])
        text_label = decoded_sequence.split(self.sep_tokens[0])[-1].split(self.sep_tokens[2])[0]

        return class_label, text_label


    def generate_beam(self,
                      input_string,
                      num_beams=5,
                      max_length=50,
                      num_return_sequences=5,):
        input_string = f"[moral_single]: {input_string}"
        input_ids = self.tokenizer(input_string,
                                   max_length=16,
                                   truncation=True,
                                   return_tensors='pt').to(self.device).input_ids
        outputs = self.model.generate(input_ids,
                                      # output_scores=True,
                                      # return_dict_in_generate=True,
                                      num_beams=num_beams,
                                      max_length=max_length,
                                      num_return_sequences=num_return_sequences,)

        decoded_sequences = self.tokenizer.batch_decode(outputs)

        class_labels = [ds.split(self.sep_tokens[0])[0].split(self.sep_tokens[1])[-1] for ds in decoded_sequences]
        text_labels = [ds.split(self.sep_tokens[0])[-1].split(self.sep_tokens[2])[0] for ds in decoded_sequences]

        return class_labels, text_labels


    def generate_with_score(self, input_string):
        input_string = f"[moral_single]: {input_string}"
        input_ids = self.tokenizer(input_string, max_length=512, return_tensors='pt').to(self.device).input_ids
        outputs = self.model.generate(input_ids, max_length=200, output_scores=True, return_dict_in_generate=True)

        probs = [(self.tokenizer.decode(i), x) for (i, x) in enumerate(outputs['scores'][self.class_token_pos][0].softmax(0))]

        class1_prob = sum([v[1].item() for v in probs if v[0] == "1"])
        class0_prob = sum([v[1].item()  for v in probs if v[0] == "0"])
        classminus1_prob = sum([v[1].item()  for v in probs if v[0] == "-1"])

        probs = [class1_prob, class0_prob, classminus1_prob]
        # probs_sum = sum(probs)

        decoded_sequence = self.tokenizer.decode(outputs["sequences"][0])
        class_label = int(decoded_sequence.split(self.sep_tokens[0])[0].split(self.sep_tokens[1])[-1])
        text_label = decoded_sequence.split(self.sep_tokens[0])[-1].split(self.sep_tokens[2])[0]

        return class_label, probs, text_label


    def generate_with_score_comparison(self, action1, action2):
        input_string = f"[moral_pair]: <action1>{action1}</action1> <action2>{action2}</action2>"
        input_ids = self.tokenizer(input_string, return_tensors='pt').to(self.device).input_ids
        outputs = self.model.generate(input_ids, max_length=200, output_scores=True, return_dict_in_generate=True)

        probs = [(self.tokenizer.decode(i), x) for (i, x) in enumerate(outputs['scores'][0][0].softmax(0))]

        class1_prob = sum([v[1].item() for v in probs if v[0] == "1"])
        class2_prob = sum([v[1].item()  for v in probs if v[0] == "2"])

        probs = [class1_prob, class2_prob]
        # probs_sum = sum(probs)

        class_label = int(self.tokenizer.decode(outputs["sequences"][0], skip_special_tokens=True))
        # print(class1_prob, class2_prob, class_label)

        return class_label, probs


if __name__ == '__main__':
    # delphi_scorer = DelphiScorer(device_id=0, model="t5-large")
    delphi_scorer = DelphiScorer(device_id=0, model="t5-11b-1239200")
    # print(delphi_scorer.generate_with_score("killing a cat"))
    # e_list = ["killing a cat",
    #           "killing a dog",
    #           "killing a child"]
    #
    # for e in e_list:
    #     print(delphi_scorer.generate_with_score(e))

    print(delphi_scorer.generate_with_score_comparison("killing a cat", "killing an evil cat"))


    # for i in range(5):
    #     print(delphi_scorer.generate_beam("Gay parents adopting a child"))\

    # for seq in ["Gay parents adopting a child", "quentin tarantino", "Can I kill a terrorist?", "Protecting China"]:
    # seq =
    # print(delphi_scorer.generate(seq))

