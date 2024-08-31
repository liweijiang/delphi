import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification


class WANLIScorer:

	def __init__(self):
		self.model = RobertaForSequenceClassification.from_pretrained('alisawuffles/roberta-large-wanli')
		self.tokenizer = RobertaTokenizer.from_pretrained('alisawuffles/roberta-large-wanli')

	def get_scores(self, premise, hypothesis, is_return_score=True):
		x = self.tokenizer(premise, hypothesis, hypothesis, return_tensors='pt', max_length=128, truncation=True)
		logits = self.model(**x).logits
		probs = logits.softmax(dim=1).squeeze(0)
		label_id = torch.argmax(probs).item()
		prediction = self.model.config.id2label[label_id]

		if is_return_score:
			scores = {"contradiction": probs[0].item(),
					  "entailment": probs[1].item(),
					  "neutral": probs[2].item()}
			return scores
		return prediction


if __name__ == "__main__":
	wanli_scorer = WANLIScorer()
	# premise = "killing a bear to save your child."
	# hypothesis = "killing a bear."
	# prediction = wanli_scorer.get_scores(premise, hypothesis)
	# print(premise, "|", hypothesis, ":", prediction)
	#
	# hypothesis = "killing a child."
	# prediction = wanli_scorer.get_scores(premise, hypothesis)
	# print(premise, "|", hypothesis, ":", prediction)
	#
	# hypothesis = "save your child."
	# prediction = wanli_scorer.get_scores(premise, hypothesis)
	# print(premise, "|", hypothesis, ":", prediction)
	#
	# hypothesis = "bear your child."
	# prediction = wanli_scorer.get_scores(premise, hypothesis)
	# print(premise, "|", hypothesis, ":", prediction)
	#
	# hypothesis = "bear."
	# prediction = wanli_scorer.get_scores(premise, hypothesis)
	# print(premise, "|", hypothesis, ":", prediction)
	#
	# hypothesis = "save."
	# prediction = wanli_scorer.get_scores(premise, hypothesis)
	# print(premise, "|", hypothesis, ":", prediction)
	#
	# hypothesis = "killing."
	# prediction = wanli_scorer.get_scores(premise, hypothesis)
	# print(premise, "|", hypothesis, ":", prediction)


	premise = "Killing a bear if I attack it."
	hypothesis = "Killing a bear that attacks you"
	prediction = wanli_scorer.get_scores(premise, hypothesis)
	print(premise, "|", hypothesis, ":", prediction)

	hypothesis = "Killing a bear if I attack it."
	premise = "Killing a bear that attacks you"
	prediction = wanli_scorer.get_scores(premise, hypothesis)
	print(premise, "|", hypothesis, ":", prediction)
