import spacy
import coreferee

# doc = nlp('Although he was very busy with his work, Peter had had enough of it. He and his wife decided they needed a holiday. They travelled to Spain because they loved the country very much.')
# doc = nlp("killing a bear if it attacks you")
# doc = nlp("Killing a bear if it walks towards you but it wouldn't be able to approach you in a zoo")
# doc = nlp("Kill a bear when it walks towards you and it would not approach you enough to hurt you")

class Coreferee:

    def __init__(self):
        self.nlp = spacy.load('en_core_web_trf')
        self.nlp.add_pipe('coreferee')


    def _get_resolved_mention_text(self, resolved_mentions):
        """
            Helper function to convert a list of objects into a text
            E.g., [bear, wolf, bird] => "bear, wolf, and bird"
        """
        resolved_mention_text = resolved_mentions[0].text
        for mention in resolved_mentions[1:-1]:
            resolved_mention_text += ", "
            resolved_mention_text += mention.text
        if len(resolved_mentions) > 1:
            resolved_mention_text += " and "
            resolved_mention_text += resolved_mentions[-1].text
        return resolved_mention_text


    def coreference_resolution(self, input_sequence):
        """
            Resolve coreference ambiguity
        """
        doc = self.nlp(input_sequence)
        doc_list = [t.text for t in doc]

        ##### print the coreference chain #####
        # doc._.coref_chains.print()

        for coref_chain in doc._.coref_chains:
            for mentions in coref_chain:
                if len(mentions) == 1:
                    mention_index = mentions.root_index
                    resolved_mentions = doc._.coref_chains.resolve(doc[mention_index])

                    if resolved_mentions != None:
                        resolved_mention_text = self._get_resolved_mention_text(resolved_mentions)
                        doc_list[mention_index] = resolved_mention_text

        return " ".join(doc_list)


if __name__ == "__main__":
    coreferee = Coreferee()
    # premise = "Kill a bear when it walks towards you and it would not approach you enough to hurt you"
    # premise = "Kill a bear if it attacks you" # , a wolf and a bird
    # premise = "Kill a bear, a wolf and a bird if they attack you"
    premise = "Killing a bear if it walks towards you in a zoo"
    resolved_premise = coreferee.coreference_resolution(premise)
    print(resolved_premise)
