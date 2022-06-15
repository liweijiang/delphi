# Commonsense Norm Bank Datasheet
This is the datasheet for the Commonsense Norm Bank using the protocol defined by [Gebru et al.](https://arxiv.org/pdf/1803.09010.pdf). This document was last updated on March 31, 2022.

## 1. Motivation

### A. For what purpose was the dataset created? Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.
The Commonsense Norm Bank was created to explicitly teach large language models to be aligned with human values. We define three tasks to do this:
- **Free-form mode**: Free-form mode is the task of given a real-life situation, providing a moral judgment to the situation.
- **Yes/no mode**: Yes/no mode is the task of given a moral judgment (e.g., "women cannot be scientists", "it's kind to express concern over your neighbor's friends"), determining whether society at large would agree or disagree with the statement.
- **Relative mode**: Relative mode is the task of given two situations, determining which situation is morally better over another. **We exclude this portion of the dataset from the Commonsense Norm Bank release because it was not trained with data that alleviate stereotypes or biased point of views towards social and demographic groups that are conventionally underrepresented when applying ethical judgments. Thus, this task is prone to problematic and harmful output. Thus, this task is especially prone to problematic and harmful output.** However, if you would like to have access to this dataset, please contact [delphi@allenai.org](mailto:delphi@allenai.org). 

### B. Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?
[Liwei Jiang](https://liweijiang.me/) created this dataset on behalf of the [xlab](https://homes.cs.washington.edu/~yejin/) at the [Paul G. Allen School of Computer Science](https://www.cs.washington.edu/) and the [Mosaic team](https://mosaic.allenai.org/) at the [Allen Institute for Artificial Intelligence](https://allenai.org/).

### C. Who funded the creation of the dataset? If there is an associated grant, please provide the name of the grantor and the grant name and number.
This work was partially supported by the [Allen Institute for Artificial Intelligence](https://allenai.org/). TODO: Liwei: Any grants from UW?

### D. Any other comments?
None.

## 2. Composition

### A. What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.
The instances from **free-form mode** represents moral judgments of real-life situations, **yes/no mode** represents general societal agreement to moral judgments, and relative mode represents a moral comparison between two situations. We describe the columns of each task in section 2D.

### B. How many instances are there in total (of each type, if appropriate)?
There are 1.7 million instances in the Commonsense Norm Bank in total. Here is the breakdown for the number of instances:
- **Free-form mode**: 1,136,568 instances
- **Yes/no mode**: 477,514 instances
- **Relative mode**: 28,296 instances

### C. Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).
This dataset contain all possible instances from the Commonsense Norm Bank.

### D. What data does each instance consist of? “Raw” data (e.g., unprocessed text or images) or features? In either case, please provide a description.
We describe the columns of each task below.

#### Free-form Mode
- `input_sequence`: A moral situation
- `class_label`: The moral judgment of the situation: good (1), bad (-1), discretionary (0)
- `text_label`: An open text moral judgment
- `input_type`: Whether `input_sequence` is a positive rule-of-thumb (positive_rot) or a negative rule-of-thumb (negative_rot)*
- `format`: The format template used to generate `input_sequence`
- `source`: The data set the situation originates from

* For more information on rule-of-thumbs, please refer to the [Social Chemistry paper](https://arxiv.org/abs/2011.00620).

#### Yes/no Mode
- `input_sequence`: A moral judgment
- `class_label`: The moral judgment of the situation: good (1), bad (-1), discretionary (0)
- `text_label`: An open text moral judgment
- `input_type`: Whether `input_sequence` is a positive rule-of-thumb (positive_rot) or a negative rule-of-thumb (negative_rot)*
- `source`: The data set the judgment originates from

* For more information on rule-of-thumbs, please refer to the [Social Chemistry paper](https://arxiv.org/abs/2011.00620).

#### Relative Mode
Relative mode is the task of given two situations, determining which situation is morally better over another. For more information on the relative mode task, please refer to the [Aligning AI with Shared Human Values](https://arxiv.org/pdf/2008.02275.pdf) paper by Hendryks et al.

This data set contains the following columns: 

- `action_1`: A moral situation
- `action_2`: A moral situation different than `action_1`
- `targets`: Which of the moral situations (1 or 2) is morally better

### E. Is there a label or target associated with each instance? If so, please provide a description.
There is a label or target associated with all instances in the Commonsense Norm Bank. We list each task's target column names below.

- **Free-form mode**: `class_label`, `text_label`
- **Yes/no mode**: `class_label`, `text_label`
- **Relative mode**: `targets`

### F. Is any information missing from individual instances? If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.
No information is missing from individual instances.

### G. Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)? If so, please describe how these relationships are made explicit.
There are no relationships between individual instances.

### H. Are there recommended data splits (e.g., training, development/validation, testing)? If so, please provide a description of these splits, explaining the rationale behind them.
The data is split into train, validation, and test sets, which are denoted by the file prefix (i.e., `train.`, `validation.`, `test.`). These splits were generated at random. TODO: Liwei---what percentage of each data is in the split?

### I. Are there any errors, sources of noise, or redundancies in the dataset? If so, please provide a description.
There may be some sources of noise as instances in the Commonsense Norm Bank were created using generic templates. This may introduce nstances that are grammatically incorrect or hard to follow.

### J. Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)? If it links to or relies on external resources, a) are there guarantees that they will exist, and remain constant, over time; b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a dataset consumer? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate.
The data relies on external resources. Commonsense Norm Bank is compiled from five existing datasets. We describe the datasets and their statuses to the best of our knowledge below.

- [Social Chemsitry](https://maxwellforbes.com/social-chemistry/) is a large-scale corpus formalizing people’s social norms and moral judgments over a rich spectrum of everyday situations described in natural language. There are no guarantees that these external resources will exist and remain constant over time. There are no official archival versions of the complete dataset. There are no restrictions associated with the external resources.
- [ETHICS Commonsense Morality](https://arxiv.org/pdf/2008.02275.pdf) is a benchmark assessing language models’ ability to predict fundamental human ethical judgments based on a complex function of implicit morally salient factors. There are no guarantees that these external resources will exist and remain constant over time. There are no official archival versions of the complete dataset. The dataset is [licensed](https://github.com/hendrycks/ethics/blob/master/LICENSE) under the MIT license, which limits liability and provides no warranty.
- [Moral Stories](https://arxiv.org/pdf/2012.15738.pdf) is a corpus of structured narratives for the study of grounded, goal-oriented, and morally-informed social reasoning. There are no guarantees that these external resources will exist and remain constant over time. There are no official archival versions of the complete dataset. There are no restrictions associated with the external resources.
- [Social Bias Inference Corpus](https://homes.cs.washington.edu/~msap/social-bias-frames/) is a corpus of instances capturing conceptual formalism that aims to model the pragmatic frames in which people project social or demographic biases and stereotypes onto others. There are no guarantees that these external resources will exist and remain constant over time. There are no official archival versions of the complete dataset. There are no restrictions associated with the external resources.
- [Scruples](https://www.aaai.org/AAAI21Papers/AAAI-6406.LourieN.pdf) is a large-scale dataset of ethical judgments over real-life anecdotes (i.e., complex situations with moral implications). There are no guarantees that these external resources will exist and remain constant over time. There are no official archival versions of the complete dataset. The dataset is [licensed](https://github.com/hendrycks/ethics/blob/master/LICENSE) under the Apache license, which limits trademark use and liability as well as provides no warranty.

### K. Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor– patient confidentiality, data that includes the content of individuals’ non-public communications)? If so, please provide a description.
No, the dataset does not contain data that might be considered confidential.

### L. Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? If so, please describe why.
Yes, the dataset contains data that, if viewed directly, may be offensive, insulting, threatening, or might otherwise cause anxiety. Generally, the Commonsense Norm Bank contains moral judgments on everyday situations, which may thus include lewd or offensive content. As an example, the Social Bias Inference Corpus, which this dataset relies on, contains utterances which may be offensive, lewd, and directed at specific identity groups.

## 3. Collection Process

### A. How was the data associated with each instance acquired? Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)? If the data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.
Not applicable---Commonsense Norm Bank was compiled from five existing datasets (see Section 2J).

### B. What mechanisms or procedures were used to collect the data (e.g., hardware apparatuses or sensors, manual human curation, software programs, software APIs)? How were these mechanisms or procedures validated?
Not applicable---Commonsense Norm Bank was compiled from five existing datasets (see Section 2J).

### C. If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?
The dataset is not a sample from a larger dataset.

### D. Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?
Not applicable---Commonsense Norm Bank was compiled from five existing datasets (see Section 2J).

### E. Over what timeframe was the data collected? Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)? If not, please describe the timeframe in which the data associated with the instances was created.
Not applicable---Commonsense Norm Bank was compiled from five existing datasets (see Section 2J).

### F. Were any ethical review processes conducted (e.g., by an institutional review board)? If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation.
This study was approved by an Institutional Review Board. TODO: Liwei: Information on this.

## 4. Preprocessing/cleaning/labeling

### A. Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. If not, you may skip the remaining questions in this section.
Yes, pre-processing of the data was done to all of the datasets that Commonsense Norm Bank relies on. For `free-form` mode, each instance is formulated based on a specific template as defined by the `format` column. We applied these transformations in the Commonsense Norm Bank. We generated the target labels differently for each dataset, which we detail below.

- **Social Chemistry**: The class label is sourced from the ethical social judgment attribute as the categorical judgment label (3-way classification of good, discretionary, bad). The text label comes from the rule-of-thumb as the open-text judgment label.
- **ETHICS Commonsense Morality**: The class label is from binary categorical moral judgment from each scenario. The text label is randomly chosen from hand-crafted text judgments based on the class label.
- **Moral Stories**: The class label is derived from whether the action is moral or immoral based on the dataset. The text label is randomly chosen from hand-crafted text judgments based on the class label.
- **Social Bias Inference Corpus**: The class label is derived from whether the instance is labeled as offensive or lewd. The text label is randomly chosen from hand-crafted text judgments based on the class label.
- **Scruples**: No modification required for the target labels.

### B. Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)? If so, please provide a link or other access point to the “raw” data.
The raw data was not saved, but may be accessed directly from the original datasets, which are publicly available at the following links: [Social Chemistry](https://maxwellforbes.com/social-chemistry/), [ETHICS Commonsense Morality](https://github.com/hendrycks/ethics), [Moral Stories](https://github.com/demelin/moral_stories), [Social Bias Inference Corpus](https://homes.cs.washington.edu/~msap/social-bias-frames/), [Scruples](https://github.com/allenai/scruples).

### C. Is the software that was used to preprocess/clean/label the data available? If so, please provide a link or other access point.
No, the software is not available.

### D. Any other comments?
None.

## 5. Uses

### A. Has the dataset been used for any tasks already? If so, please provide a description.
Yes, it has been used for three tasks: **free-form mode**, **yes/no mode**, and **relative mode**. For descriptions of each task, please refer to Section 1A.

### B. Is there a repository that links to any or all papers or systems that use the dataset? If so, please provide a link or other access point.
No, there is not a repository that links to any or all papers or systems that use this dataset.

### C. What (other) tasks could the dataset be used for?
TODO: Liwei: Can you answer this?

### D. Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? For example, is there anything that a dataset consumer might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other risks or harms (e.g., legal risks, financial harms)? If so, please provide a description. Is there anything a dataset consumer could do to mitigate these risks or harms?
Yes, the composition of the dataset may impact future uses. The **free-form mode** and **yes/no mode** tasks were trained on Social Bias Frames to mitigate harms against minoritized communities, but language models trained on Commonsense Norm Bank may still emit utterances that reflect social biases and stereotypes or generate harmful and offensive language. The **relative mode** task was not trained on any dataset to mitigate harms against minoritized communities, so this task is especially prone to generating harmful and biased language.

Dataset consumers can reduce harm by not using this dataset for commercial purposes as stipulated in our license. For additional information on the license, refer to Section 6D. Additionally, to reduce potential harm against groups, we do not recommend training models with only the **relative mode** data as-is.

### E. Are there tasks for which the dataset should not be used? If so, please provide a description.
Yes. Generally, this dataset should not be used for commercial purposes as stipulated in our license. For additional information on the license, refer to Section 6D.

### F. Any other comments?
None.

## 6. Distribution

### A. Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? If so, please provide a description.
Yes, the dataset is available to the public. For distribution details, please refer to Section 6B.

### B. How will the dataset will be distributed (e.g., tarball on website, API, GitHub)? Does the dataset have a digital object identifier (DOI)?
The dataset is distributed via a [Google Forms survey](https://forms.gle/DxFR71PFajBFgJANA). Individuals enter their email and the Google Forms will return a link to the online drive containing Commonsense Norm Bank. 

### C. When will the dataset be distributed?
The dataset is publicly available as of May 2022.

### D. Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)? If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.
Yes, the dataset is distributed under a copyright. We use the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). Under this license, the Commonsense Norm Bank cannot be used for commercial purposes. Usage of the dataset requires attribution. Material that is adapted from the Commonsense Norm Bank can be shared with attribution.

### E. Have any third parties imposed IP-based or other restrictions on the data associated with the instances? If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these restrictions.
Yes, third parties have imposed IP-based restrictions on the data. ETHICS Commonsense Morality is [licensed](https://github.com/hendrycks/ethics/blob/master/LICENSE) under the Apache license, which limits trademark use and liability as well as provides no warranty. Scruples is [licensed](https://github.com/hendrycks/ethics/blob/master/LICENSE) under the Apache license, which limits trademark use and liability as well as provides no warranty.

### F. Do any export controls or other regulatory restrictions apply to the dataset or to individual instances? If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any supporting documentation.
No, no export controls or regulatory restrictions apply to the dataset.

### G. Any other comments?
None.

## 7. Maintenance

### A. Who will be supporting/hosting/maintaining the dataset?
The dataset will be maintained by the Delphi study team at the [Allen Institute for Artificial Intelligence](https://allenai.org/).

### B. How can the owner/curator/manager of the dataset be contacted (e.g., email address)?
The dataset manager can be contacted via [delphi@allenai.org](mailto:delphi@allenai.org).

### Is there an erratum? If so, please provide a link or other access point.
No, there is not an erratum.

### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)? If so, please describe how often, by whom, and how updates will be communicated to dataset consumers (e.g., mailing list, GitHub)?
The dataset will not be updated.

### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were the individuals in question told that their data would be retained for a fixed period of time and then deleted)? If so, please describe these limits and explain how they will be enforced.
Not applicable---Commonsense Norm Bank does not contain data related to individuals.

### Will older versions of the dataset continue to be supported/hosted/maintained? If so, please describe how. If not, please describe how its obsolescence will be communicated to dataset consumers
Not applicable---the dataset will not be updated.

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to dataset consumers? If so, please provide a description.
There is no mechanism to extend, augment, build on, or contribute to the dataset.

### Any other comments?
None.