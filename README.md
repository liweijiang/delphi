# An Empirical Investigation of Machines' Capabilities for Moral Judgment with the Delphi Experiment

This is the official repository for our preprint:

Liwei Jiang, Jena D. Hwang, Chandra Bhagavatula, Ronan Le Bras, Jenny Liang, Jesse Dodge, Keisuke Sakaguchi, Maxwell Forbes, Jon Borchardt, Saadia Gabriel, Yulia Tsvetkov, Oren Etzioni, Maarten Sap, Regina Rini, Yejin Choi. [An Empirical Investigation of Machines' Capabilities for Moral Judgment with the Delphi Experiment](https://arxiv.org/abs/2110.07574).

> As our society adopts increasingly powerful AI systems for pervasive use, there are growing concerns about machine morality---or lack thereof. Millions of users already rely upon the outputs of AI systems, such as chatbots, as decision aids. Meanwhile, AI researchers continue to grapple with the challenge of aligning these systems with human morality and values. In response to this challenge, we build and test Delphi, an open-source AI system trained to predict human moral judgments. The computational framework of Delphi is grounded in the philosophical moral framework proposed by the prominent moral philosopher John Rawls. Our results speak to the promises and limits of machine's capabilities to learn about human morality. On the one hand, Delphi demonstrates improved generalization capabilities over those exhibited by off-the-shelf neural language models. At the same time, Delphi's failures also underscore important challenges in this arena. For instance, Delphi has limited cultural awareness and is susceptible to pervasive biases. Despite these shortcomings, we demonstrate several compelling use cases of Delphi, including incorporating it as a component within an ensemble of AI systems. Finally, we computationally demonstrate the potential of Rawls' prospect of hybrid approaches for reliable moral reasoning, inspiring future research in computational morality.

## The Theoretical and Computational Frameworks of Delphi

 <details close>
  <summary><b>Details of the Figure:</b></summary>

 > (a) The theoretical framework of ethics proposed by the prominent moral philosopher John Rawls. In 1951, Rawls proposed a “decision procedure of ethics” that takes a bottom-up approach to capture patterns of human ethics via crowd- sourcing moral opinions of a wide variety of people. Later in 1971, Rawls complemented the theoretial procedure with top-down constraints in his most famous work, A Theory of Justice. Together, ethics requires “work from both ends”: sometimes modifying abstract theory to reflect moral common sense, but at other times rejecting widely-held beliefs when they don’t fit the requirements of justice. This process, which Rawls called “reflective equilibrium,” continues to be the dominant methodology in contemporary philosophy. (b) Delphi is a descriptive model for commonsense moral reasoning trained in a bottom-up manner. Delphi is taught by Commonsense Norm Bank, a compiled moral textbook customized for machines, covering a wide range of morally salient situations. Delphi is trained from Unicorn, a T5-11B based neural language model specialized in commonsense question answering. Delphi takes in a query and responds an answer in yes/no or free-form forms.

</details>

<img src=assets/overall.png width=800/>

<details close>
  <summary><b>Overview of Commonsense Norm Bank Content:</b></summary>

> Representative N-grams cover topics including people, relationships, actions, life & society, cognition, and others. The lemmatized and normalized 4-grams used for the topic analysis are bolded. Auxiliary words from the original form of data instances that are not used in the topics analysis are unbolded.

<img src=assets/norm_bank_content.png width=600/>

</details>

<details close>
  <summary><b>Examples from Delphi:</b></summary>

> Delphi shows impressive ability to generalize to unseen situations beyond Commonsense Norm Bank, and is robust to adjust its judgment against changing contexts. Colors of labels indicate Delphi’s classification results (green: positive, gray: neutral, red: negative). Textual labels come from Delphi’s open-text responses.

<img src=assets/examples.png width=800/>

</details>


## Main Results of Delphi

 <details close>
  <summary><b>Details of the Figure:</b></summary>

 > (a) Delphi achieves better performance on Norm Bank comparing to GPT-3 baselines. (b) Comparing the effect of the size of the base T5 model. (c) Ablation results showing the scale of training data improves Delphi’s learning. (d) Ablation results showing the compositionality of training instances improves Delphi’s learning. (e) Delphi, with minimal supervisions, outperforms baseline models on hate speech detection under both in-distribution and out-of-distribution settings. (g) Plugging Delphi into language generation models helps improve the prosocial implication scores of the generated stories, without sacrificing the language quality. (g) Delphi outperforms other baselines on transferring knowledge to specific theoretically motivated moral frameworks.

</details>

<img src=assets/main_results.png width=800/>


## Social Bias Evaluation Results of Delphi

 <details close>
  <summary><b>Details of the Figure:</b></summary>

 > (a) Results for the Universal Declaration of Human Rights probing, including top identities that Delphi shows biases against and their level of biases, and the average % error for each identity group. (b) Delphi and Delphi+’s performance under current-world and ideal-world settings. Statistical significance test is performed between Delphi under the current-world compared to other models or settings.

</details>

<img src=assets/bias_results.png width=800/>

## An Illustration of the Delphi-Hybrid Framework and an Example Output Moral Constraint Graph


 <details close>
  <summary><b>Details of the Figure:</b></summary>

> (a) A hybrid system that incorporates an optional symbolically guided rea- soning mechanism to complement the neural language model based Delphi. (b) An example of the moral constraint graph produced by Delphihybrid for the event “Mass genocide for greater good.” Nodes denote judgments derived either from top-down moral principles or bottom-up Delphi. Edges denote logical violations (i.e., identity, entailment, and contradiction) between nodes. ❌ denotes inconsistent nodes identified by the constrained optimization step. Note that each top-down moral principle may result in multiple nodes depending on engineering details (e.g., the same rule “Do not kill” applied at the full event level or constituent level). The final judgment is negative.

</details>


<img src=assets/delphi_hybrid.png width=800/>


## Codebase Structure

This codebase contains the training and evaluation code for Delphi, Delphi+, and the Delphi-Hybrid system.

### Delphi:

- `src/delphi/evaluate`: scripts for evaluating the Delphi models on the yes/no QA and freeform QA tasks, as well as downstream tasks like Hate Speech Detection.

- `src/delphi/train`: the scripts for finetuning T5 for Delphi.

### Delphi+:

- `src/delphi_plus/evaluate`: scripts for evaluating the Delphi+ models on the yes/no QA and freeform QA tasks, as well as downstream tasks like Hate Speech Detection.

- `src/delphi/train`: the scripts for finetuning T5 for Delphi+.

### Delphi-Hybrid:

- `src/delphi_plus/collective_reasoning`: codebase for the collective reasoning component of Delphi-Hybrid.

- `src/delphi/components`: components of the Delphi-Hybrid system.

- `src/delphi/prepare_data`: scripts for preparing test data for Delphi-Hybrid experiments.

### Datasheet:

- `data/datasheet.md`: the datasheet for the Commonsense Norm Bank dataset.

## Data and Model Access
You can access the Commonsense Norm Bank dataset by [filling out this form](https://forms.gle/VoAVuPUJFNChWhSj8).

For accessing the Delphi model checkpoints and API calls please feel free to reach out to Liwei Jiang at [lwjiang@cs.washington.edu](lwjiang@cs.washington.edu).

If you find our paper or data useful, please cite the paper:
```
@article{jiangdelphi2022,
      title={Can Machines Learn Morality? The Delphi Experiment}, 
      author={Liwei Jiang and Jena D. Hwang and Chandra Bhagavatula and Ronan Le Bras and Jenny Liang and Jesse Dodge and Keisuke Sakaguchi and Maxwell Forbes and Jon Borchardt and Saadia Gabriel and Yulia Tsvetkov and Oren Etzioni and Maarten Sap and Regina Rini and Yejin Choi},
      year={2022},
      eprint={2110.07574},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

