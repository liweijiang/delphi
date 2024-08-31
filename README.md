# An Empirical Investigation of Machines' Capabilities for Moral Judgment with the Delphi Experiment

This is the official repository for our preprint:

Liwei Jiang, Jena D. Hwang, Chandra Bhagavatula, Ronan Le Bras, Jenny Liang, Jesse Dodge, Keisuke Sakaguchi, Maxwell Forbes, Jon Borchardt, Saadia Gabriel, Yulia Tsvetkov, Oren Etzioni, Maarten Sap, Regina Rini, Yejin Choi. [An Empirical Investigation of Machines' Capabilities for Moral Judgment with the Delphi Experiment](https://arxiv.org/abs/2110.07574).

As our society adopts increasingly powerful AI systems for pervasive use, there are growing concerns about machine morality---or lack thereof. Millions of users already rely upon the outputs of AI systems, such as chatbots, as decision aids. Meanwhile, AI researchers continue to grapple with the challenge of aligning these systems with human morality and values. In response to this challenge, we build and test Delphi, an open-source AI system trained to predict human moral judgments. The computational framework of Delphi is grounded in the philosophical moral framework proposed by the prominent moral philosopher John Rawls. Our results speak to the promises and limits of machine's capabilities to learn about human morality. On the one hand, Delphi demonstrates improved generalization capabilities over those exhibited by off-the-shelf neural language models. At the same time, Delphi's failures also underscore important challenges in this arena. For instance, Delphi has limited cultural awareness and is susceptible to pervasive biases. Despite these shortcomings, we demonstrate several compelling use cases of Delphi, including incorporating it as a component within an ensemble of AI systems. Finally, we computationally demonstrate the potential of Rawls' prospect of hybrid approaches for reliable moral reasoning, inspiring future research in computational morality.


<img src=assets/overall.png width=1200/>

<img src=assets/norm_bank_content.png width=1200/>

<img src=assets/examples.png width=1200/>

<img src=assets/main_results.png width=1200/>

<img src=assets/bias_results.png width=1200/>

<img src=assets/delphi)hybrid.png width=1200/>


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

