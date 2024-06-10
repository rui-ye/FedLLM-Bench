# FedLLM-Bench: Realistic Benchmarks for Federated Learning of Large Language Models

**FedLLM-Bench** is the first realistic benchmark for FedLLM community, which is a follow-up of the [OpenFedLLM](https://arxiv.org/abs/2402.06954) framework. Please check our [paper](https://arxiv.org/pdf/2406.04845) for details and the corresponding empirical study.

FedLLM-Bench includes the following key features:
- 3 datasets for **federated instruction tuning** tasks (i.e., *Fed-Aya*, *Fed-ChatbotIT*, and *Fed-WildChat*).
- 1 dataset for **federated preference alignment** task (i.e., *Fed-ChatbotPA*).
- **Diversities** covering *language*, *quality*, *quantity*, *instruction*, *sequence length*, *embedding*, and *preference*.

## Overview
A summary of our four realistic FedLLM datasets. IT denotes instruction tuning and PA denotes preference alignment. # denotes ‘the number of’ and L. denotes ‘the length of’. Our datasets
exhibit diversities in characteristic, task, client number, quantity, length, and quality
![](./assets/2024-06-10_165701.jpg)

## Dataset
The dataset can be downloaded at [data](https://drive.google.com/file/d/1hKv5A0ROmTQQkcsTcYogCUIeF7Ux1pmy/view?usp=sharing). After unzipping the data files, please place it in the "data" directory in the project.

## Citation

Please cite our paper if you find the repository helpful.

```
@article{ye2024fedllm,
  title={FedLLM-Bench: Realistic Benchmarks for Federated Learning of Large Language Models},
  author={Ye, Rui and Ge, Rui and Zhu, Xinyu and Chai, Jingyi and Du, Yaxin and Liu, Yang and Wang, Yanfeng and Chen, Siheng},
  journal={arXiv preprint arXiv:2406.04845},
  year={2024}
}
```
and
```
@article{ye2024openfedllm,
  title={OpenFedLLM: Training Large Language Models on Decentralized Private Data via Federated Learning},
  author={Ye, Rui and Wang, Wenhao and Chai, Jingyi and Li, Dihan and Li, Zexi and Xu, Yinda and Du, Yaxin and Wang, Yanfeng and Chen, Siheng},
  journal={arXiv preprint arXiv:2402.06954},
  year={2024}
}
```
