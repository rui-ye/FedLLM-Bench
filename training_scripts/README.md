# Training Scripts

We provide all necessary training scripts for our four datasets. Before using the bash scripts to train your models, you may need to adjust some parameters as needed for your setup.

To learn more about the key arguments of the training scripts, please refer to [OpenFedLLM](https://github.com/rui-ye/OpenFedLLM).

## TODO
- `gpu` : Adjust this parameter according to your GPU specifications.
- `fed_alg` : Modify this parameter to train models with different federated algorithms. Available algorithms include **local0, fedavg, fedprox, fedavgm, scaffold, fedyogi, fedadam, fedadagrad, feddp**.
- ```model_name_or_path``` : We recommend downloading the base model and setting this parameter to the path of your local base model.  Alternatively, you can provide the model name to fetch the model from Hugging Face. We use [Llama2-7B](https://huggingface.co/meta-llama/Llama-2-7b) for instruction tuning and [Alpaca-7B](https://github.com/tatsu-lab/stanford_alpaca) for preference alignment.


