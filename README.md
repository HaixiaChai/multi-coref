# Investigating Multilingual Coreference Resolution by Universal Annotations

This repository contains the code and trained model from the paper ["Investigating Multilingual Coreference Resolution by Universal Annotations"](https://arxiv.org/pdf/2310.17734.pdf).

## Basic Setup
Set up environment and data for training and evaluation:
* Install Python3 dependencies: `pip install -r requirements.txt`
* Create a directory for data that will contain all data files, models and log files; set `data_dir = /path/to/data/dir` in [experiments.conf](experiments.conf)
* Prepare dataset (requiring [CorefUD 1.0](https://ufallab.ms.mff.cuni.cz/~popel/CorefUD-1.0-public.zip) corpus):
* `python preprocess.py [config]`
  * e.g. `python preprocess.py train_mbert_czech`

## Evaluation

The name of each directory corresponds with a **configuration** in [experiments.conf](experiments.conf). Each directory has two trained models inside.

If you want to use the official evaluator, download and unzip [corefUD scorer](https://cs.emory.edu/~lxu85/conll-2012.zip) under this directory.

Evaluate a model on the dev/test set:
* Download the corresponding model directory and unzip it under `data_dir`
* `python evaluate.py [config] [model_id] [gpu_id]`
    * e.g. Attended Antecedent:`python evaluate.py train_spanbert_large_ml0_d2 May08_12-38-29_58000 0`

Download our [trained model](https://drive.google.com/file/d/1IGbSucxmekQrUQv6F81-NIfL5HPgYtmf/view?usp=sharing).

## Training
`python run.py [config] [gpu_id]`

* [config] can be any **configuration** in [experiments.conf](experiments.conf)
* Log file will be saved at `your_data_dir/[config]/log_XXX.txt`
* Models will be saved at `your_data_dir/[config]/model_XXX.bin`
* Tensorboard is available at `your_data_dir/tensorboard`


## Configurations
Some important configurations in [experiments.conf](experiments.conf):
* `data_dir`: the full path to the directory containing dataset, models, log files
* `bert_pretrained_name_or_path`: the name/path of the pretrained BERT model ([HuggingFace BERT models](https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained))
* `max_training_sentences`: the maximum segments to use when document is too long.

## Results

|                  | F1   |
|------------------|------|
| catalan          | 55.7 |
| czech-pcedt      | 68.5 |
| czech-pdt        | 64.9 |
| english-gum      | 50.1 |
| hungarian        | 47.1 |
| polish           | 50.4 |
| spanish          | 57.7 |
| lithuanian       | 62.1 |
| french           | 58.6 |
| german-parcor    | 35.1 |
| german-potsdamcc | 44.9 |
| english-parcor   | 48.5 |
| russian          | 66.5 |
| avg              | 54.6 |

## Citation
```
@article{chai2023investigating,
  title={Investigating Multilingual Coreference Resolution by Universal Annotations},
  author={Chai, Haixia and Strube, Michael},
  journal={arXiv preprint arXiv:2310.17734},
  year={2023}
}
```
