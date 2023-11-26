# 112-1-IAI HW5
This repository is the template for the homework 5 of 人工智慧概論/Introduction to Artificial Intelligence. Department of Biomechatronics Engineering, National Taiwan University.

## Introduction
Recent advances in generative artificial intelligence (AI) have created many possibilities, but this new technology also poses many challenges to society. Currently, generative AI can generate highly realistic audio data. In this project, you will design a supervised learning prediction model to verify whether the given audio data is from a real recording or the result of AI generation.

## Prepare Training Data
1. Download and extract [train_dataset.zip](https://drive.google.com/file/d/1GztoCT0Hjmt-Yqw-6rfiwu2QYkRI1uI5/view?usp=drive_link).
2. Unzip `sample.zip`. The directory should look like the following:
```
train_dataset
├── meta.csv
└── wavs
    ├── 0.wav
    ├── 1.wav
    ...
    └── 500.wav
dataset.py
main.py
model.py
README.md
requirements.txt
```
3. The `meta` contains the path to each `wav` file (column 1), and the corresponding label (column 2). If the audio is a real recording, the label will be `0`. If the audio is generated from AI, the label will be `1`.

## Setup Environment
Please check the [Pytorch](https://pytorch.org/) website if CUDA version needs to be downloaded.
```
conda create -n fastspeech python=3.8.2
conda activate fastspeech
conda install --file requirements.txt -c pytorch -c defaults -c anaconda -c conda-forge
```

## Todo
* Design the prediction model to distinguish if the provided recoding is real recording or is generated from AI.
* You will have to finish the following or the corresponding TODO block in [colab](https://colab.research.google.com/drive/1RgNJxyG_9LoePk95NdBzU9oi-ChUyi-M?usp=sharing):
    * `main.py`:
        * [5 Points] create `train_dataset, val_dataset`
        * [5 Points] create `train_loader, val_loader`
    * `HW5Model` of `model.py`.
        * [35 Points] finish `def train_epochs(dataloader)`.
        * [10 Points] finish `def predict_prob(dataloader)`.
        * [10 Points] finish `def predict(dataloader)`.
        * [10 Points] finish `def evaluate(y_true, y_pred)`.
    * Note that the provided sample data is very imbalance, please check [FastSpeech-FloWaveNet](https://github.com/cjlin8787/FastSpeech-FloWaveNet) for data generation.
        * [10 Points] setup environment and generate at least 5 audio (2 points for each file) based on your custom text prompt.
        * Please put the audio you generated under `train_dataset/wavs/`. The directory should look like the following
 ```
train_dataset
├── meta.csv
└── wavs
    ├── 0.wav
    ├── 1.wav
    ...
    ├── 500.wav
    ├── 0_flowavenet_auio.wav
    ├── 1_flowavenet_auio.wav
    ├── 2_flowavenet_auio.wav
    ...
```
    * Model evaluation on held-out test set.
        * 30 × (Accuracy - 0.5).
* Please check `dataset.py` for the definition of `Dataset`. `main.py` for the main training and prediction workflow.
* Please do not change anything except `def setup_model()`, `def train_epochs()`, `def pred_prob()`,
 `def predict()`, and `def evaluate()`. Please do not change the API (parameters and return values) of these function.
* When evaluating the homework, only the following instructions will be used:
```
test_dataset = HW5Dataset('test_dataset/meta.csv')
test_loader = # Create from test_dataset

model = HW5Model(hidden_size=hidden_size, num_layers=num_layers)

# If you directly clone this github repo
model.load_state_dict(torch.load("best_model.ckpt"))
# If you use colab and submit the .ipynb
model.load_state_dict(torch.load("/content/best_model.ckpt"))

y_pred_prob = model.predict_prob(test_loader)
y_pred = model.predict(test_loader)
y_true = torch.concat([labels for mel, labels in val_loader]).numpy().astype('float32')
print(f'Accuracy on test set: {(y_pred == y_true).sum() / len(y_true):.2f}')
print(f'Area under precision recall curve on test set: {model.evaluate(y_true, y_pred_prob):.2f}')
```