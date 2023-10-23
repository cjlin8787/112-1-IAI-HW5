# 112-1-IAI HW5
This repository is the template for the homework 5 of 人工智慧概論/Introduction to Artificial Intelligence. Department of Biomechatronics Engineering, National Taiwan University.

## Introduction
Recent advances in generative artificial intelligence (AI) have created many possibilities, but this new technology also poses many challenges to society. Currently, generative AI can generate highly realistic audio data. In this project, you will design a supervised learning prediction model to verify whether the given audio data is from a real recording or the result of AI generation.

## Prepare Training Data
1. Download and extract [sample.zip](https://drive.google.com/file/d/1LoZUIqasi6kd_yHtW3fZRUkZdRHLbmfm/view?usp=sharing).
2. Unzip `sample.zip`. The directory should look like the following:
```
sample
├── meta.txt
└── wavs
    ├── 0.wav
    ├── 1.wav
    ...
    └── 500.wav
```
3. The `meta` contains the path to each `wav` file (column 1), and the corresponding label (column 2). If the audio is generated from real recording, the label will be `0`. If the audio is generated from AI, the label will be `1`.

## Setup Environment
Please check the [Pytorch]() website if CUDA version needs to be downloaded.
```
conda create -n fastspeech python=3.8.0
conda activate fastspeech
conda install --file requirements.txt -c pytorch -c defaults -c anaconda
```

## Todo
* Design the prediction model, finish `def train(dataloader)` and `def predict(dataloader)` in `HW5Model` in `model.py`.
* If the model predict the audio is generated from real recording, output `0`. If the model predict the audio is generated from AI, output `1`.
* Note that the provided sample data is very imbalance, please check [FastSpeech-FloWaveNet](https://github.com/dn070017/FastSpeech-FloWaveNet) for data generation.
* Please check `dataset.py` for the definition of `Dataset`. `main.py` for the main training and prediction workflow.
* Please do not change anything except `def setup_model()`, `def train` and `def predict`. Please do not change the API (parameters and return values) of these function.
* When evalauting the homework, only the following instructions will be used:
```
test_dataset = HW5Dataset(data_path, 'test', 0.1)
test_loader = DataLoader(test_dataset, batch_size=1, pin_memory=True)

model = HW5Model()
model.setup_model()
model.load_state(f'{id}.pt')
model.predict(test_loader)
```