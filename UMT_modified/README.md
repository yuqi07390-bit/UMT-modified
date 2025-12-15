# BERT-Enhanced UMT Architecture (ARIN 5201 Group Project)

## Group Members
- DING, Lianli(21215270)
- GUO, Yixin(21249324)
- WANG, Yuqi(21205720)
- SU, Likai(21205562)


## Requirements

- CUDA 11.3
- Python 3.10.19
- PyTorch 1.11.0
- [NNCore](https://github.com/yeliudev/nncore) 0.3.6
- Numpy 1.24.1

## Install packages

```
pip install -r requirements.txt
```

## Getting Started

### Download and prepare the datasets

1. Download and extract the datasets.

- [Charades-STA](https://huggingface.co/yeliudev/UMT/resolve/main/datasets/charades-2c9f7bab.zip)


2. Prepare the files in the following structure.

```
UMT
├── configs
├── datasets
├── models
├── tools
├── data
│   └── charades
│       ├── *features
│       └── charades_sta_{train,test}.txt
├── README.md
├── setup.cfg
└── ···
```

### Train a model

```shell
# Original Implementation
python tools/launch.py configs/charades/umt_base_vo_100e_charades.py

# Our modification
python tools/launch.py configs/charades/umt_base_vo_100e_charades_clipencoder_train_nofreeze.py
```

### Test a model and evaluate results

```
# For original implementation
python tools/launch.py configs/charades/umt_base_vo_100e_charades_eval.py --checkpoint ${path-to-checkpoint} --eval


# For out modification 
python tools/launch.py configs/charades/umt_base_vo_100e_charades_clipencoder_eval.py --checkpoint ${path-to-checkpoint} --eval
```

Checkpoints are under "./work_dirs" folder.

