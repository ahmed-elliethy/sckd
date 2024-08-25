# Supervised Contrastive Knowledge Distillation

## Table of Contents

1. [Introduction](#introduction)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [License](#license)
6. [Reference](#Reference)
## Introduction

SCKD is designed to facilitate knowledge distillation using supervised contrastive learning. The repository includes Python scripts for evaluating model performance, as well as saved weights from student models that have been distilled from teacher models. It supports CIFAR-10 and CIFAR-100 datasets.

## Repository Structure
```
SCKD/
├── evaluate_model.py     # Script to evaluate model accuracy
├── utils.py              # Helper functions
├── models/               # Model architectures
└── README.md             # Project documentation
```
#### 1- evaluate_model.py

Calculates the accuracy of a specified model on either the CIFAR-10 or CIFAR-100 dataset.

#### 2- utils.py

Contains various helper functions necessary for the evaluation and other processes.

#### 3- models/

Directory containing all the model architectures used in the distillation process.

## Installation

To set up the SCKD repository, follow these steps:

### Clone the repository:

```
git clone https://github.com/your-username/SCKD.git
```

## Usage
To evaluate a model's accuracy, use the evaluate_model.py script. You need to specify the dataset, the model name, and the path to the model's weights.
You can download the weights of our results from this link: https://drive.google.com/drive/folders/1dI9yEv82_iylNkRT_j5EQ4PS_lQqUiac?usp=sharing.

### Example command:

```
python evaluate_model.py --dataset cifar10 --model_t resnet20 --model_path save/resnet20.pth
```
This command evaluates the accuracy of the resnet20 model on the CIFAR-10 dataset using the saved weights from save/resnet20.pth.

### Helper Functions
All necessary helper functions are located in the utils.py file, which is automatically imported when running evaluate_model.py.


### Full source code
To be published ...

## License
All rights reserved.

This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the terms of the license, as specified in the document LICENSE.txt (included in this package)


## Reference
**Under review**
```
@article{Elshazly_2024_sckd,
  title={Supervised Contrastive Knowledge Distillation},
  author={Elshazly, A., Elliethy, A. and Elshafey, M.A},
  journal={International Journal of Intelligent Engineering and Systems},
  notes={under review}
}
```


# License

All rights reserved.

This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the terms of the license, as specified in the document LICENSE.txt (included in this package)

## Installation
The code requires Python 3.x and PyTorch 1.12.

To install Python 3.x for Ubuntu, you can run:

```
apt-get update
apt-get install -y python3.8 python3.8-dev python3-pip python3-venv
```

To install PyTorch, follow the link here https://pytorch.org

## How to use
To be published


## Reference
**Under review**
```
@article{Elshazly_2024_sckd,
  title={Supervised Contrastive Knowledge Distillation},
  author={Elshazly, A., Elliethy, A. and Elshafey, M.A},
  journal={International Journal of Intelligent Engineering and Systems},
  notes={under review}
}
```
