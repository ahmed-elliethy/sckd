# Supervised Contrastive Knowledge Distillation
## Table of Contents

1. [Introduction](#introduction)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [License](#license)
6. [Contact](#contact)
7. [Reference](#Reference)
## Introduction

SCKD is a framework designed to facilitate knowledge distillation using supervised contrastive learning. The repository includes Python scripts for:

1. Evaluating model performance: Assessing the accuracy and other metrics of the distilled student models.
2. Training the teacher network: Creating the original teacher model.
3. Training the projector: Extracting informative features from the teacher model.
4. Training the student: Transferring knowledge from the teacher to the student.
5. Defining architectures: Providing the architectures of the projector and regressor.
6. Implementing supervised contrastive loss: The loss function used to train the projector.
7. The repository also includes pre-trained student model weights for CIFAR-10 and CIFAR-100 datasets.

## Repository Structure
```
SCKD/
├── evaluate_model.py                   # Script to evaluate model accuracy.
├── utils.py                            # Helper functions.
├── train_teacher.py                    # Used to train the Teacher.
├── feature_projection.py               # Contains the architectures of the Projector and the regressor.
├── sup_con_loss.py                     # Contains the supervised COntrastive loss function.
├── train_projector.py                  # Used to train the Projector.
├── train_student.py                    # Used to train the Student and its regressor.
├── models                              # Model architectures.
└── README.md                           # Project documentation.
```
#### 1- evaluate_model.py

Calculates the accuracy of a specified model on the CIFAR-10 or CIFAR-100 datasets.

#### 2- utils.py

Contains various helper functions required across the different processes.

#### 3- train_teacher.py

This script is used to train the Teacher model on the CIFAR-10 or CIFAR-100 datasets.

#### 4- feature_projection.py

Contains the architectures for both the projector and the regressor.

#### 5- sup_con_loss.py

Defines the supervised contrastive loss function, which is used to train the projector.

#### 6- train_projector.py

This script is used to train the Projector on the CIFAR-10 or CIFAR-100 datasets.

#### 7- train_student.py

This script is used to train the Student model on the CIFAR-10 or CIFAR-100 datasets using SCKD. 

#### 8- models

Directory containing all the model architectures used during the knowledge distillation process.

## Installation

To set up the SCKD repository, follow these steps:

### Clone the repository:

```
git clone https://github.com/ahmed-elliethy/SCKD.git
```

## Usage
### evaluate_model.py:
To evaluate a model's accuracy, use the evaluate_model.py script. You need to specify the dataset, the model name, and the path to the model's weights.
You can download the weights of our results from this link: https://drive.google.com/drive/folders/1dI9yEv82_iylNkRT_j5EQ4PS_lQqUiac?usp=sharing.

### Example command:

```
python evaluate_model.py --dataset cifar10 --model_t resnet20 --model_path save/resnet20.pth
```
This command evaluates the accuracy of the resnet20 model on the CIFAR-10 dataset using the saved weights from save/resnet20.pth.

### utils.py
All necessary helper functions are located in the utils.py file, which is automatically imported when using the other python scripts.

### train_teacher.py
This script is used to train the Teacher model on the CIFAR-10 or CIFAR-100 datasets.

### Example command:

```
python train_student.py --dataset cifar10 --model resnet56
```
This command trains resnet56 and outputs its baseline accuracy. We can you the resulted model as a baseline for our comparisons or as teacher model. The resulted model is saved in save/models/cifar10/resnet56_cifar10/resnet56_best.pth

### feature_projection.py
Contains the architectures for both the projector and the regressor. This file is called automatically when needed.

### sup_con_loss.py
Defines the supervised contrastive loss function, which is used to train the projector. This file is called automatically when needed.

### train_projector.py
This script is used to train the Projector on the CIFAR-10 or CIFAR-100 datasets.

### Example command:

```
python train_projector.py --dataset cifar10 --path_t save/models/cifar10/resnet56_cifar10/resnet56_best.pth
```
When using this script, you should specify the dataset and provide the path to the teacher model's weights. The trained projector will be saved in /save/projector/cifar10/projector_resnet56_cifar10_SCKD/.

### train_student.py
This script is used to train the Student model on the CIFAR-10 or CIFAR-100 datasets using SCKD. 

### Example command:

```
python train_student.py --dataset cifar10 --model_s resnet20 --path_t "./save/models/cifar10/resnet56_cifar10/resnet56_best.pth" --Projector_path "./save/projector/cifar10/projector_resnet56_cifar10_SCKD/projector_model.pth" --beta 80 --few_shot 1
```
this command trains a ResNet20 student model using SCKD on CIFAR-10. It leverages the knowledge from a pre-trained ResNet56 teacher model and a projector network specifically trained for this task. The hyperparameters beta and few-shot are set to specific values for this particular experiment.

1. --beta 80: This sets the beta hyperparameter for the SCKD knowledge distillation technique, currently set to 80. This value controls the balance between the student's own predictions and the teacher's guidance.
2. --few_shot 1: This configures the training for a few-shot learning scenario where only one sample per class is available during training.
Note:
In case of different families you will have to set the values of kernel_size, stride and padding manually.

## License
All rights reserved.

This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the terms of the license, as specified in the document LICENSE.txt (included in this package)
## Contact
For any questions or inquiries, please reach out to:





## Reference
```
@article{Elshazly_2024_sckd,
  title={Supervised Contrastive Knowledge Distillation},
  author={Elshazly, A., Elliethy, A. and Elshafey, M.A},
  journal={International Journal of Intelligent Engineering and Systems}
}
https://www.researchgate.net/publication/384568099_Supervised_Contrastive_Knowledge_Distillation
```
