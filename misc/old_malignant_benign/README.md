# Malignant vs. Benigh (Hierarchical Classification #3)

![](../ui/hierarchy_3.png)


In this third stage of the hierarchical skin disease detection project, we focus on building a model that differentiates between MALIGNANT skin cancer and BENIGN skin cancer.

## Table of Contents
1. [Overview](#overview)
2. [Dataset Preparation](#dataset)
3. [Directory Structure](#structure)
4. [Usage](#usage)
5. [Model Architecture](#models)
6. [Results](#results)


## Dataset Preparation <a name="dataset"></a>
The dataset used was the most recent Kaggle Competition dataset on skin cancer detection, https://www.kaggle.com/competitions/isic-2024-challenge. The dataset has an astounishing 401,059 images. However after some interesting Exploratory Data Analysis I found that all these images came from only 1042 patients. The average patient had 385 images in the dataset (STRANGE). An even more striking issue was the class imbalance in the dataset. Of the 2 classes in the dataset malignant and benign, the malignant class had only 393 images while the benign class had 400,666 images. This class imbalance was a major issue that needed to be addressed.

![](../ui/class_imbalance_malignant.png)

## Model Architecture <a name="models"></a>

I decided to use a pre-trained model, specifically ResNet18. 