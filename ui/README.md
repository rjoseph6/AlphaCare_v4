# Skin Cancer vs. Non-Cancerous Skin Disease (Hierarchical Classification #2)

![](../ui/hierarchy_2.png)


In this second stage of the hierarchical skin disease detection project, we focus on building a model that differentiates between skin cancer and non-cancerous skin diseases.

## Table of Contents
1. [Overview](#overview)
2. [Dataset Preparation](#dataset)
3. [Directory Structure](#structure)
4. [Usage](#usage)
5. [Model Architecture](#models)
6. [Results](#results)

## Overview <a name="overview"></a>
adsfadsf


## Dataset Preparation <a name="dataset"></a>

### Skin Cancer Images

- HAM10000

The HAM10000 dataset is a collection of 10,015 dermatoscopic images of common pigmented skin lesions. The data consists of 7 classes of skin cancer, including melanoma, basal cell carcinoma, actinic keratoses, benign keratosis, dermatofibroma, and vascular lesions. The dataset is available at https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T.

```	
| Disease Class                                      | Code   | Number of Images | Cancerous/Non-Cancerous |
|----------------------------------------------------|--------|------------------|-------------------------|
| Actinic Keratoses and Intraepithelial Carcinoma    | akiec  | 327              | **Cancerous** |
| Basal Cell Carcinoma                                | bcc   | 514               | **Cancerous** |
| Benign Keratosis-like Lesions                       | bkl   | 1099              | Non-Cancerous |
| Dermatofibroma                                      | df    | 115               | Non-Cancerous |
| Melanoma                                            | mel   |  1113             | **Cancerous** |
| Melanocytic Nevi                                    | nv      | 6705              | Non-Cancerous |
| Vascular Lesions                                    | vasc  | 142                | Non-Cancerous |
```

I am missing 2003 images from the dataset. I have to check if the images are missing from the dataset or if they are not being loaded properly.

```python
Counts Before Adding HAM10000:
  Train - Cancer: 1708
  Validation - Cancer: 430
  Train - Non-Cancerous: 18699
  Validation - Non-Cancerous: 4886

Number of images added from HAM10000:
  Train - Cancer: 1241
  Validation - Cancer: 311
  Train - Non-Cancerous: 5168
  Validation - Non-Cancerous: 1292

Counts After Adding HAM10000:
  Train - Cancer: 2949
  Validation - Cancer: 741
  Train - Non-Cancerous: 23867
  Validation - Non-Cancerous: 6178
```

- SIIM-ISIC Melanoma Classification
https://www.kaggle.com/competitions/siim-isic-melanoma-classification/overview 

The dataset is roughly 32 GB. 





### Non-Cancerous Skin Disease Images
- DermNet

The data consists of images of 23 types of skin diseases taken from http://www.dermnet.com/dermatology-pictures-skin-disease-pictures. The total number of images are around 19,500, out of which approximately 15,500 have been split in the training set and the remaining in the test set.

The categories include acne, melanoma, Eczema, Seborrheic Keratoses, Tinea Ringworm, Bullous disease, Poison Ivy, Psoriasis, Vascular Tumors, etc.
https://www.kaggle.com/datasets/shubhamgoel27/dermnet 

```
| Condition Name                                                        | Training Data | Testing Data | Cancer/Non-Cancerous |
|---------------------------------------------------------------------- |---------------|--------------|----------------------|
| Urticaria Hives                                                      | 212           | 53           | Non-Cancerous        |
| Seborrheic Keratoses and other Benign Tumors                         | 1371          | 343          | Non-Cancerous        |
| Poison Ivy Photos and other Contact Dermatitis                       | 260           | 65           | Non-Cancerous        |
| Acne and Rosacea Photos                                              | 840           | 312          | Non-Cancerous        |
| Vascular Tumors                                                      | 482           | 121          | Non-Cancerous        |
| Eczema Photos                                                        | 1235          | 309          | Non-Cancerous        |
| Psoriasis pictures Lichen Planus and related diseases                | 1405          | 352          | Non-Cancerous        |
| Exanthems and Drug Eruptions                                         | 404           | 101          | Non-Cancerous        |
| Lupus and other Connective Tissue diseases                           | 420           | 105          | Non-Cancerous        |
| Scabies Lyme Disease and other Infestations and Bites                | 431           | 108          | Non-Cancerous        |
| Bullous Disease Photos                                               | 448           | 113          | Non-Cancerous        |
| Nail Fungus and other Nail Disease                                   | 1040          | 261          | Non-Cancerous        |
| Tinea Ringworm Candidiasis and other Fungal Infections               | 1300          | 325          | Non-Cancerous        |
| Systemic Disease                                                     | 606           | 152          | Non-Cancerous        |
| Light Diseases and Disorders of Pigmentation                         | 568           | 143          | Non-Cancerous        |
| Atopic Dermatitis Photos                                             | 489           | 123          | Non-Cancerous        |
| Warts Molluscum and other Viral Infections                           | 1086          | 272          | Non-Cancerous        |
| **Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions** | **1149**      | **288**      | **Cancer**           |
| **Melanoma Skin Cancer Nevi and Moles**                              | **463**       | **116**      | **Cancer**           |
| Vasculitis Photos                                                    | 416           | 105          | Non-Cancerous        |
| Cellulitis Impetigo and other Bacterial Infections                   | 288           | 73           | Non-Cancerous        |
| Hair Loss Photos Alopecia and other Hair Diseases                    | 239           | 60           | Non-Cancerous        |
| Herpes HPV and other STDs Photos                                     | 405           | 102          | Non-Cancerous        |
```

```python
# dataset pre-processing script
dermnet.py
```

We move the cancerous and non-cancerous images to separate folders and then split them into training and validation sets. The final directory structure is as follows:

```python
Counts Before Adding Dermnet:
  Train - Cancer: 0
  Validation - Cancer: 0
  Train - Non-Cancerous: 0
  Validation - Non-Cancerous: 0

Number of images added from Dermnet:
  Train - Cancer: 1612
  Validation - Cancer: 404
  Train - Non-Cancerous: 13579
  Validation - Non-Cancerous: 3491

Counts After Adding Dermnet:
  Train - Cancer: 1612
  Validation - Cancer: 404
  Train - Non-Cancerous: 13579
  Validation - Non-Cancerous: 3491
```

- SD-198 Dataset

Contains 6,584 clinical images spanning 198 skin disease classes. Got dataset from https://huggingface.co/datasets/resyhgerwshshgdfghsdfgh/SD-198/tree/main. 

```
| Disease Class | Number of Images | Cancerous/Non-Cancerous |
|---------------|------------------|-------------------------|
...
| Balanitis_Xerotica_Obliterans | 12 | Non-Cancerous |
| Basal_Cell_Carcinoma | 61 | **Cancerous** |
| Beau's_Lines | 35 | Non-Cancerous |
| Becker's_Nevus | 18 | Non-Cancerous |
...
| Keratoacanthoma | 61 | **Cancerous** |
...
| Xerosis | 39 | Non-Cancerous |
```

```python
Counts Before Adding SD-198:
  Train - Cancer: 1612
  Validation - Cancer: 404
  Train - Non-Cancerous: 13579
  Validation - Non-Cancerous: 3491

Number of images added from SD-198:
  Train - Cancer: 96
  Validation - Cancer: 26
  Train - Non-Cancerous: 5120
  Validation - Non-Cancerous: 1395

Counts After Adding SD-198:
  Train - Cancer: 1708
  Validation - Cancer: 430
  Train - Non-Cancerous: 18699
  Validation - Non-Cancerous: 4886
```