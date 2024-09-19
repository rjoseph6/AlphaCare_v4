# ReadME

![](ui/ui_stepper.png)

![](ui/hierarchy.png)

```
backend
    ├── app.py                     # inference
    ├── model_4.keras               # model weights
    ├── requirements.txt
frontend
    ├── public
    │   ├── ...
    ├── src
    │   ├── App.css 
    │   ├── App.js
    │   ├── App.test.js
    │   └── index.js
    ├── package.json
    ├── package-lock.json
step_1                                  # Skin or Non-Skin
    ├── inference_skin_nonskin.ipynb               
    ├── preprocess_coco.py          
    ├── preprocess_skin.py
    ├── skin_nonskin_classifier.py
    ├── README.md
step_2                                  # Malignant or Benign
    ├── cancer_noncancer_classifier.py              
    ├── ham10000_eda.ipynb          
    ├── ham10000.py
    ├── README_2.md
step_3                                  # Melanoma, Basal Cell Carcinoma,Squamous Cell Carcinoma
    ├── ham10000_malignant.py              
    ├── malignant_train.py         
    ├── README.md
ui
    ├── ...                     
```



# UI Designs

```
Detailed discussion in the UI.md file
```

## v1
![ui](ui/ui_v1.png)

## v2

![ui](ui/ui_v2.png)

Figma: https://www.figma.com/design/NdtFBlTj6g0E0xhGH18msF/SearchGPT?node-id=0-1&node-type=CANVAS&t=kYT1Qnh3UBI5Y6Kd-0 


# Frontend 
Using React 

# Backend
Previously I used the FastAPI to create API Layer. I switched to Flask for the backend due to the ease of use. 

# Models
For all 3 of the hierarchial classifications I use the same model architecture. The model is the pretrained ResNet18 Model


# Bugs
1. Make sure using Global Environment Python Version 3.11 not the local version 3.9
2. Make sure you are always inside the skin-cancer-detector directory when launching frontend or backend