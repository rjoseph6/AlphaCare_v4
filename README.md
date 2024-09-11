# ReadME


![](frontend/v1_frontend.png) 

## Project Structure
```
backend
    ├── main.py                     # model inference
    ├── model_3.keras               # test model
    ├── requirements.txt
frontend
    ├── public
    │   ├── ...
    ├── src
    │   ├── App.css 
    │   ├── App.js
    │   ├── App.test.js
    │   ├── components
    │   │   ├── ImageUploader.js
    │   │   ├── Results.js
    │   └── index.js
```

# How To Run 
1. Open the terminal and run the following command: 

![](ui/terminal.png)

```bash
cd Downloads 
```
```bash
git clone https://github.com/rjoseph6/alphacare_v3.git
```
We are downloading the project from the repository.

2. Run these commands to start the frontend:

```bash
cd alphacare_v3/frontend
```

```bash
cd frontend/
```

```bash
npm start
```

4. A browser window should open with the frontend running. 

![ui](frontend/v1_frontend2.png)

5. In the second terminal, run these commands to start the backend

```bash
# go into app
cd Downloads/alphacare_v3/backend # YOUR PATH

# create virtual environment
python3 -m venv venv

# activate virtual environment
source venv/bin/activate

# install requirements
pip3 install -r requirements.txt

# run python script
python3 app.py

```

6. The frontend (browser) should now be connected to the backend (python script). 

7. Input an image of a skin disease (Melanoma/Vascular Lesion) and click the "Submit" button. The model will predict the image and display the results.

![ui](frontend/v1_frontend.png)


# Next Meeting

TODO:
- [ ] Add a loading spinner
- [ ] Add a button to clear the image

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

```bash
# ---- TERMINAL #1 ----

# go into react app 
cd desktop/Masters_Project/skin-cancer-detector

# run react app
npm start
```

# Backend
Previously I used the FastAPI to create API Layer. I switched to Flask for the backend due to the ease of use. 

```bash
# ---- TERMINAL #2 ----

# go into app
cd desktop/alphacare_v3/backend

# create virtual environment
python3 -m venv venv
# checking version
# python3 --version

# activate virtual environment
source venv/bin/activate

# install requirements
pip3 install -r requirements.txt

# run python script
python3 app.py

```

# Models
## Testing Model 
This model was quickly trained on a very easy dataset (HAM10000). It is just a test model to see if the backend and frontend are working. 

```python
# model weights
model_3.keras
```

## Official Model
Model design decisions were made based on the following:
https://github.com/Tirth27/Skin-Cancer-Classification-using-Deep-Learning

```python
# pretrained model EfficientNetB4
base_model = EfficientNetB4(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

# Adding custom layers
x = base_model.output
x = Flatten()(x)
x = Dense(500, activation='relu', name='500_Dense')(x)
x = Dense(256, activation='relu', name='256_Dense')(x)
predictions = Dense(NUM_CLASSES, activation='softmax', name='Final')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Unfreeze the top 20 layers, except BatchNorm layers
for layer in model.layers[-20:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True  # Keep True to train, False to freeze

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```



# Bugs
1. Make sure using Global Environment Python Version 3.11 not the local version 3.9
2. Make sure you are always inside the skin-cancer-detector directory when launching frontend or backend