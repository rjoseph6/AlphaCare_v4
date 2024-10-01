import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

# Constants
IMG_SIZE = 224

# directories
data_dir = '../datasets/aptos2019/'
img_dir_train = os.path.join(data_dir, 'train_images')
img_dir_test = os.path.join(data_dir, 'test_images')

train_df = pd.read_csv(data_dir + 'train.csv')
test_df = pd.read_csv(data_dir + 'test.csv')
print(f"Train: {train_df.head()}\n")
print(f"Test: {test_df.head()}")

# shape
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

#train_df['diagnosis'].value_counts().plot(kind='pie')
train_df['diagnosis'].value_counts().plot(kind='bar')
plt.title('Diagnosis')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.savefig('class_count_bar.png')
#plt.show()

train_x = train_df['id_code']
train_y = train_df['diagnosis']

'''fig = plt.figure(figsize=(25, 16))
# display 10 images from each class
for class_id in sorted(train_y.unique()):
    for i, (idx, row) in enumerate(train_df.loc[train_df['diagnosis'] == class_id].sample(5, random_state=12).iterrows()):
        ax = fig.add_subplot(5, 5, class_id * 5 + i + 1, xticks=[], yticks=[])
        #path=f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png"
        path = os.path.join(data_dir, 'train_images', f"{row['id_code']}.png")
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        #print(f"Image shape: {image.shape}")
        #print(f"Image: {image}")
        plt.imshow(image)
        ax.set_title('Label: %d' % (class_id) )'''

#plt.savefig('img_classes.png')
#plt.show()