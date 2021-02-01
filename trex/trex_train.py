# -*- coding: utf-8 -*-

import glob
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import warnings
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

imgs = glob.glob("./img/*.png")

width = 125
height = 50

X = []
y = []

for i in imgs:
    
    filename = os.path.basename(i)
    label = filename.split("_")[0]
    im = np.array(Image.open(i).convert("L").resize((width, height)))
    im = im/255
    X.append(im)
    y.append(label)
    
X = np.array(X)
X = X.reshape(X.shape[0], width, height, 1)
    
    
sns.countplot(y)
    


def one_hot_labels(values):
    
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse = False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded
    
    
Y = one_hot_labels(y)
train_X,test_X,train_Y,test_Y = train_test_split(X,Y, test_size = 0.25, random_state = 3)

# CNN Model Building

model = Sequential()

model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", input_shape = (width, height, 1)))
model.add(Conv2D(64, kernel_size = (3,3), activation = "relu"))

model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())

# classifier

model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.4))
model.add(Dense(3, activation = "softmax"))

# if your model exists just call this lines beyond this code just train don't call those code lines
#if os.path.exists("./trex_weight.h5"):
#   model.load_weights("trex_weight.h5")
#  print("Model has been uploaded")

model.compile(loss = "categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])
earlyStopping = EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=25)

model.fit(train_X, train_Y, epochs = 34, batch_size=64,callbacks=[earlyStopping])

train_score = model.evaluate(train_X, train_Y)
print("Training accuracy %",train_score[1]*100)
test_score = model.evaluate(test_X, test_Y)
print("Test accuracy %",test_score[1]*100)
    
    
open("model.json","w").write(model.to_json())
model.save_weights("trex_weight.h5")   
    
modelLoss = pd.DataFrame(model.history.history)
modelLoss.plot();

   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    