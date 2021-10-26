from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
#from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

INIT_LR = 1e-4 
EPOCHS = 3
BS = 32

classes = "mask", "no_mask"

#dataset =  "./Cutted_Training_Image"
dataset = r"C:\Users\pinoc\Desktop\cap_new\tensorflow_mobilenetv2\content\face-mask-detector\dataset"
print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset))
data = []
labels = []
# loop over the image paths
for imagePath in imagePaths:
 # extract the class label from the filename
 label = imagePath.split(os.path.sep)[-2]
 # load the input image (224x224) and preprocess it
 image = load_img(imagePath, target_size=(224, 224))
 image = img_to_array(image)
 image = preprocess_input(image)
 # update the data and labels lists, respectively
 data.append(image)
 labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)


print("------------1-----------")
print(labels)
print("-----------------------")
#lb = LabelBinarizer() #여기서 
#labels = lb.fit_transform(labels) # 이것때문에 2차원으로 줄어들어서 문제가 생기는 듯?
lb = preprocessing.LabelEncoder()
labels = lb.fit_transform(labels)

labels = to_categorical(labels)

print("------------2-----------")
print(labels)
print("-----------------------")
print(data.shape)
print(labels.shape)
print("-------------3----------")
print(labels)
print("-----------------------")

"""n, nx, ny ,sth= data.shape
d2_data = data.reshape((n*nx*ny*sth))


#이미지를 잡지 못해서 생긴다 하나만 잡네
la_x, la_y= labels.shape
d2_labels = data.reshape((la_x*la_y)*75264)  
#근데 왜 18816을 곱해야 하는거지? 디멘션 변화 시 필요한 맞추기인가?
"""

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

aug = ImageDataGenerator(
 rotation_range=20,
 zoom_range=0.15,
 width_shift_range=0.2,
 height_shift_range=0.2,
 shear_range=0.15,
 horizontal_flip=True,
 fill_mode="nearest")

baseModel = MobileNetV2(weights="imagenet", include_top=False,
 input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)  #모델의 클래스 수가 늘어나면 이 값도 늘어나야 한다.

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
 layer.trainable = False

print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


print("[INFO] training head...")
H = model.fit(
 aug.flow(trainX, trainY, batch_size=BS),
 steps_per_epoch=len(trainX) // BS,
 validation_data=(testX, testY),
 validation_steps=len(testX) // BS,
 epochs=EPOCHS)

print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testY.argmax(axis=1), predIdxs,
 target_names=lb.classes_))

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('fig1.png', dpi=300)

print("[INFO] saving mask detector model...")
model.save('1022_mask.h5')



"""
import tensorflowjs as tfjs
from tensorflow.keras.models import load_model

def keras2tfjs(model_path,dir_out):
 #import tensorflowjs as tfjs
 MODEL_PATH = model_path
 print('Model loading...')
 model=load_model(MODEL_PATH)
 #print('Model loaded. Started serving...')
 tfjs.converters.save_keras_model(model, dir_out)

model_path='/content/mask-no-mask.h5'
dir_out='/content/keras2js'
keras2tfjs(model_path,dir_out)
"""