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
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

INIT_LR = 1e-4
EPOCHS = 20
BS = 32

dr = r"D:\Face-Mask-Detection-master\dataset"
cats = ["with_mask", "without_mask"]

print("Loading images......")

data = []
labels = []

for category in cats:
    path = os.path.join(dr, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path ,
                         target_size=(224,224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)


lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype= "float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data,labels,
                                                  test_size=0.20,
                                                  stratify=labels,
                                                  random_state=42)


augment = ImageDataGenerator(
    rotation_range= 20,
    zoom_range= 0.15,
    width_shift_range= 0.2,
    height_shift_range= 0.2,
    shear_range= 0.15,
    horizontal_flip= True,
    fill_mode="nearest")

bModel = MobileNetV2(weights="imagenet",
                     include_top=False,
                     input_tensor= Input(shape=(224,224,3)))

hModel = bModel.output
hModel = AveragePooling2D(pool_size=(7, 7))(hModel)
hModel = Flatten(name="flatten")(hModel)
hModel = Dense(128, activation="relu")(hModel)
hModel = Dropout(0.5)(hModel)
hModel = Dense(2, activation ="Softmax")(hModel)

model = Model(inputs = bModel.input, outputs = hModel)


for layer in bModel.layers:
     layer.trainable = False

print("compiling model.....")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss = "binary_crossentropy",
              optimizer=opt ,
              metrics=["accuracy"])

print("training head.......")
H = model.fit(
    augment.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch= len(trainX) // BS,
    validation_data= (testX, testY),
    validation_steps= len(testX) // BS,
    epochs = EPOCHS
)

print("evaluating network.....")
predIDxs = model.predict(testX, batch_size=BS)

predIDxs = np.argmax(predIDxs, axis= 1)

print(classification_report(testY.argmax(axis=1),
                            predIDxs,
                            target_names=lb.classes_))

print("saving mask detector model.........")
model.save("masl_detector.model",save_format="h5")


N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0 , N), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0 , N), H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0 , N), H.history["accuracy"], label = "train_acc")
plt.plot(np.arange(0 , N), H.history["val_accuracy"], label = "val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc = "lower left")
plt.savefig("plot.png")




