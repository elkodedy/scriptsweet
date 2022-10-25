# import the necessary packages
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
import numpy as np
import os

# initialize path and model names
dataset_folder_path = "dataset"
output_model_name = "my_mask_model_2000"

# initialize learning rate, epoch size, dan batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# grab the list of images in our dataset directory
print("__LOG__ loading images...")
imagePaths = list(paths.list_images(dataset_folder_path))
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]

	# load the input image (150x150) and preprocess it
	image = load_img(imagePath, target_size=(150, 150))
	image = img_to_array(image)
	image = preprocess_input(image)

	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)

# convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# partition data into training(80%) and testing(20%)
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# initialize training image
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load the MobileNetV2 network
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor = Input(shape=(150, 150, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(5, 5))(headModel)  #  with average pooling 5x5 kernel
headModel = Flatten(name="flatten")(headModel)  # flatten -> fully connected layer --->
headModel = Dense(128, activation="relu")(headModel)  #
headModel = Dropout(0.5)(headModel)  # dropout bad probability
headModel = Dense(2, activation="relu")(headModel)  #
# headModel = Dense(2, activation="softmax")(headModel)  #

# mark the model
model = Model(inputs=baseModel.input, outputs=headModel)

# loop all layers and mark it so they don't update it self during training
for layer in baseModel.layers:
	layer.trainable = False

# compile model
print("__LOG__ compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("__LOG__ training network...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# predict on testing
print("__LOG__ evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each testing image to find largest probability
predIdxs = np.argmax(predIdxs, axis=1)

# print report
print(classification_report(testY.argmax(axis=1), predIdxs,	target_names=lb.classes_))

# save the model
print("__LOG__ Menyimpan Model...")
model.save(output_model_name, save_format="h5")
