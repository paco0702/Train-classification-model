import tensorflow as tf
import matplotlib.pyplot as plt

import tensorflow_hub as hub
import tensorflow_datasets as tfds

from tensorflow.keras import layers
import os
import numpy as np
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import keras
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image


mobile = keras.applications.mobilenet_v2.MobileNetV2()
#mobile = keras.applications.mobilenet.MobileNet()

#data is stored in the zip file
import zipfile
with zipfile.ZipFile('/"yourdata".zip', 'r') as zip_ref:
    zip_ref.extractall('your directory')

base_dir = '/content/my_folder'
#zip_dir = '/content/my_folder/instrument_full_dataset.zip'
zip_dir = '/content/my_folder/{data zip file}.zip'
#base_dir = os.path.join(os.path.dirname(zip_dir), 'instrument_full_dataset')
base_dir = os.path.join(os.path.dirname(zip_dir), 'instrument_third_datasetnew')
base_dir

#number of classification you want
#classes = ['acordian','alphorn', 'bagpipes', 'banjo', 'bongo drum', 'casaba', 'castanets', 'clarinet', 'clavichord', 'concertina', 'Didgeridoo', 'drums', 'dulcimer','flute','guiro','guitar', 'harmonica','harp','marakas','ocarina', 'piano','saxaphone', 'sitar','stell drum', 'Tambourine', 'trombone','trumpet', 'tuba', 'violin', 'Xylophone']
#classes = ['acordian','alphorn',  'banjo', 'bongo drum', 'casaba', 'castanets','guitar', 'piano']
classes = ['acordian', 'alphorn', 'banjo', 'bongo drum', 'casaba', 'castanets', 'clarinet', 'flute', 'guitar', 'piano', 'recorder']


train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

test_data = os.path.join(test_dir, 'new_test')

BATCH_SIZE = 150 # specify HOW MANY train example feeding to the model
IMG_SHAPE = 224 # want to resize all the images to 150x150 height and width

#now we do the augmentation here
# this function is to show the image after did the augmentation
def plotImages(images_arr):
  fig, axes = plt.subplots(1,5, figsize =(20,20))
  axes = axes.flatten()
  for img, ax in zip(images_arr, axes):
    ax.imshow(img)
  plt.tight_layout()
  plt.show()


#training set
image_gen_train = ImageDataGenerator(rescale = 1./255,
                    horizontal_flip=True,
                    zoom_range = 0.5,
                    width_shift_range=15,
                    height_shift_range=15,
                    rotation_range=45)

train_data_gen = image_gen_train.flow_from_directory(
    batch_size=BATCH_SIZE,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    color_mode = 'rgb',
    class_mode='sparse',
    directory=train_dir,
    shuffle = True )


#test set
image_gen_test = ImageDataGenerator(rescale = 1./255)

test_data_gen = image_gen_test.flow_from_directory(
    batch_size=BATCH_SIZE,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    color_mode = 'rgb',
    class_mode='sparse',
    directory=test_dir,
    shuffle = True,
   )


image_gen_val = ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,
                           directory=validation_dir,
                           shuffle = False,
                           target_size=(IMG_SHAPE,IMG_SHAPE),
                           class_mode='sparse')

# keep the original images without augmentation in validation set

base_model=MobileNet(input_shape=(224,224,3),weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

#multiple output

x=base_model.output
x=tf.keras.layers.MaxPooling2D(2,2)(x)
x=tf.keras.layers.Conv2D(1024,(3,3), padding='same', activation='relu',input_shape=(IMG_SHAPE,IMG_SHAPE,3))(x)
x=tf.keras.layers.MaxPooling2D(2,2)(x)
x=tf.keras.layers.Conv2D(1024,(3,3), padding='same', activation='relu')(x)
x=tf.keras.layers.MaxPooling2D(2,2, padding='same')(x)
x=tf.keras.layers.Conv2D(2048,(3,3), padding='same', activation='relu')(x)
x=tf.keras.layers.MaxPooling2D(2,2, padding='same')(x)
x=tf.keras.layers.Conv2D(2048,(3,3), padding='same', activation='relu')(x)
x=tf.keras.layers.MaxPooling2D(2,2, padding='same')(x)
x=tf.keras.layers.Conv2D(2048,(3,3), padding='same', activation='relu')(x)
x=tf.keras.layers.MaxPooling2D(2,2, padding='same')(x)
x=tf.keras.layers.Conv2D(2048,(3,3), padding='same', activation='relu')(x)
x=tf.keras.layers.MaxPooling2D(2,2, padding='same')(x)
x=tf.keras.layers.Conv2D(2048,(3,3), padding='same', activation='relu')(x)

x=tf.keras.layers.Dropout(0.5)(x)
x=tf.keras.layers.Flatten()(x) #define the input shape in the mobile net and it works
#x=tf.keras.Input(shape=(IMG_SHAPE,IMG_SHAPE,3))
#x=tf.keras.layers.GlobalAveragePooling
x=tf.keras.layers.Dense(512, activation='relu')(x)
preds=tf.keras.layers.Dense(11, activation='sigmoid')(x) #try multiple output



model=Model(inputs=base_model.input,outputs=preds)


for layer in model.layers:
    layer.trainable=False
# or if we want to set the first 70 layers of the network to be non-trainable
for layer in model.layers[:90]:
    layer.trainable=False
for layer in model.layers[90:]:
    layer.trainable=True


model.compile(optimizer='Adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']) #for now two categories

step_size_train=train_data_gen.n//train_data_gen.batch_size

history = model.fit(x = train_data_gen,
      epochs =30, #epochs 20 is the best #15 is pervious lets try 40 next time
      steps_per_epoch = step_size_train,
      validation_data =val_data_gen )


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(6)


image_batch, label_batch = test_data_gen.next()

predicted_batch = model.predict(image_batch)

squeeze_predicted_batch = tf.squeeze(predicted_batch).numpy()
squeeze_predicted_batch

predicted_batch = model.predict(image_batch)
squeeze_predicted_batch = tf.squeeze(predicted_batch).numpy()

most_likely_predicted_ids = np.argmax(squeeze_predicted_batch, axis=-1)

class_names = np.array(classes)
#class name in string array


predicted_class_names = class_names[most_likely_predicted_ids]

plt.figure(figsize=(10,9))
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.subplots_adjust(hspace = 0.3)
  plt.imshow(image_batch[n])
  #color = "blue" if predicted_ids[n] == label_batch[n] else "red"
  most_likely_predicted_ids
  color = "blue" if most_likely_predicted_ids[n] == label_batch[n] else "red"
  plt.title(predicted_class_names[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")


def classify_images(test_data_gen):
    image_batch, label_batch = test_data_gen.next()
    predicted_batch = model.predict(image_batch)
    predicted_batch = tf.squeeze(predicted_batch).numpy()
    predicted_ids = np.argmax(predicted_batch, axis=-1)
    class_names = np.array(classes)
    predicted_class_names = class_names[predicted_ids]
    plot_result(image_batch, predicted_ids, label_batch, predicted_class_names)


def plot_result(image_batch, predicted_ids, label_batch, predicted_class_names):
    plt.figure(figsize=(10, 9))
    for n in range(30):
        plt.subplot(6, 5, n + 1)
        plt.subplots_adjust(hspace=0.3)
        plt.imshow(image_batch[n])
        color = "blue" if predicted_ids[n] == label_batch[n] else "red"
        plt.title(predicted_class_names[n].title(), color=color)
        plt.axis('off')
        _ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")

classify_images(test_data_gen)


#tenserflow servering
# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors,
# and stored with the default serving key
import tempfile
MODEL_DIR = tempfile.gettempdir()
version = 1
export_path = os.path.join(MODEL_DIR, str(version))
print('export_path = {}\n'.format(export_path))

tf.keras.models.save_model(
    model,
    export_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)

#save the model
import time

t = time.time()

export_path_keras = "./{}.h5".format(int(t))
print(export_path_keras)

model.save(export_path_keras)

reloaded = tf.keras.models.load_model(
  #export_path_keras,
  "./1644300452.h5",
  # `custom_objects` tells keras how to load a `hub.KerasLayer`
  custom_objects={'KerasLayer': hub.KerasLayer})

reloaded.summary()








