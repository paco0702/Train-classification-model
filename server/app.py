import flask
from flask import request
import werkzeug
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import numpy as np
import os
import re

IMG_SHAPE = 224
model_path = 'C:"mypath"/FYP/All categories.h5'
acc_model_path = 'C:/"mypath"/FYP/Accessary.h5'
instr_model_path = 'C:/"mypath"/instrument.h5'
daily_model_path = 'C:/"mypath"/Daily_product.h5'
elect_model_path = 'C:/"mypath"/Electronic device.h5'
fash_model_path = 'C:"mypath"/Fashion.h5'
furn_model_path = 'C:/"mypath"/Furniture.h5'
sport_model_path = 'C:/"mypath"/Sport_Items.h5'
station_model_path = 'C:/"mypath"/Stationery.h5'

model = tf.keras.models.load_model(
        # export_path_keras,
        model_path,
        # `custom_objects` tells keras how to load a `hub.KerasLayer`
        custom_objects={'KerasLayer': hub.KerasLayer})


app = flask.Flask(__name__)
currentUserID = ''

@app.route("/")
def showHomePage():
    return "This is home page"


@app.route("/", methods=['GET'])
def getUserID():
    currentUserID = flask.request.headers['userID']
    currentUserID = re.sub(r'b', '',  str(currentUserID))
    currentUserID = re.sub(r'\'', '', str(currentUserID))


@app.route("/predict", methods=['GET','POST'])
def predict():
    currentUserID = flask.request.headers['userID']
    currentUserID = re.sub(r'b', '', str(currentUserID))
    currentUserID = re.sub(r'\'', '', str(currentUserID))
    print(currentUserID)
    path = 'C:/Users/Pacowawo Chiu/Desktop/Server/'+str(currentUserID)+'/storage/pictures'
    run_path = 'C:/Users/Pacowawo Chiu/Desktop/Server/'+str(currentUserID)+'/storage'
    print(path)
    print("run_path "+run_path)
    for i in range(len(flask.request.files)):
        imagefile = flask.request.files['image'+str(i)]
        print(imagefile.filename)
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        img_path = os.path.join(path, filename)
        if not os.path.exists(path):
            os.makedirs(path)
        imagefile.save(img_path)


    major_category = predict_major(run_path, model)
    print(major_category)
    result = ""
    for x in range(len(major_category)):
        result = "|"+sub_classify(run_path, major_category[x])+"| "+result

    print("result "+result)
    return ""+result


def sub_classify(path, major_category):
    if (major_category == 'Accessory'):
        accModel = reloadAccModel()
        output = accessory_predict(path,accModel)
        print("result is Accessory "+ output)
        result = "Accessory, "+output
        return result

    if (major_category == 'Daily product'):
        dailyModel = reloadAccModel();
        output = daily_predict(path, dailyModel)
        print("result is Daily product " + output)
        result = "Daily product, " + output
        return result

    if (major_category == 'Electronic item'):
        elecModel = reloadElectrModel();
        output = elect_predict(path, elecModel)
        print("result is Electronic item " + output)
        result = "Electronic item, " + output
        return result

    if (major_category == 'Fashion'):
        fashModel = reloadFashModel();
        output = fash_predict(path, fashModel)
        print("result is Fashion " + output)
        result = "Fashion, " + output
        return result

    if (major_category == 'Furniture'):
        furnModel = reloadFurnModel();
        output = furn_predict(path, furnModel)
        print("result is Fashion " + output)
        result = "Fashion, " + output
        return result

    if (major_category == 'Instrument'):
        instrModel = reloadInstrModel();
        output = instrument_predict(path, instrModel)
        print("result is Instrument " + output)
        result = "Instrument, " + output
        return result

    if (major_category == 'Sport item'):
        sportModel = reloadSportModel();
        output = sport_predict(path, sportModel)
        print("result is Sport item " + output)
        result = "Sport item, " + output
        return result

    if (major_category == 'Stationery'):
        statModel = reloadStatModel();
        output = stat_predict(path, statModel)
        print("result is Stationery " + output)
        result = "Stationery, " + output
        return result



def reloadAccModel():
    reloaded = tf.keras.models.load_model(
        # export_path_keras,
        acc_model_path,
        # `custom_objects` tells keras how to load a `hub.KerasLayer`
        custom_objects={'KerasLayer': hub.KerasLayer})
    return reloaded

def reloadInstrModel():
    reloaded = tf.keras.models.load_model(
        # export_path_keras,
        instr_model_path,
        # `custom_objects` tells keras how to load a `hub.KerasLayer`
        custom_objects={'KerasLayer': hub.KerasLayer})
    return reloaded


def reloadDailyModel():
    reloaded = tf.keras.models.load_model(
        # export_path_keras,
        daily_model_path,
        # `custom_objects` tells keras how to load a `hub.KerasLayer`
        custom_objects={'KerasLayer': hub.KerasLayer})
    return reloaded


def reloadElectrModel():
    reloaded = tf.keras.models.load_model(
        # export_path_keras,
        elect_model_path,
        # `custom_objects` tells keras how to load a `hub.KerasLayer`
        custom_objects={'KerasLayer': hub.KerasLayer})
    return reloaded


def reloadFashModel():
    reloaded = tf.keras.models.load_model(
        # export_path_keras,
        fash_model_path,
        # `custom_objects` tells keras how to load a `hub.KerasLayer`
        custom_objects={'KerasLayer': hub.KerasLayer})
    return reloaded


def reloadFurnModel():
    reloaded = tf.keras.models.load_model(
        # export_path_keras,
        furn_model_path,
        # `custom_objects` tells keras how to load a `hub.KerasLayer`
        custom_objects={'KerasLayer': hub.KerasLayer})
    return reloaded


def reloadSportModel():
    reloaded = tf.keras.models.load_model(
        # export_path_keras,
        sport_model_path ,
        # `custom_objects` tells keras how to load a `hub.KerasLayer`
        custom_objects={'KerasLayer': hub.KerasLayer})
    return reloaded


def reloadStatModel():
    reloaded = tf.keras.models.load_model(
        # export_path_keras,
        station_model_path,
        # `custom_objects` tells keras how to load a `hub.KerasLayer`
        custom_objects={'KerasLayer': hub.KerasLayer})
    return reloaded


def predict_major(path, model):
    classes = ['Accessory', 'Daily product', 'Electronic item', 'Fashion', 'Furniture', 'Instrument', 'Sport item', 'Stationery']
    print("path is "+path)
    IMG_SHAPE = 224  # want to resize all the images to 150x150 height and width
    image_gen_test = ImageDataGenerator(rescale=1. / 255)
    test_data_gen = image_gen_test.flow_from_directory(
        target_size=(IMG_SHAPE, IMG_SHAPE),
        color_mode='rgb',
        class_mode='sparse',
        directory=path,
    )
    return classify_major_images(test_data_gen, model, classes)


def classify_major_images(test_data_gen, model, classes):
    count_class = [0, 0, 0, 0, 0, 0, 0, 0]
                #'Accessory', 'Daily product', 'Electronic item', 'Fashion', 'Furniture', 'Instrument', 'Sport item', 'Stationery'
    image_batch, label_batch = test_data_gen.next()
    predicted_batch = model.predict(image_batch)
    predicted_batch = tf.squeeze(predicted_batch).numpy()
    predicted_ids = np.argmax(predicted_batch, axis=-1)
    class_names = np.array(classes)
    predicted_class_names = class_names[predicted_ids]
    print(predicted_class_names)

    return predicted_class_names



def accessory_predict(path, model):
    classes = ['bracelet', 'brooch', 'necklace', 'ring', 'sun glasses', 'watch'];
    print("path is "+path)
    IMG_SHAPE = 224  # want to resize all the images to 150x150 height and width
    image_gen_test = ImageDataGenerator(rescale=1. / 255)
    test_data_gen = image_gen_test.flow_from_directory(
        target_size=(IMG_SHAPE, IMG_SHAPE),
        color_mode='rgb',
        class_mode='sparse',
        directory=path,
    )
    return classify_acc_images(test_data_gen, model, classes)


def classify_acc_images(test_data_gen, model, classes):
    count_class = [0, 0, 0, 0, 0, 0]
    image_batch, label_batch = test_data_gen.next()
    predicted_batch = model.predict(image_batch)
    predicted_batch = tf.squeeze(predicted_batch).numpy()
    predicted_ids = np.argmax(predicted_batch, axis=-1)
    class_names = np.array(classes)
    predicted_class_names = class_names[predicted_ids]
    print(predicted_class_names)
    for x in range(len(predicted_class_names)):
        if(predicted_class_names[x]=='bracelet'):
            count_class[0] = count_class[0]+1
        if (predicted_class_names[x] == 'brooch'):
            count_class[1] = count_class[1] + 1
        if (predicted_class_names[x] == 'necklace'):
            count_class[2] = count_class[2] + 1
        if (predicted_class_names[x] == 'ring'):
            count_class[3] = count_class[3] + 1
        if (predicted_class_names[x] == 'sun glasses'):
            count_class[4] = count_class[4] + 1
        if (predicted_class_names[x] == 'watch'):
            count_class[5] = count_class[5] + 1

    getCount = -1
    for x in range(len(count_class)):
        print(count_class[x])
        if(count_class[x]>getCount):
            getCount = x

    output = classes[getCount]
    print("frequent appear is "+output)
    return output


def instrument_predict(path, model):
    classes = ['acordian', 'alphorn', 'banjo', 'bongo drum', 'casaba', 'castanets', 'clarinet', 'flute', 'guitar',
               'piano', 'recorder']
    print("path is "+path)
    IMG_SHAPE = 224  # want to resize all the images to 150x150 height and width
    image_gen_test = ImageDataGenerator(rescale=1. / 255)
    test_data_gen = image_gen_test.flow_from_directory(
        target_size=(IMG_SHAPE, IMG_SHAPE),
        color_mode='rgb',
        class_mode='sparse',
        directory=path,
    )
    return classify_instrument_images(test_data_gen, model, classes)


def classify_instrument_images(test_data_gen, model, classes):
    #'acordian', 'alphorn', 'banjo', 'bongo drum', 'casaba', 'castanets', 'clarinet', 'flute', 'guitar', 'piano', 'recorder'
    count_class = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    image_batch, label_batch = test_data_gen.next()
    predicted_batch = model.predict(image_batch)
    predicted_batch = tf.squeeze(predicted_batch).numpy()
    predicted_ids = np.argmax(predicted_batch, axis=-1)
    class_names = np.array(classes)
    predicted_class_names = class_names[predicted_ids]
    print(predicted_class_names)
    for x in range(len(predicted_class_names)):
        if (predicted_class_names[x] == 'acordian'):
            count_class[0] = count_class[0] + 1
        if (predicted_class_names[x] == 'alphorn'):
            count_class[1] = count_class[1] + 1
        if (predicted_class_names[x] == 'banjo'):
            count_class[2] = count_class[2] + 1
        if (predicted_class_names[x] == 'bongo drum'):
            count_class[3] = count_class[3] + 1
        if (predicted_class_names[x] == 'casaba'):
            count_class[4] = count_class[4] + 1
        if (predicted_class_names[x] == 'castanets'):
            count_class[5] = count_class[5] + 1
        if (predicted_class_names[x] == 'clarinet'):
            count_class[6] = count_class[6] + 1
        if (predicted_class_names[x] == 'flute'):
            count_class[7] = count_class[7] + 1
        if (predicted_class_names[x] == 'guitar'):
            count_class[8] = count_class[8] + 1
        if (predicted_class_names[x] == 'piano'):
            count_class[9] = count_class[9] + 1
        if (predicted_class_names[x] == 'recorder'):
            count_class[10] = count_class[10] + 1

    getCount = -1
    for x in range(len(count_class)):
        print(count_class[x])
        if (count_class[x] > getCount):
            getCount = x

    output = classes[getCount - 1]
    print("frequent appear is " + output)
    return output


def daily_predict(path, model):
    classes = ['bag', 'clock', 'computer bag', 'cup', 'knapsack', 'tumbler', 'wallet']
    print("path is "+path)
    IMG_SHAPE = 224  # want to resize all the images to 150x150 height and width
    image_gen_test = ImageDataGenerator(rescale=1. / 255)
    test_data_gen = image_gen_test.flow_from_directory(
        target_size=(IMG_SHAPE, IMG_SHAPE),
        color_mode='rgb',
        class_mode='sparse',
        directory=path,
    )
    return classify_daily_images(test_data_gen, model, classes)


def classify_daily_images(test_data_gen, model, classes):
    #'bag', 'clock', 'computer bag', 'cup', 'knapsack', 'tumbler', 'wallet'
    count_class = [0, 0, 0, 0, 0, 0, 0]
    image_batch, label_batch = test_data_gen.next()
    predicted_batch = model.predict(image_batch)
    predicted_batch = tf.squeeze(predicted_batch).numpy()
    predicted_ids = np.argmax(predicted_batch, axis=-1)
    class_names = np.array(classes)
    predicted_class_names = class_names[predicted_ids]
    print(predicted_class_names)
    for x in range(len(predicted_class_names)):
        if (predicted_class_names[x] == 'bag'):
            count_class[0] = count_class[0] + 1
        if (predicted_class_names[x] == 'clock'):
            count_class[1] = count_class[1] + 1
        if (predicted_class_names[x] == 'computer bag'):
            count_class[2] = count_class[2] + 1
        if (predicted_class_names[x] == 'cup'):
            count_class[3] = count_class[3] + 1
        if (predicted_class_names[x] == 'knapsack'):
            count_class[4] = count_class[4] + 1
        if (predicted_class_names[x] == 'tumbler'):
            count_class[5] = count_class[5] + 1
        if (predicted_class_names[x] == 'wallet'):
            count_class[6] = count_class[6] + 1

    getCount = -1
    for x in range(len(count_class)):
        print(count_class[x])
        if (count_class[x] > getCount):
            getCount = x

    output = classes[getCount]
    print("frequent appear is " + output)
    return output


def elect_predict(path, model):
    classes = ['camera', 'charger', 'game boy', 'head phone', 'ipad', 'laptop', 'nds', 'phone', 'small head phone', 'switch']
    print("path is " + path)
    IMG_SHAPE = 224  # want to resize all the images to 150x150 height and width
    image_gen_test = ImageDataGenerator(rescale=1. / 255)
    test_data_gen = image_gen_test.flow_from_directory(
        target_size=(IMG_SHAPE, IMG_SHAPE),
        color_mode='rgb',
        class_mode='sparse',
        directory=path,
    )
    return classify_elect_images(test_data_gen, model, classes)


def classify_elect_images(test_data_gen, model, classes):
    # 'camera', 'charger', 'game boy', 'head phone', 'ipad', 'laptop', 'nds', 'phone', 'small head phone', 'switch'
    count_class = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    image_batch, label_batch = test_data_gen.next()
    predicted_batch = model.predict(image_batch)
    predicted_batch = tf.squeeze(predicted_batch).numpy()
    predicted_ids = np.argmax(predicted_batch, axis=-1)
    class_names = np.array(classes)
    predicted_class_names = class_names[predicted_ids]
    print(predicted_class_names)
    for x in range(len(predicted_class_names)):
        if (predicted_class_names[x] == 'camera'):
            count_class[0] = count_class[0] + 1
        if (predicted_class_names[x] == 'charger'):
            count_class[1] = count_class[1] + 1
        if (predicted_class_names[x] == 'game boy'):
            count_class[2] = count_class[2] + 1
        if (predicted_class_names[x] == 'head phone'):
            count_class[3] = count_class[3] + 1
        if (predicted_class_names[x] == 'ipad'):
            count_class[4] = count_class[4] + 1
        if (predicted_class_names[x] == 'laptop'):
            count_class[5] = count_class[5] + 1
        if (predicted_class_names[x] == 'nds'):
            count_class[6] = count_class[6] + 1
        if (predicted_class_names[x] == 'phone'):
            count_class[7] = count_class[7] + 1
        if (predicted_class_names[x] == 'small head phone'):
            count_class[8] = count_class[8] + 1
        if (predicted_class_names[x] == 'switch'):
            count_class[9] = count_class[9] + 1


    getCount = -1
    for x in range(len(count_class)):
        print(count_class[x])
        if (count_class[x] > getCount):
            getCount = x

    output = classes[getCount]
    print("frequent appear is " + output)
    return output


def fash_predict(path, model):
    classes = ['dress', 'hat', 'longsleeve', 'outwear', 'pants', 'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']
    print("path is " + path)
    IMG_SHAPE = 224  # want to resize all the images to 150x150 height and width
    image_gen_test = ImageDataGenerator(rescale=1. / 255)
    test_data_gen = image_gen_test.flow_from_directory(
        target_size=(IMG_SHAPE, IMG_SHAPE),
        color_mode='rgb',
        class_mode='sparse',
        directory=path,
    )
    return classify_fash_images(test_data_gen, model, classes)


def classify_fash_images(test_data_gen, model, classes):
    #'dress', 'hat', 'longsleeve', 'outwear', 'pants', 'shirt', 'shoes', 'shorts', 'skirt', 't-shirt'
    count_class = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    image_batch, label_batch = test_data_gen.next()
    predicted_batch = model.predict(image_batch)
    predicted_batch = tf.squeeze(predicted_batch).numpy()
    predicted_ids = np.argmax(predicted_batch, axis=-1)
    class_names = np.array(classes)
    predicted_class_names = class_names[predicted_ids]
    print(predicted_class_names)
    for x in range(len(predicted_class_names)):
        if (predicted_class_names[x] == 'dress'):
            count_class[0] = count_class[0] + 1
        if (predicted_class_names[x] == 'hat'):
            count_class[1] = count_class[1] + 1
        if (predicted_class_names[x] == 'longsleeve'):
            count_class[2] = count_class[2] + 1
        if (predicted_class_names[x] == 'outwear'):
            count_class[3] = count_class[3] + 1
        if (predicted_class_names[x] == 'pants'):
            count_class[4] = count_class[4] + 1
        if (predicted_class_names[x] == 'shirt'):
            count_class[5] = count_class[5] + 1
        if (predicted_class_names[x] == 'shoes'):
            count_class[6] = count_class[6] + 1
        if (predicted_class_names[x] == 'shorts'):
            count_class[7] = count_class[7] + 1
        if (predicted_class_names[x] == 'skirt'):
            count_class[8] = count_class[8] + 1
        if (predicted_class_names[x] == 't-shirt'):
            count_class[9] = count_class[9] + 1

    getCount = -1
    for x in range(len(count_class)):
        print(count_class[x])
        if (count_class[x] > getCount):
            getCount = x

    output = classes[getCount]
    print("frequent appear is " + output)
    return output


def furn_predict(path, model):
    classes = ['bed', 'chairs', 'sofa', 'table']
    print("path is " + path)
    IMG_SHAPE = 224  # want to resize all the images to 150x150 height and width
    image_gen_test = ImageDataGenerator(rescale=1. / 255)
    test_data_gen = image_gen_test.flow_from_directory(
        target_size=(IMG_SHAPE, IMG_SHAPE),
        color_mode='rgb',
        class_mode='sparse',
        directory=path,
    )
    return classify_fash_images(test_data_gen, model, classes)


def classify_furn_images(test_data_gen, model, classes):
    #['bed', 'chairs', 'sofa', 'table']
    count_class = [0, 0, 0, 0]
    image_batch, label_batch = test_data_gen.next()
    predicted_batch = model.predict(image_batch)
    predicted_batch = tf.squeeze(predicted_batch).numpy()
    predicted_ids = np.argmax(predicted_batch, axis=-1)
    class_names = np.array(classes)
    predicted_class_names = class_names[predicted_ids]
    print(predicted_class_names)
    for x in range(len(predicted_class_names)):
        if (predicted_class_names[x] == 'bed'):
            count_class[0] = count_class[0] + 1
        if (predicted_class_names[x] == 'chairs'):
            count_class[1] = count_class[1] + 1
        if (predicted_class_names[x] == 'sofa'):
            count_class[2] = count_class[2] + 1
        if (predicted_class_names[x] == 'table'):
            count_class[3] = count_class[3] + 1

    getCount = -1
    for x in range(len(count_class)):
        print(count_class[x])
        if (count_class[x] > getCount):
            getCount = x

    output = classes[getCount]
    print("frequent appear is " + output)
    return output



def sport_predict(path, model):
    classes = ['badmintionRacket', 'baseball', 'basketball', 'football', 'helmat', 'hockey stick', 'tableTennisRacket', 'tennisRacket', 'volleyball']
    print("path is " + path)
    IMG_SHAPE = 224  # want to resize all the images to 150x150 height and width
    image_gen_test = ImageDataGenerator(rescale=1. / 255)
    test_data_gen = image_gen_test.flow_from_directory(
        target_size=(IMG_SHAPE, IMG_SHAPE),
        color_mode='rgb',
        class_mode='sparse',
        directory=path,
    )
    return classify_sport_images(test_data_gen, model, classes)


def classify_sport_images(test_data_gen, model, classes):
    #['badmintionRacket', 'baseball', 'basketball', 'football', 'helmat', 'hockey stick', 'tableTennisRacket', 'tennisRacket', 'volleyball' ]
    count_class = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    image_batch, label_batch = test_data_gen.next()
    predicted_batch = model.predict(image_batch)
    predicted_batch = tf.squeeze(predicted_batch).numpy()
    predicted_ids = np.argmax(predicted_batch, axis=-1)
    class_names = np.array(classes)
    predicted_class_names = class_names[predicted_ids]
    print(predicted_class_names)
    for x in range(len(predicted_class_names)):
        if (predicted_class_names[x] == 'badmintionRacket'):
            count_class[0] = count_class[0] + 1
        if (predicted_class_names[x] == 'baseball'):
            count_class[1] = count_class[1] + 1
        if (predicted_class_names[x] == 'basketball'):
            count_class[2] = count_class[2] + 1
        if (predicted_class_names[x] == 'football'):
            count_class[3] = count_class[3] + 1
        if (predicted_class_names[x] == 'helmat'):
            count_class[4] = count_class[4] + 1
        if (predicted_class_names[x] == 'hockey stick'):
            count_class[5] = count_class[5] + 1
        if (predicted_class_names[x] == 'tableTennisRacket'):
            count_class[6] = count_class[6] + 1
        if (predicted_class_names[x] == 'tennisRacket'):
            count_class[7] = count_class[7] + 1
        if (predicted_class_names[x] == 'volleyball'):
            count_class[8] = count_class[8] + 1


    getCount = -1
    for x in range(len(count_class)):
        print(count_class[x])
        if (count_class[x] > getCount):
            getCount = x

    output = classes[getCount]
    print("frequent appear is " + output)
    return output


def stat_predict(path, model):
    classes = ['calculator', 'pen', 'scissor']
    print("path is " + path)
    IMG_SHAPE = 224  # want to resize all the images to 150x150 height and width
    image_gen_test = ImageDataGenerator(rescale=1. / 255)
    test_data_gen = image_gen_test.flow_from_directory(
        target_size=(IMG_SHAPE, IMG_SHAPE),
        color_mode='rgb',
        class_mode='sparse',
        directory=path,
    )
    return classify_stat_images(test_data_gen, model, classes)


def classify_stat_images(test_data_gen, model, classes):
    # ['calculator', 'pen', 'scissor']
    count_class = [0, 0, 0]
    image_batch, label_batch = test_data_gen.next()
    predicted_batch = model.predict(image_batch)
    predicted_batch = tf.squeeze(predicted_batch).numpy()
    predicted_ids = np.argmax(predicted_batch, axis=-1)
    class_names = np.array(classes)
    predicted_class_names = class_names[predicted_ids]
    print(predicted_class_names)
    for x in range(len(predicted_class_names)):
        if (predicted_class_names[x] == 'calculator'):
            count_class[0] = count_class[0] + 1
        if (predicted_class_names[x] == 'pen'):
            count_class[1] = count_class[1] + 1
        if (predicted_class_names[x] == 'scissor'):
            count_class[2] = count_class[2] + 1

    getCount = -1
    for x in range(len(count_class)):
        print(count_class[x])
        if (count_class[x] > getCount):
            getCount = x

    output = classes[getCount]
    print("frequent appear is " + output)
    return output



if __name__ == "__main__":
    app.run(host="0.0.0.0")
