import sys
sys.path.append("/home/covid/Covid/BE/venv/Lib/site-packages/")
from flask import Flask, request, render_template, jsonify
import time
from PIL import Image as pillow
from cv2 import imread, resize, cvtColor, COLOR_BGR2RGB
from tensorflow.keras.models import load_model
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
from shutil import copyfile
import json
import socket

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

model = load_model('mdl_wts.hdf5')
last_conv_layer_name = "separable_conv2d_20"
# last_conv_layer_name = "dropout_9"

classifier_layer_names = [
    #   "separable_conv2d_19",
    #   "batch_normalization_9",
    #   "max_pooling2d_15",
    #   "dropout_9",
    #   "separable_conv2d_20",
    "separable_conv2d_21",
    "batch_normalization_10",
    "max_pooling2d_16",
    "dropout_10",
    "separable_conv2d_22",
    "separable_conv2d_23",
    "batch_normalization_11",
    "max_pooling2d_17",
    "dropout_11",
    "flatten_1",
    "dense_5",
    "dropout_12",
    "dense_6",
    "dropout_13",
    "dense_7",
    "dropout_14",
    "dense_8",
    "dropout_15",
    "dense_9",
]

def get_Host_name_IP():
	while True:
		try:
			s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
			s.connect(("8.8.8.8",80))
			host_ip = s.getsockname()[0]
			s.close()
			return host_ip
		except:
			print("unable to get IP")


def dataPrediction(path):
    image = imread(path)
    image = cvtColor(image, COLOR_BGR2RGB)
    image = resize(image, (224, 224))

    # add to array (input format for keras model must be array - one position in our case)
    inputData=[]
    inputData.append(image)

    data = np.array(inputData) / 255.0 #normalization

    pred_Y = model.predict(data, verbose = True)


    return pred_Y

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input

    # x = model.get_layer(classifier_layer_names)(x)
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)

    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def diferenca(path):
    # read image transform and resize
    image = imread(path)

    image = cvtColor(image, COLOR_BGR2RGB)
    image = resize(image, (224, 224))

    # add to array (input format for keras model must be array - on position in our case)
    inputData = []
    inputData.append(image)
    # normalize values like they were used in train
    data = np.array(inputData) / 255.0  # normalization

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(
        data, model, last_conv_layer_name, classifier_layer_names
    )

    img = keras.preprocessing.image.load_img(path)

    img = keras.preprocessing.image.img_to_array(img)

    # We rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap('jet')

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Display Grad CAM
    Image(superimposed_img)

    nano = time.time_ns()
    pathWithOutStatic = "{:f}.jpg".format(nano)
    pathImg1 = "static/{:s}".format(pathWithOutStatic)
    superimposed_img = superimposed_img.resize((2160,1920))
    superimposed_img.save(pathImg1, "JPEG")
    return pathWithOutStatic

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/app')
def aplication():
    return render_template("app.html")

@app.route('/disagree', methods=['POST'])
def disagree():
    req = request.get_json()
    if (req.get("classification") == ""):
        return jsonify({"data": "Error"})

    values = req.get("value")
    if values[0]>values[1] and values[0]>values[2]:
        ourAnalysis = "normal"
    if values[1]>values[0] and values[1]>values[2]:
        ourAnalysis = "covid"
    if values[2] > values[0] and values[2] > values[1]:
        ourAnalysis = "other virus"

    oldPath = req.get("path")
    oldPath2 = oldPath.split("/")
    x = len(oldPath2)
    maniplulatedPath = oldPath2[x-2]+"/"+oldPath2[x-1]

    copyfile(maniplulatedPath, maniplulatedPath.replace("static", "disagree"))

    if not os.path.exists("disagree/data.csv"):
        with open('disagree/data.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter="|")
            writer.writerow(["Our Classification", "Values", "Img Path", "user classification"])


    with open('disagree/data.csv', 'a', newline='') as file:
        writer = csv.writer(file, delimiter="|")
        writer.writerow([ourAnalysis, req.get("value"), maniplulatedPath.replace("static/", ""), req.get("classification")])


    return jsonify({"data": "Success"})

@app.route('/upload', methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, "static/")

    if not os.path.isdir(target):
        os.mkdir(target)

    nano = time.time_ns()
    pathWithOutStatic = "{:f}.jpg".format(nano)
    pathImg = "static/{:s}".format(pathWithOutStatic)

    for file in request.files.getlist("image"):
        filename = file.filename
        file.save(pathImg)


    img = pillow.open(pathImg).convert("RGB")
    img = img.resize((2160,1920))
    img.save(pathImg)

    #Receber valores do prediction
    values = dataPrediction(pathImg)
    valuesPercentage = []

    for i in values[0]:
        valuesPercentage.append(i*100.00)


    imgDiffPath = diferenca(pathImg)

    labels = ["Normal", "Covid", "Other Vírus"]

    return render_template("upload.html", send = pathWithOutStatic, received = imgDiffPath, values = valuesPercentage, labels= labels)

if __name__ == "__main__":
    ip = get_Host_name_IP()
    app.run(debug=False,host=ip, port=80)
