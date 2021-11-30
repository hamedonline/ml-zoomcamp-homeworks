import json
import tflite_runtime.interpreter as tflite
from numpy import array
from io import BytesIO
from urllib import request
from PIL import Image


# initialize interpreter
interpreter = tflite.Interpreter(model_path='cats-dogs-v2.tflite')
interpreter.allocate_tensors()

# get input and output tensors
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x


def predict(url):
    img = download_image(url)
    img_resized = prepare_image(img, (150, 150))
    img_array = array(img_resized, dtype='float32')
    img_array /= 255.0
    X = array([img_array])

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_index)

    # compose result dictionary and return it as a json string
    result = {
        'prediction': float(prediction),
    }
    return json.dumps(result)


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result