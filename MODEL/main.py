### Import the packages

from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from keras.models import load_model

### Load our model

app = Flask(__name__) 
model = load_model('model.h5')

### Connect to our website
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    file = request.files['file']
    file.save('input_images/input.jpg')
    
    
    image_path = "input_images/input.jpg"
    height, width = 64,64 
    image = Image.open(image_path)
    image = image.resize((width,height))

    image_array = np.asarray(image)
    image_array = image_array/255.0
    
    image_array = image_array.reshape(1,width,height,3)
    
    prediction = model.predict(image_array)
    output = ""
    if prediction[0][0]>=0.5:
        output = 'Healthy'
    else: 
        output = 'Down Syndrome'
    
    return render_template('index.html', prediction_text = output)

if __name__ == "__main__":
    app.run(debug = False)
