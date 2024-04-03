from flask import Flask, request, render_template
from flask_cors import CORS
import os
import io
import base64
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# Load the saved model
model_path = 'model.h5'  # Replace 'path/to/your/saved/model.h5' with the actual path of your saved model
cnn = load_model(model_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/toolwear', methods=['POST'])
def tool_wear_handler():
    if request.method != 'POST':
        return 'Method not allowed', 405

    # Access form values
    cutting_force = float(request.form.get('cuttingForce'))
    depth_of_cut = float(request.form.get('depthOfCut'))
    cutting_speed = float(request.form.get('cuttingSpeed'))
    feed = float(request.form.get('feed'))


    # Example data processing
    cutting_force = cutting_force / 384.2597
    depth_of_cut = depth_of_cut / 0.8
    cutting_speed = cutting_speed / 390
    feed = feed / 0.13

    # Load the new model
    new_model = load_model('toollifeprediction.h5')

    # Prepare data for prediction
    data = {'F': [cutting_force], 'ap': [depth_of_cut], 'vc': [cutting_speed], 'f': [feed]}
    df = pd.DataFrame(data)

    # Make prediction
    prediction = new_model.predict(df)
    tool_wear_result = ((0.3 - prediction[0][0]) / 0.3) * 100
    tool_wear_result = 100-tool_wear_result

    # Render the HTML template with results
    return render_template('toolwear_result.html', tool_wear_result=tool_wear_result)

@app.route('/chipmorphology', methods=['POST'])
def chip_morphology_handler():
    if request.method != 'POST':
        return 'Method not allowed', 405

    if 'image' in request.files:
        image = request.files['image'].read()
        image_base64 = base64.b64encode(image).decode('utf-8')
        image_bytes = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((64, 64))
        img_array = np.array(img) / 255.0

        # Predict the class of the image
        predictions = cnn.predict(np.expand_dims(img_array, axis=0))
        predicted_class_index = np.argmax(predictions)

        # Map predicted class index to class label and parameters
        class_labels = ['Continious long chip', 'Continious short chip', 'Discontinious chip']
        parameters = [
            {'depth_of_cut': '2mm', 'velocity_of_cutting': '60.85m/mim', 'feed_rate': '0.08mm/rev'},
            {'depth_of_cut': '2mm', 'velocity_of_cutting': '60.85m/mim', 'feed_rate': '0.16mm/rev'},
            {'depth_of_cut': '2mm', 'velocity_of_cutting': '60.85m/mim', 'feed_rate': '0.2mm/rev'}
        ]
        predicted_class = class_labels[predicted_class_index]
        chip_parameters = parameters[predicted_class_index]

        # Prepare response based on predicted class
        response = {
            'predicted_class': predicted_class,
            'chip_parameters': chip_parameters
        }



        # Render the HTML template with results
        return render_template('chipmorphology_result.html', response=response)

    return 'Invalid file format', 400

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=os.environ.get('PORT', 8080))
