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
parameter_units = {
    'Depth of Cut': 'mm',
    'Velocity of Cutting': 'm/min',
    'Feed Rate': 'mm/rev'
}

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
        class_labels = ['Continuous long chip', 'Continuous short chip', 'Discontinuous chip']
        parameters_1 = [
            {'Depth of Cut': '2mm', 'Velocity of Cutting': '60.85m/min', 'Feed Rate': '0.08mm/rev'},
            {'Depth of Cut': '2mm', 'Velocity of Cutting': '60.85m/min', 'Feed Rate': '0.16mm/rev'},
            {'Depth of Cut': '2mm', 'Velocity of Cutting': '60.85m/min', 'Feed Rate': '0.2mm/rev'}
        ]

        parameters_2 = [
            {'Depth of Cut': 2.0, 'Velocity of Cutting': 60.85, 'Feed Rate': 0.08},
            {'Depth of Cut': 2.0, 'Velocity of Cutting': 60.85, 'Feed Rate': 0.16},
            {'Depth of Cut': 2.0, 'Velocity of Cutting': 60.85, 'Feed Rate': 0.20}
        ]
        predicted_class = class_labels[predicted_class_index]
        chip_parameters_1 = parameters_1[predicted_class_index]
        chip_parameters_2 = parameters_2[predicted_class_index]

        # Calculate changes required to achieve desired parameters
        desired_parameters = {'Velocity of Cutting': 280, 'Feed Rate': 0.08}
        changes_required = {}
        for key, desired_value in desired_parameters.items():
            if key in chip_parameters_2:
                current_value = chip_parameters_2[key]
                unit = parameter_units[key]
                if current_value != desired_value:
                    if current_value < desired_value:
                        change_direction = 'increase'
                    else:
                        change_direction = 'decrease'
                    change_amount = abs(desired_value - current_value)
                    changes_required[key] = f"{change_direction} by {change_amount} {unit}"
                else:
                    changes_required[key] = f"No change needed ({desired_value} {unit})"
            else:
                changes_required[key] = f"Parameter not found, set to {desired_value} {unit}"

        # Prepare response based on predicted class and changes required
        response = {
            'predicted_class': predicted_class,
            'chip_parameters': chip_parameters_1,
            'changes_required': changes_required
        }

       
        return render_template('chipmorphology_result.html', response=response)

    return 'Invalid file format', 400

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=os.environ.get('PORT', 8080))
