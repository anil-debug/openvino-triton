# from flask import Flask, request, jsonify
# from openvino_inference import TritonInstanceSegmentationModel
# import os

# app = Flask(__name__)
# model = TritonInstanceSegmentationModel(model_name="yolov8n")

# @app.route('/infer', methods=['POST'])
# def infer():
#     try:
#         if 'image' not in request.files:
#             return jsonify({'error': 'No image file provided'})

#         image_file = request.files['image']
#         if image_file.filename == '':
#             return jsonify({'error': 'Empty filename provided'})

#         # Save the uploaded image to a file
#         image_path = "./uploaded_image.jpg"  
#         image_file.save(image_path)

#         output_data = model.infer(image_path)
#         # Process output_data as needed

#         return jsonify({'output': output_data})
#     except Exception as e:
#         return jsonify({'error': str(e)})
#     finally:
#         # Clean up the uploaded image file
#         if os.path.exists(image_path):
#             os.remove(image_path)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)
# Using flask to make an api 
# import necessary libraries and functions 

from flask import Flask, request, jsonify
import os
from io import BufferedReader
from openvino_inference import TritonInstanceSegmentationModel
import cv2
import os
import numpy as np
# Create the Flask app
app = Flask(__name__)
server_url = "localhost:8000"  # Adjust port if needed
# Load the segmentation model (assuming it's already downloaded)
model = TritonInstanceSegmentationModel(model_name="yolov8n", server_url=server_url)

@app.route('/hello', methods=['GET', 'POST'])
def hello():
    if request.method == 'GET':
        data = "Hello world!"
        return jsonify({'data': data})
    elif request.method == 'POST':
        # Handle potential errors in data reception
        try:
            data = request.get_json()  # Assuming data is sent as JSON in POST requests
            if data is None:
                return jsonify({'error': 'No data provided in request body'}), 400
            message = data.get('message')
            if message is None:
                return jsonify({'error': 'Missing "message" key in request data'}), 400
            return jsonify({'response': f"You sent: {message}"})
        except Exception as e:
            return jsonify({'error': str(e)}), 400



def infer():
    try:
        # Read the uploaded image from the request files
        image_file = request.files['files']
        
        # Read and decode the image
        npimg = np.fromstring(image_file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        # Get the current working directory
        current_directory = os.getcwd()

        # Save the decoded image in the current directory
        # save2_path = os.path.join('openvino_triton_inference', '99.jpg')
        # save_path = '/openvino_triton_inference/99.jpg'
        # save_successful = cv2.imwrite(save_path, img)

        # if save_successful:
        # Perform segmentation inference
        output_data = model.infer(img)  # Pass the decoded image to the model
        print(output_data)
        # Process output_data as needed

        return jsonify({'output': output_data})
        # else:
        #     return jsonify({'error': 'Failed to save the image. Check directory permissions.'}), 500
    except cv2.error as cv2_error:
        return jsonify({'error': f'OpenCV error: {cv2_error}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
