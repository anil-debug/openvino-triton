from flask import Flask, request, jsonify
from openvino_inference import TritonInstanceSegmentationModel

app = Flask(__name__)
model = TritonInstanceSegmentationModel(model_name="yolov8n")

@app.route('/infer', methods=['POST'])
def infer():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'})

    image_file = request.files['image']
    image_path = "./uploaded_image.jpg"  # Save the uploaded image to a file
    image_file.save(image_path)

    output_data = model.infer(image_path)
    # Process output_data as needed

    return jsonify({'output': output_data})

if __name__ == '__main__':
    app.run(debug=True)
