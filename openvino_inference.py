import logging
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
import cv2
import numpy as np

# Create a logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Define a console handler and set the formatter
ch = logging.StreamHandler()
ch.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(ch)

class TritonInstanceSegmentationModel:
    def __init__(self, model_name, server_url="localhost:8000"):
        self.model_name = model_name
        self.server_url = server_url
        self.client = InferenceServerClient(url=self.server_url)
        self.input_name = "images"
        self.input_shape = (3, 640, 640)  # Remove the batch dimension
        self.output_names = ["output0"]

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (640, 640))
        normalized_image = resized_image.astype(np.float32) / 255.0
        normalized_image = normalized_image.transpose(2, 0, 1)  # Transpose dimensions
        return normalized_image

    def infer(self, image_path):
        try:
            normalized_image = self.preprocess_image(image_path)
            input_tensor = InferInput(self.input_name, self.input_shape, "FP32")
            input_tensor.set_data_from_numpy(normalized_image)

            outputs = [InferRequestedOutput(output, binary_data=True) for output in self.output_names]
            response = self.client.infer(self.model_name, inputs=[input_tensor], outputs=outputs)
            output_data = [response.as_numpy(name) for name in self.output_names]

            return output_data
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            return None

    def draw_annotations(self, image_path, output_data):
        try:
            image = cv2.imread(image_path)
            annotated_image = image.copy()

            if isinstance(output_data, list) and len(output_data) == 1 and isinstance(output_data[0], np.ndarray):
                output_array = output_data[0]

                num_channels = output_array.shape[1]

                # Assuming first four channels contain bounding box coordinates
                num_bbox_channels = 4

                # Assuming each class has a confidence score channel
                num_class_channels = num_channels - num_bbox_channels

                if num_bbox_channels < 4 or num_class_channels <= 0:
                    logger.error("Invalid number of channels. Unable to extract bounding box and confidence score information.")
                    return

                # Extract bounding box coordinates from the first four channels
                # Extract bounding box coordinates from the first four channels
                for channel_idx in range(num_bbox_channels):
                    channel_data = output_array[0, channel_idx]  # Assuming batch dimension at index 0

                    # Assuming bounding box coordinates are [x1, y1, x2, y2]
                    x1, y1, x2, y2 = channel_data

                    cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)


                # Extract confidence scores for each class
                for class_idx in range(num_class_channels):
                    confidence_scores = output_array[0, num_bbox_channels + class_idx]  # Assuming batch dimension at index 0

                    # Assuming each confidence score corresponds to a class
                    class_id = class_idx  # Assuming class IDs start from 0

                    # Assuming confidence score is the maximum value across the confidence scores
                    confidence_score = np.max(confidence_scores)

                    threshold = 0.2  # Adjust threshold as needed

                    if confidence_score > threshold:
                        cv2.putText(annotated_image, f"Class: {class_id}", (10, 30 * class_idx + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Display or save the annotated image
                cv2.imshow("Annotated Image", annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                logger.info("Annotated image displayed successfully.")
        except Exception as e:
            logger.error(f"An error occurred while drawing annotations: {str(e)}")



def main():
    # Replace with your Triton server URL without the scheme
    server_url = "localhost:8000"  # Adjust port if needed

    try:
        model = TritonInstanceSegmentationModel(model_name="yolov8n", server_url=server_url)
        output_data = model.infer("./test1.jpg")
        # logger.info(output_data)
        logger.info("Inference outputs:")
        for name, data in zip("output0", output_data):
            logger.info(f"{name}: {data}")


        if output_data is not None:
            model.draw_annotations("./test1.jpg", output_data)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
