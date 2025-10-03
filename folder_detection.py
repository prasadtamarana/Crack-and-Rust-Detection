import cv2
import numpy as np
import tensorflow as tf
import logging
import os

def setup_logging():
    """Set up logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('crack_detection.log'),
            logging.StreamHandler()
        ]
    )

def detect_cracks_with_model(image, model_path=r"C:\Users\prasa\OneDrive\Desktop\crack_dataset_2\rust_vs_crust_model.h5"):
    """Attempt to detect cracks using a pre-trained .h5 model."""
    try:
        # Load the pre-trained model
        model = tf.keras.models.load_model(model_path)
        logging.info(f"Loaded .h5 model from {model_path}")

        # Preprocess image for the model (assuming model expects 256x256 grayscale)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (256, 256))
        input_image = resized / 255.0  # Normalize
        input_image = np.expand_dims(input_image, axis=(0, -1))  # Shape: (1, 256, 256, 1)

        # Predict cracks
        prediction = model.predict(input_image)[0, :, :, 0]
        mask = (prediction > 0.5).astype(np.uint8) * 255  # Binary mask

        # Resize mask back to original image size
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Find contours from the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logging.info("Cracks detected using .h5 model")
        return contours

    except Exception as e:
        logging.warning(f"Failed to use .h5 model: {str(e)}. Falling back to OpenCV method.")
        return None

def detect_cracks(image_path, output_path="crack.jpg", model_path=r"C:\Users\prasa\OneDrive\Desktop\crack_dataset_2\rust_vs_crust_model.h5"):
    """Detect cracks in an image using .h5 model or OpenCV fallback."""
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found or unable to load: {image_path}")

        # Try detecting cracks with .h5 model
        contours = detect_cracks_with_model(image, model_path)

        # Fallback to OpenCV method if model fails or contours are not found
        if contours is None or len(contours) == 0:
            logging.info("Using OpenCV-based crack detection")
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)
            # Morphological operations to enhance cracks
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            eroded = cv2.erode(dilated, kernel, iterations=1)
            # Find contours
            contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        output_image = image.copy()
        cv2.drawContours(output_image, contours, -1, (0, 0, 255), 2)  # Red contours

        # Save the output image
        cv2.imwrite(output_path, output_image)
        print(f"Processed image saved to {output_path}")
        print(f"Number of detected cracks: {len(contours)}")

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        raise

if __name__ == "__main__":
    setup_logging()
    input_folder = r"C:\Users\prasa\Downloads\R\R"  # Your image folder path
    output_folder = "output"  # Output directory
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    # Process all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            input_image = os.path.join(input_folder, filename)
            output_image = os.path.join(output_folder, f"crack_{filename}")
            print(f"Processing {input_image}...")
            detect_cracks(input_image, output_image)