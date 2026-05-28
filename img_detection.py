import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("rust_vs_crust_model.h5")

def detect_cracks(image_path, output_path="crack_1.jpg"):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Find contours of detected cracks
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    output_image = image.copy()
    cv2.drawContours(output_image, contours, -1, (0, 0, 255), 2)

    # CNN model prediction part
    cnn_input = cv2.resize(image, (128, 128))
    cnn_input = cnn_input / 255.0
    cnn_input = np.expand_dims(cnn_input, axis=0)

    prediction = model.predict(cnn_input)[0][0]

    if prediction > 0.5:
        label = "Rust"
    else:
        label = "Crack"

    cv2.putText(
        output_image,
        f"CNN Prediction: {label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Save the output image
    cv2.imwrite(output_path, output_image)
    print(f"Processed image saved to {output_path}")
    print(f"Number of detected crack-like regions: {len(contours)}")
    print(f"CNN Prediction: {label}")
    print(f"Confidence Score: {prediction:.2f}")

if __name__ == "__main__":
    input_image = r"C:\Users\prasa\Downloads\R\R\rust_3.jpg"
    detect_cracks(input_image)
if __name__ == "__main__":
    input_image = r"C:\Users\prasa\Downloads\R\R\rust_3.jpg"
    detect_cracks(input_image)
