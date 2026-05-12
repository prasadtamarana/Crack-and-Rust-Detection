import cv2
import numpy as np

def detect_cracks(image_path, output_path="crack.jpg"):
# Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load")

# Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)

# Find contours of detected cracks
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
    output_image = image.copy()
    cv2.drawContours(output_image, contours, -1, (0, 0, 255), 2)  # Red contours

# Save the output image
    cv2.imwrite(output_path, output_image)
    print(f"Processed image saved to {output_path}")
    print(f"Number of detected cracks: {len(contours)}")

# Example usage
if __name__ == "__main__":
    input_image = r"C:\Users\prasa\Downloads\R\R\rust_3.jpg"  
    detect_cracks(input_image)  
