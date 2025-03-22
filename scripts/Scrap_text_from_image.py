import cv2
import pytesseract

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load image
image_path = "C:/Users/Joshua/Downloads/Ai/dataset/img_dataset/Whatsapp_2.jpg"
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Show the processed image
cv2.imshow("Processed Image", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the processed image for debugging
cv2.imwrite("processed_image.png", thresh)

# Extract text
extracted_text = pytesseract.image_to_string(thresh)
print("\nExtracted Text:\n", extracted_text)
