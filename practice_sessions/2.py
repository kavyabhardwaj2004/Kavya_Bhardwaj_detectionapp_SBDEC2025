import cv2

# Read an image
img = cv2.imread(r"C:\Users\HP\PycharmProjects\InfosysInternship\.venv\practice session\computer vision\kavya profile pic.jpeg")   # replace with your file path

# Check if image loaded successfully
if img is None:
    print("Error: Could not read image.")
else:
    # Show the image in a window
    cv2.imshow("My Image", img)

    # Wait until a key is pressed, then close
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import cv2

# Read image
img = cv2.imread(r"C:\Users\HP\PycharmProjects\InfosysInternship\.venv\practice session\computer vision\kavya profile pic.jpeg")

if img is None:
    print("Error: Could not read image.")
else:
    # Flip vertically (0), horizontally (1), or both (-1)
    flip_vertical = cv2.flip(img, 0) #0 is for vertical flip
    flip_horizontal = cv2.flip(img, 1) #1 for horizontal flip
    flip_both = cv2.flip(img, -1) #-1 for both

    # Show results
    cv2.imshow("Original", img)
    cv2.imshow("Flipped Vertically", flip_vertical)
    cv2.imshow("Flipped Horizontally", flip_horizontal)
    cv2.imshow("Flipped Both", flip_both)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

import cv2

# Load an image
img = cv2.imread(r"C:\Users\HP\PycharmProjects\InfosysInternship\.venv\practice session\computer vision\kavya profile pic.jpeg")   # Replace with your file path

# Check if the image loaded correctly
if img is None:
    print("Error: Could not read image.")
    exit()

# Resize image (width=300, height=300)
resized = cv2.resize(img, (300, 300))

# Show both
cv2.imshow("Original", img)
cv2.imshow("Resized", resized)

# Wait for a key press
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: Save the resized image
cv2.imwrite("resized_output.jpg", resized)

import cv2
import numpy as np

img = np.zeros((500, 500, 3), dtype="uint8")

# Draw shapes
cv2.line(img, (0, 0), (500, 500), (255, 0, 0), 5)
cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 3)
cv2.circle(img, (300, 300), 80, (0, 0, 255), -1)

# Add text
cv2.putText(img, "OpenCV Demo", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow("Shapes and Text", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2

img = cv2.imread(r"C:\Users\HP\PycharmProjects\InfosysInternship\.venv\practice session\computer vision\grayscale_output.jpg", 0)  # Load in grayscale

_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("Original", img)
cv2.imshow("Thresholded", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
#non-original img..............................................................................
import cv2

img = cv2.imread(r"C:\Users\HP\PycharmProjects\InfosysInternship\.venv\practice session\computer vision\grayscale_output.jpg", 0)

edges = cv2.Canny(img, 100, 200)

cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

#original img..........................................................
import cv2

img = cv2.imread(r"C:\Users\HP\PycharmProjects\InfosysInternship\.venv\practice session\computer vision\kavya profile pic.jpeg", 0)

edges = cv2.Canny(img, 100, 200)

cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2

img = cv2.imread("kavya profile pic.jpeg", 0)

edges = cv2.Canny(img, 100, 200)

cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2

# Load pre-trained classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

img = cv2.imread("kavya profile pic.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2

img = cv2.imread("shapes.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

cv2.imshow("Contours", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2

img = cv2.imread("ball.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define blue color range
lower_blue = (100, 150, 0)
upper_blue = (140, 255, 255)

mask = cv2.inRange(hsv, lower_blue, upper_blue)
result = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("Original", img)
cv2.imshow("Mask", mask)
cv2.imshow("Filtered", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

img = cv2.imread("person.jpg")
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (50, 50, 400, 500)  # ROI

cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
result = img * mask2[:, :, np.newaxis]

cv2.imshow("Original", img)
cv2.imshow("Foreground Extracted", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Blue color range
    lower_blue = (100, 150, 0)
    upper_blue = (140, 255, 255)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Tracked", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np

img = cv2.imread("text.png", 0)

_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((5, 5), np.uint8)

erosion = cv2.erode(thresh, kernel, iterations=1)
dilation = cv2.dilate(thresh, kernel, iterations=1)

cv2.imshow("Original", thresh)
cv2.imshow("Erosion", erosion)
cv2.imshow("Dilation", dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

img = cv2.imread("text.png", 0)

_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((5, 5), np.uint8)

erosion = cv2.erode(thresh, kernel, iterations=1)
dilation = cv2.dilate(thresh, kernel, iterations=1)

cv2.imshow("Original", thresh)
cv2.imshow("Erosion", erosion)
cv2.imshow("Dilation", dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()
