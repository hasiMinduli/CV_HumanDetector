import cv2
import numpy as np
import math

def convolution(input_matrix, filter_matrix):
    outputconv_value = 0  # Initialize output value
    # Loop through each element in the matrices and perform convolution
    for i in range(3):
        for j in range(3):
            outputconv_value += input_matrix[i][j] * filter_matrix[2-i][2-j]  # Flip the filter matrix
    return outputconv_value 

def sobel(input_matrix):
    sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    setsobel_x = convolution(input_matrix, sobel_x)
    setsobel_y = convolution(input_matrix, sobel_y)
    sobel_magnitude = math.sqrt(setsobel_x ** 2 + setsobel_y ** 2)  
    return sobel_magnitude

def sobel_edge_detection(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Manually apply Gaussian blur for noise reduction
    gaussian_kernel = np.array([[1, 4, 7, 4, 1],
                                [4, 16, 26, 16, 4],
                                [7, 26, 41, 26, 7],
                                [4, 16, 26, 16, 4],
                                [1, 4, 7, 4, 1]]) / 273
    blurred = np.zeros_like(gray)
    for i in range(2, gray.shape[0] - 2):
        for j in range(2, gray.shape[1] - 2):
            blurred[i, j] = np.sum(gray[i-2:i+3, j-2:j+3] * gaussian_kernel)

    # Manually apply Sobel edge detection
    edges = np.zeros_like(blurred)
    for i in range(1, blurred.shape[0] - 1):
        for j in range(1, blurred.shape[1] - 1):
            input_matrix = blurred[i-1:i+2, j-1:j+2]
            edges[i, j] = sobel(input_matrix)

    return edges

# Function to perform human detection using color thresholding
def detect_humans(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper bounds for skin color in HSV
    lower_skin = np.array([0, 50, 80], dtype=np.uint8)  # Example values, adjust as needed
    upper_skin = np.array([20, 250, 250], dtype=np.uint8)  # Example values, adjust as needed
    
    # Threshold the HSV image to extract skin regions
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Manually apply morphological operations to remove noise (erode)
    eroded = erode(mask)
    
    # Manually apply morphological operations to remove noise (dilate)
    dilated = dilate(eroded)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around detected humans
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color, thickness 2
    
    return image

# Manual erosion function
def erode(image):
    kernel = np.ones((5, 5), dtype=np.uint8)
    eroded = np.zeros_like(image)
    for i in range(2, image.shape[0] - 2):
        for j in range(2, image.shape[1] - 2):
            if np.all(image[i-2:i+3, j-2:j+3]):
                eroded[i, j] = 255
    return eroded

# Manual dilation function
def dilate(image):
    kernel = np.ones((5, 5), dtype=np.uint8)
    dilated = np.zeros_like(image)
    for i in range(2, image.shape[0] - 2):
        for j in range(2, image.shape[1] - 2):
            if np.any(image[i-2:i+3, j-2:j+3]):
                dilated[i, j] = 255
    return dilated


# Function to process a single frame
def process_frame(frame):
    # Divide the frame into upper and lower parts
    height, width, _ = frame.shape
    upper_frame = frame[0:height//2, :]
    lower_frame = frame[height//2:, :]
    
    # Apply Sobel edge detection for motion detection
    edges_upper = sobel_edge_detection(upper_frame)
    
    # Detect humans in the upper part of the frame
    detected_upper = detect_humans(upper_frame)
    
    # Concatenate the upper processed part with the unchanged lower part
    processed_frame = np.vstack((detected_upper, lower_frame))
    
    return edges_upper, processed_frame

# Load the video
cap = cv2.VideoCapture("sample_video.mp4")  # Update with the path to your video

# Read the first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error: Failed to read the first frame.")
    exit()

while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Process the current frame
    edges_current, processed_frame = process_frame(frame)
    edges_prev, _ = process_frame(prev_frame)

    # Calculate absolute difference between current and previous frames
    diff_frame = cv2.absdiff(edges_prev, edges_current)

    # Apply threshold to the difference frame
    _, thresh = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)

    # Display the result
    cv2.imshow('Motion Detection', thresh)

    # Update the previous frame
    prev_frame = frame.copy()

    # Break the loop if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()