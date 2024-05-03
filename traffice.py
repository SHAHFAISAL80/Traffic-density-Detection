import cv2

# Threshold to detect objects
thres = 0.52  

# Path to your video file
video_file_path = r'C:\Users\Atif Traders\Videos\park\Traffic-density-Detection-using-python-Deep-learning-main\demo.mp4'  # Replace with the path to your video file

# Open video capture from video file
cap = cv2.VideoCapture(video_file_path)

# Set the resolution and frame rate (if needed)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

# Load class names from a text file
classNames = []
classFile = r'C:\Users\Atif Traders\Videos\park\Traffic-density-Detection-using-python-Deep-learning-main\names'  # Path to your names file
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load the pre-trained model (SSD = Single Shot Multibox Detector)
configPath = r'C:\Users\Atif Traders\Videos\park\Traffic-density-Detection-using-python-Deep-learning-main\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = r'C:\Users\Atif Traders\Videos\park\Traffic-density-Detection-using-python-Deep-learning-main\frozen_inference_graph.pb'  # Path to your model files

# Initialize the model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Create a window
cv2.namedWindow('Output', cv2.WINDOW_NORMAL)

# Main loop
while True:
    # Read frame from the video file
    success, img = cap.read()
    
    # Break the loop if the video ends or if there is an error reading the frame
    if not success:
        break

    # Detect objects in the frame
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    # Initialize vehicle count
    vehicle_count = 0

    # Process detections
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # Check if the detected object is a vehicle (using specific class IDs for vehicles)
            # Example class IDs for different types of vehicles:
            if classId in [2, 3, 4, 8]:  # bicycle, car, motorcycle, truck
                vehicle_count += 1
                # Draw a bounding box around the detected vehicle
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                # Add text for the vehicle label and confidence
                cv2.putText(img, f'{classNames[classId - 1].upper()}', (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, f'{confidence * 100:.2f}%', (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            else:
                continue  # Ignore other classes
    
    # Calculate traffic density (e.g., by using the number of vehicles and frame size)
    # Simple approach: Use vehicle_count to estimate traffic density
    traffic_density = vehicle_count

    # Display the vehicle count and traffic density on the frame
    cv2.putText(img, f'Vehicle Count: {vehicle_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(img, f'Traffic Density: {traffic_density}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the output frame
    cv2.imshow('Output', img)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
