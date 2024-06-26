import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe pose solution
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose()

# Define landmark indices and their corresponding names
landmark_mapping = {
    11: "LEFT SHOULDER",
    12: "RIGHT SHOULDER",
    13: "LEFT ELBOW",
    14: "RIGHT ELBOW",
    15: "LEFT WRIST",
    16: "RIGHT WRIST"
}

connections = [
    (11, 13),  # Left shoulder to left elbow
    (13, 15),  # Left elbow to left wrist
    (12, 14),  # Right shoulder to right elbow
    (14, 16),  # Right elbow to right wrist
    (11, 12)  # Left shoulder to right shoulder
]
# Take video input for pose detection
cap = cv2.VideoCapture(0)  # You can put here video of your choice ("sampleVideo.mp4")
desiredImageHeight = 400
desiredImageWidth = 640
center_x = desiredImageWidth // 2
center_y = desiredImageHeight // 2

def normalizeCoords(x, y): 
    return x -center_x, y-center_y

def mapAngle(angle):
    mappedAngle = angle-270
    mappedAngle %=360
    return mappedAngle


# Read each frame/image from capture object
while True:
    ret, img = cap.read()
    # Resize image/frame so we can accommodate it on our screen
    img = cv2.resize(img, (desiredImageWidth, desiredImageHeight))

    # Do Pose detection
    results = pose.process(img)
    if results is None:
        continue
    # Draw specific landmarks on the original image
    try:
        for landmark_idx, landmark_name in landmark_mapping.items():
            landmark = results.pose_landmarks.landmark[landmark_idx]
            # Draw the landmark on the original image
            x = int(landmark.x * img.shape[1])
            y = int(landmark.y * img.shape[0])
            cv2.circle(img, (x, y), 6, (0, 0, 255), 6)
            # Print landmark name and coordinates
            if landmark_name == "LEFT SHOULDER" or landmark_name == "LEFT ELBOW":
                print(f"{landmark_name}\t: ({x}, {y})")
            
            # Draw connections
        for connection in connections:
            start_landmark = results.pose_landmarks.landmark[connection[0]]
            end_landmark = results.pose_landmarks.landmark[connection[1]]
            
            start_x = int(start_landmark.x * img.shape[1])
            start_y = int(start_landmark.y * img.shape[0])
            end_x = int(end_landmark.x * img.shape[1])
            end_y = int(end_landmark.y * img.shape[0])
            cv2.line(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 4)

            angle = np.degrees(np.arctan2(end_y - start_y, end_x - start_x))
            angle = angle + 360 if angle < 0 else angle
            if landmark_mapping.get(connection[0]) == "LEFT SHOULDER" and landmark_mapping.get(connection[1]) == "LEFT ELBOW":
                print("{}\t- {}\t: {}".format(landmark_mapping.get(connection[0]), landmark_mapping.get(connection[1]),mapAngle(angle)).expandtabs(9))
            
            
        
    except:  # noqa: E722
        print("Detection is lost")
    print()

    # Display pose on original video/live stream
    cv2.imshow("Pose Estimation", img)

    # Exit loop if any key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
