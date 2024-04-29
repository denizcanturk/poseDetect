import cv2
import mediapipe as mp
import numpy as np

#-----------------------------------------------

class RollingAverageFilter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []

    def add_value(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

    def get_filtered_value(self):
        if len(self.values) == 0:
            return 0  # Return 0 if there are no values in the window
        return sum(self.values) / len(self.values)
    
#-----------------------------------------------

class PoseDetector:
    def __init__(self, window_size=5):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose()
        self.window_size = window_size
        self.landmark_mapping = {
            11: "LEFT SHOULDER",
            12: "RIGHT SHOULDER",
            13: "LEFT ELBOW",
            14: "RIGHT ELBOW",
            15: "LEFT WRIST",
            16: "RIGHT WRIST"
        }
        self.connections = [
            (11, 13),  # Left shoulder to left elbow
            (13, 15),  # Left elbow to left wrist
            (12, 14),  # Right shoulder to right elbow
            (14, 16),  # Right elbow to right wrist
            (11, 12)  # Left shoulder to right shoulder
        ]
        self.leftShoulderToElbowMvgAvg = RollingAverageFilter(window_size)
        self.leftElbowToWristMvgAvg = RollingAverageFilter(window_size)
        self.rightShoulderToElbowMvgAvg = RollingAverageFilter(window_size)
        self.righttElbowToWristMvgAvg = RollingAverageFilter(window_size)

    def process_frame(self, img):
        # Do pose detection
        results = self.pose.process(img)
        if results is None:
            return None

        # Draw landmarks and connections
        try:
            #Putting Landmark with red circles
            for landmark_idx, _ in self.landmark_mapping.items():
                landmark = results.pose_landmarks.landmark[landmark_idx]
                x = int(landmark.x * img.shape[1])
                y = int(landmark.y * img.shape[0])
                cv2.circle(img, (x, y), 6, (0, 0, 255), 6)
            
            #Drawing lines for each connections
            for connection in self.connections:
                start_landmark = results.pose_landmarks.landmark[connection[0]]
                end_landmark = results.pose_landmarks.landmark[connection[1]]
                
                start_x = int(start_landmark.x * img.shape[1])
                start_y = int(start_landmark.y * img.shape[0])
                end_x = int(end_landmark.x * img.shape[1])
                end_y = int(end_landmark.y * img.shape[0])
                
                cv2.line(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 4)

                angle = np.degrees(np.arctan2(end_y - start_y, end_x - start_x))
                angle = angle + 360 if angle < 0 else angle
                mapped_angle = map_angle(angle)
                # Apply rolling average filter to joint angles
                self.leftShoulderToElbowMvgAvg.add_value(mapped_angle)

                filtered_angle = self.leftShoulderToElbowMvgAvg.get_filtered_value()
                
                
                if self.landmark_mapping.get(connection[0]) == "LEFT SHOULDER" and \
                   self.landmark_mapping.get(connection[1]) == "LEFT ELBOW":
                    print("{}\t- {}\t: {}".format(self.landmark_mapping.get(connection[0]), \
                                                  self.landmark_mapping.get(connection[1]), filtered_angle).expandtabs(9))
                
        except Exception as e:
            print("Error:", e)
        print()

        return img

def map_angle(angle):
    mapped_angle = angle - 270
    mapped_angle %= 360
    return mapped_angle

# Example usage:
def main():
    # Initialize PoseDetector object
    pose_detector = PoseDetector(window_size=5)

    # Take video input for pose detection
    cap = cv2.VideoCapture(0)  # You can put here video of your choice ("sampleVideo.mp4")

    # Read each frame/image from capture object
    while True:
        ret, img = cap.read()
        # Resize image/frame so we can accommodate it on our screen
        img = cv2.resize(img, (640, 480))

        # Process frame
        processed_img = pose_detector.process_frame(img)

        # Display pose on original video/live stream
        cv2.imshow("Pose Estimation", processed_img)

        # Exit loop if any key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

