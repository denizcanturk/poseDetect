import cv2
import mediapipe as mp
import numpy as np

#-----------------------------------------------

class RollingAverageFilter:
    def __init__(self, window_size:int):
        self.window_size = window_size
        self.values = []

    def add_value(self, value:float):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

    def get_filtered_value(self)->int:
        if len(self.values) == 0:
            return 0  # Return 0 if there are no values in the window
        return int(sum(self.values) / len(self.values))
    
#-----------------------------------------------

class PoseDetector:
    def __init__(self, window_size:int=5):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose()
        self.window_size = window_size
        self.angleContainer = []
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

        self.filter_mapping = {
            ("LEFT SHOULDER", "LEFT ELBOW"): self.leftShoulderToElbowMvgAvg,
            ("LEFT ELBOW", "LEFT WRIST"): self.leftElbowToWristMvgAvg,
            ("RIGHT SHOULDER", "RIGHT ELBOW"): self.rightShoulderToElbowMvgAvg,
            ("RIGHT ELBOW", "RIGHT WRIST"): self.righttElbowToWristMvgAvg
        }
    def normalizeAngle(self, angle:float)->int: #angle_range:tuple=(270,90)
        mappedAngle = angle - 360
        mappedAngle %= 360
        return int(mappedAngle)
    
    def process_frame(self, img):
        self.angleContainer.clear()
        # Do pose detection
        results = self.pose.process(img)
        if results is None:
            return None

        # Draw landmarks and connections
        try:
            #TODO Elbow-Wrist angles change with the shoulder - elbow angle change. need a
            # wat to fix that issue... 
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
                start_landmark_name = self.landmark_mapping.get(connection[0])
                end_landmark_name = self.landmark_mapping.get(connection[1])
                
                start_x, start_y, end_x, end_y = int(start_landmark.x * img.shape[1]), int(start_landmark.y * img.shape[0]), \
                                                int(end_landmark.x * img.shape[1]), int(end_landmark.y * img.shape[0])
                
                cv2.line(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 4)

                angle = np.degrees(np.arctan2(end_y - start_y, end_x - start_x))
                angle = angle + 360 if angle < 0 else angle
                mapped_angle = self.normalizeAngle(angle)
                
                # Apply rolling average filter to joint angles based on landmark names
                key = (start_landmark_name, end_landmark_name)
                if key in self.filter_mapping:
                    self.filter_mapping[key].add_value(mapped_angle)
                    filtered_angle = self.filter_mapping[key].get_filtered_value()
                    print("{}\t- {}\t: {}".format(start_landmark_name, end_landmark_name, filtered_angle).expandtabs(9))
                    self.angleContainer.append(filtered_angle)
            self.angleContainer[1] = self.angleContainer[1]-self.angleContainer[0]
            self.angleContainer[3] = self.angleContainer[3]-self.angleContainer[2]
            print(self.angleContainer)
        
                
        except Exception as e:
            print("Error:", e)
        print()

        return img

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

