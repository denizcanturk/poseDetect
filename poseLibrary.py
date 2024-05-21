import cv2
import mediapipe as mp
import numpy as np
counter = 0
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
        return sum(self.values) / len(self.values)
    
#-----------------------------------------------

class PoseDetector:
    def __init__(self, window_size:int=100):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose()
        self.window_size = window_size
        self.slopeContainer = []
        self.slope = 0
        self.prevSlope = 0
        self.landmark_mapping = {
            11: "LEFT SHOULDER",
            12: "RIGHT SHOULDER",
            13: "LEFT ELBOW",
            14: "RIGHT ELBOW",
            15: "LEFT WRIST",
            16: "RIGHT WRIST"
        }
        self.connections = [
            (11, 12),  # Left shoulder to right shoulder
            (11, 13),  # Left shoulder to left elbow
            (13, 15),  # Left elbow to left wrist
            (12, 14),  # Right shoulder to right elbow
            (14, 16)   # Right elbow to right wrist
        ]
        self.lineContainer = {}
        self.leftShoulderToElbowMvgAvg = RollingAverageFilter(window_size)
        self.leftElbowToWristMvgAvg = RollingAverageFilter(window_size)
        self.rightShoulderToElbowMvgAvg = RollingAverageFilter(window_size)
        self.righttElbowToWristMvgAvg = RollingAverageFilter(window_size)
        self.shoulderToShoulderMvgAvg = RollingAverageFilter(window_size)

        self.filter_mapping = {
            ("LEFT SHOULDER", "LEFT ELBOW"): self.leftShoulderToElbowMvgAvg,
            ("LEFT ELBOW", "LEFT WRIST"): self.leftElbowToWristMvgAvg,
            ("RIGHT SHOULDER", "RIGHT ELBOW"): self.rightShoulderToElbowMvgAvg,
            ("RIGHT ELBOW", "RIGHT WRIST"): self.righttElbowToWristMvgAvg,
            ("LEFT SHOULDER", "RIGHT SHOULDER"): self.shoulderToShoulderMvgAvg
        }
    
    def process_frame(self, img):
        self.slopeContainer.clear()
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
            print("--- Cycle Start Point ---")

            #Drawing lines for each connections
            for connection in self.connections:
                start_landmark = results.pose_landmarks.landmark[connection[0]]
                end_landmark = results.pose_landmarks.landmark[connection[1]]
                start_landmark_name = self.landmark_mapping.get(connection[0])
                end_landmark_name = self.landmark_mapping.get(connection[1])
                
                start_x, start_y, end_x, end_y = int(start_landmark.x * img.shape[1]), int(start_landmark.y * img.shape[0]), \
                                                int(end_landmark.x * img.shape[1]), int(end_landmark.y * img.shape[0])
                # cv2.line(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 4)
                # slope = (end_y - start_y) / (end_x - start_x)
                # if abs(end_x - start_x) > 5:
                #     slope = (end_y - start_y) / (end_x - start_x)
                #     prevSlope = slope
                # else:
                #     slope = prevSlope
                
                    # Apply rolling average filter to joint angles based on landmark names
                key = (start_landmark_name, end_landmark_name)
                
                if key in self.filter_mapping:
                    cv2.line(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 4)
                    if abs(end_x - start_x) > 2 :
                        self.slope = ((end_y) - (start_y)) / (end_x - start_x)
                        self.prevSlope = self.slope
                    else:
                        self.slope = self.prevSlope

                    self.filter_mapping[key].add_value(self.slope)
                    filtered_slope = self.filter_mapping[key].get_filtered_value()
                    self.slopeContainer.append(filtered_slope)
                print("{}\t- {}\t: sX:{}, sY:{}, eX:{}, eY:{}".format(start_landmark_name, end_landmark_name, start_x, (start_y), end_x, (end_y)).expandtabs(9))
            print("Slope Container Values : ")
            print("Shoulder to Shoulder Slope         :", self.slopeContainer[0])
            print("Left Shoulder to Left Elbow Slope  :", self.slopeContainer[1])
            print("Left Elbow to Left Wrist Slope     :", self.slopeContainer[2])
            print("Right Shoulder to Right Elbow Slope:", self.slopeContainer[3])
            print("Right Elbow to Right Wrist Slope   :", self.slopeContainer[4])
            # slopeContainer 0 = Shoulder to Shoulder Slope value
            # slopeContainer 1 = Left Shoulder to Left Elbow Slope value
            # slopeContainer 2 = Left Elbow to Left Wrist Slope value
            # slopeContainer 3 = Right Shoulder to Right Elbow Slope value
            # slopeContainer 4 = Right Elbow to Right Wrist Slope value

            print("Calculated Radian Values : ")
            print("lSlE : ", (self.slopeContainer[0]-self.slopeContainer[1]) / (1+(self.slopeContainer[0]*self.slopeContainer[1])))
            print("lElW : ", (self.slopeContainer[1]-self.slopeContainer[2]) / (1+(self.slopeContainer[1]*self.slopeContainer[2])))
            print("rSrE : ", (self.slopeContainer[0]-self.slopeContainer[3]) / (1+(self.slopeContainer[0]*self.slopeContainer[3])))
            print("rErW : ", (self.slopeContainer[3]-self.slopeContainer[4]) / (1+(self.slopeContainer[3]*self.slopeContainer[4])))

            lSlE = np.degrees((self.slopeContainer[0]-self.slopeContainer[1]) / (1+self.slopeContainer[0]*self.slopeContainer[1]))%360
            lElW = np.degrees((self.slopeContainer[1]-self.slopeContainer[2]) / (1+self.slopeContainer[1]*self.slopeContainer[2]))%360
            rSrE = np.degrees((self.slopeContainer[0]-self.slopeContainer[3]) / (1+self.slopeContainer[0]*self.slopeContainer[3]))%360
            rErW = np.degrees((self.slopeContainer[3]-self.slopeContainer[4]) / (1+self.slopeContainer[3]*self.slopeContainer[4]))%360
            print("Calculated Degree Values : ")
            print("Left Shoulder - Elbow  Deg:", lSlE)
            print("Left Elbow - Wrist     Deg:", lElW) # TODO : PROBLEMATIC CHECK THIS OUT
            print("Right Shoulder - Elbow Deg:", rSrE)
            print("Right Elbow - Shoulder Deg:", rErW) # TODO : PROBLEMATIC CHECK THIS OUT
            print("--- Cycle End Point ---")
        except Exception as e:
            print(str(e))
        print()

        return img

# Example usage:
def main():
    # Initialize PoseDetector object
    pose_detector = PoseDetector(window_size=10)

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
        # if counter >=300:
        #     break
        # counter +=1
        # Exit loop if any key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()