from tracemalloc import start
import cv2
import mediapipe as mp
import numpy as np
import math
import matplotlib.pyplot as plt



imageProc = False
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
        self.angleContainertoSend = []

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
            (13, 11),  # Left shoulder to left elbow
            (13, 15),  # Left elbow to left wrist
            (14, 12),  # Right shoulder to right elbow
            (14, 16)   # Right elbow to right wrist
        ]
        self.angleContainer = [
            (14,12,16), # right shoulder to right wrist
            (12,14,11), # right elbow to right shoulder
            (11,12,13), # left shoulder to left elbow
            (13,11,15) # left elbow to left wrist
        ]
    
    def process_frame(self, img):
        self.angleContainertoSend.clear()
        # Do pose detection
        results = self.pose.process(img)
        if results is None:
            return None

        # Draw landmarks and connections
        try:

            for landmark_idx, _ in self.landmark_mapping.items():
                landmark = results.pose_landmarks.landmark[landmark_idx]
                x = int(landmark.x * img.shape[1])
                y = int(landmark.y * img.shape[0])
                cv2.circle(img, (x, y), 4, (0, 0, 255), 5)
                cv2.putText(img, str(landmark_idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (255, 0, 0), 2)

            #Drawing lines for each connections
            for connection in self.connections:
                start_landmark = results.pose_landmarks.landmark[connection[0]]
                end_landmark = results.pose_landmarks.landmark[connection[1]]
                start_landmark_name = self.landmark_mapping.get(connection[0])
                end_landmark_name = self.landmark_mapping.get(connection[1])
                
                start_x, start_y, start_z = int(start_landmark.x * img.shape[1]), int(start_landmark.y * img.shape[0]), start_landmark.z
                end_x, end_y, end_z = int(end_landmark.x * img.shape[1]), int(end_landmark.y * img.shape[0]), end_landmark.z
                cv2.line(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 4)
  
                # print("{}\t- {}\t: sX:{}, sY:{}, sZ:{}, eX:{}, eY:{}, eZ:{}".format(start_landmark_name, end_landmark_name, start_x, start_y, start_z, end_x, end_y, end_z).expandtabs(9))

            # Calculate angles using the calculate_angle_3d function
            for angle_points in self.angleContainer:
                point1 = results.pose_landmarks.landmark[angle_points[0]]
                point2 = results.pose_landmarks.landmark[angle_points[1]]
                point3 = results.pose_landmarks.landmark[angle_points[2]]

                angle = self.calculate_angle_3d(point1.x, point1.y, point1.z, point2.x, point2.y, point2.z, point3.x, point3.y, point3.z)
                angle2d = self.find_angle2d(point1.x, point1.y, point2.x, point2.y, point3.x, point3.y)
                # print(f"3D Angle between points {angle_points}: {angle} degrees")
                # print(f"2D Angle between points {angle_points}: {angle2d} degrees")

                self.angleContainertoSend.append(round(angle2d))

        except Exception as e:
            print(str(e))

        return img
    
    def stringify_angles(self)->str:
        # right shoulder to right wrist
        # right elbow to right shoulder
        # left shoulder to left elbow
        # left elbow to left wrist  
        result = ",".join(map(str, self.angleContainertoSend))
        return result

    def find_angle2d(self, x1, y1, x2, y2, x3, y3):
        # Vector u
        ux, uy = x2 - x1, y2 - y1
        # Vector v
        vx, vy = x3 - x1, y3 - y1
        
        # Dot product
        dot_product = ux * vx + uy * vy
        # Magnitudes
        magnitude_u = math.sqrt(ux**2 + uy**2)
        magnitude_v = math.sqrt(vx**2 + vy**2)
        
        # Cosine of the angle
        cos_theta = dot_product / (magnitude_u * magnitude_v)
        
        # Angle in radians
        theta_radians = math.acos(cos_theta)
        
        # Convert to degrees
        theta_degrees = math.degrees(theta_radians)
        
        return theta_degrees

    def calculate_angle_3d(self, x1, y1, z1, x2, y2, z2, x3, y3, z3):
        # Calculate vectors between the points
        vec1 = np.array([x1 - x2, y1 - y2, z1 - z2])
        vec2 = np.array([x3 - x2, y3 - y2, z3 - z2])

        # Calculate the dot product of the two vectors
        dot_product = np.dot(vec1, vec2)

        # Calculate the magnitudes of the vectors
        mag1 = np.linalg.norm(vec1)
        mag2 = np.linalg.norm(vec2)

        # Calculate the cosine of the angle between the vectors
        cos_angle = dot_product / (mag1 * mag2)

        # Calculate the angle in radians
        angle_radians = np.arccos(cos_angle)

        # Convert the angle to degrees
        angle_degrees = np.degrees(angle_radians)
        # print(f"The angle is {angle_degrees:.2f} degrees")
        return angle_degrees

    def calculate_angle_3d2(self, x1, y1, z1, x2, y2, z2, x3, y3, z3):
        # Vector u: Point 1 to Point 2
        ux, uy, uz = x2 - x1, y2 - y1, z2 - z1
        # Vector v: Point 1 to Point 3
        vx, vy, vz = x3 - x1, y3 - y1, z3 - z1
        
        # Dot product
        dot_product = ux * vx + uy * vy + uz * vz
        # Magnitudes
        magnitude_u = math.sqrt(ux**2 + uy**2 + uz**2)
        magnitude_v = math.sqrt(vx**2 + vy**2 + vz**2)
        
        # Calculate cos(theta)
        cos_theta = dot_product / (magnitude_u * magnitude_v)
        
        # Handle numerical issues (rounding errors can push cos_theta slightly out of range)
        cos_theta = max(min(cos_theta, 1.0), -1.0)
    
    # Angle in radians
        theta_radians = math.acos(cos_theta)
        
        # Convert to degrees
        theta_degrees = math.degrees(theta_radians)
        # print(f"The angle is {theta_degrees:.2f} degrees")
        return theta_degrees

    def plot_landmark_values(self):
        # Data for plotting
        labels = self.labels
        x = self.x
        y = self.y
        z = self.z

        # Plotting x, y, and z values for start and end landmarks with lines connecting consecutive points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot points
        ax.scatter(x, y, z, c='r', marker='o')

        # Connect consecutive points with lines
        for i in range(len(x) - 1):
            ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], marker='o')

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

# Example usage:

if __name__ == "__main__":
    print("This file is not intented for direct run!...")