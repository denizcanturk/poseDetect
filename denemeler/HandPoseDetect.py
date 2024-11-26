import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

pose = mp_pose.Pose()

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
    (11, 12)   # Left shoulder to right shoulder
]



# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    image = cv2.resize(image, (900, 600))
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.reboot
    #image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)
    pose_results = pose.process(image)

    if pose_results is None:
      continue

    try:
        # Limit pose detection to one person
        if pose_results.pose_landmarks is not None:
            # Draw landmarks
            for landmark_idx, landmark_name in landmark_mapping.items():
                landmark = pose_results.pose_landmarks.landmark[landmark_idx]
                # Draw the landmark on the original image
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 6, (0, 0, 255), 6)
                # Print landmark name and coordinates
                print(f"{landmark_name}\t: ({x}, {y})")
            
            # Draw connections
            for connection in connections:
                start_landmark = pose_results.pose_landmarks.landmark[connection[0]]
                end_landmark = pose_results.pose_landmarks.landmark[connection[1]]
                
                start_x = int(start_landmark.x * image.shape[1])
                start_y = int(start_landmark.y * image.shape[0])
                end_x = int(end_landmark.x * image.shape[1])
                end_y = int(end_landmark.y * image.shape[0])
                
                cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 4)
        
        # Limit hand detection to two hands
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks[:2]:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
                
    except Exception as e:
        print(f"Error: {e}")
        print("Detection is lost")

    

    ############################################################
    # FROM THIS POINT IT ONLY SHOWS THE WHITE IMAGE WITH MARKS #
    ############################################################
    h, w, c = image.shape   # get shape of original frame
    opImg = np.zeros([h, w, c], dtype=np.uint8)  # create blank image with original frame size
    opImg.fill(0)

    try:
    # Draw specific landmarks on the blank image
        for landmark_idx, landmark_name in landmark_mapping.items():
            landmark = pose_results.pose_landmarks.landmark[landmark_idx]
            # Draw the landmark on the original image
            x = int(landmark.x * opImg.shape[1])
            y = int(landmark.y * opImg.shape[0])
            cv2.circle(opImg, (x, y), 6, (0, 0, 255), 6)
            
            # Print landmark name and coordinates
            #print(f"{landmark_name}\t: ({x}, {y})")
            
            # Draw connections
        for connection in connections:
            start_landmark = pose_results.pose_landmarks.landmark[connection[0]]
            end_landmark = pose_results.pose_landmarks.landmark[connection[1]]
            
            start_x = int(start_landmark.x * opImg.shape[1])
            start_y = int(start_landmark.y * opImg.shape[0])
            end_x = int(end_landmark.x * opImg.shape[1])
            end_y = int(end_landmark.y * opImg.shape[0])
            
            cv2.line(opImg, (start_x, start_y), (end_x, end_y), (0, 255, 0), 4)
            
            # Draw the hand annotations on the image.
            opImg.flags.writeable = True
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        opImg, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
        
    except:  # noqa: E722
        print("Detection is lost")
    # Display extracted pose on blank image
    cv2.imshow("Extracted Pose", opImg)

    ############################################################
    #  TO THIS POINT IT ONLY SHOWS THE WHITE IMAGE WITH MARKS  #
    ############################################################

    cv2.imshow('Pose and Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()