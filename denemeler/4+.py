import cv2
import mediapipe as mp

class poseDetector():

    def __init__(self, mode=False, complex=1, smooth_landmarks=True, segmentation=True, smooth_segmentation=True,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.complex = complex
        self.smooth_landmarks = smooth_landmarks
        self.segmentation = segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawStyle = mp.solutions.drawing_styles
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complex, self.smooth_landmarks, self.segmentation,
                                     self.smooth_segmentation, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

def main():
    cap = cv2.VideoCapture("..//assets//videos//v4.mp4")
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
