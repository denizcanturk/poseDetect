from poseLibrary import PoseDetector, imageProc
from UDPManager import UDPManager
import cv2

# Example usage:
def main():
    # Initialize PoseDetector object
    pose_detector = PoseDetector(window_size=10)
    udp_manager = UDPManager("127.0.0.1", 12345)
    # Take video input for pose detection
    #cap = cv2.VideoCapture(0)  # You can put here video of your choice ("sampleVideo.mp4")
    
    #for video file
    #cap = cv2.VideoCapture("sampleVideo.mp4")
    if imageProc:
        img = cv2.imread("poses\\12.jpeg")
        img = cv2.resize(img, (612,808))
        processed_img = pose_detector.process_frame(img)
        print(pose_detector.stringify_angles())
        udp_manager.send_data(pose_detector.stringify_angles())

        cv2.imshow("Pose Estimation", processed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    else:
        cap = cv2.VideoCapture(0)

        # Read each frame/image from capture object
        while True:
            ret, img = cap.read()
            # Resize image/frame so we can accommodate it on our screen
            img = cv2.resize(img, (640,480))

            # Process frame
            processed_img = pose_detector.process_frame(img)
            print(pose_detector.stringify_angles())
            udp_manager.send_data(pose_detector.stringify_angles())
            # Display pose on original video/live stream
            cv2.imshow("Pose Estimation", processed_img)
            # if counter >=300:
            #     break
            # counter +=1"
            # Exit loop if any key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()