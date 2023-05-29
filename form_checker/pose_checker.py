import cv2
import mediapipe as mp
import numpy as np
import os



mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class form_checker():

    def __init__(self,
                 detect_conf: int = 0.5,
                 track_conf: int = 0.5):
        ''' Initialize our class that holds our Pose model parameters.'''

        self.pose_drawing = mp.solutions.drawing_utils
        self.pose_tools = mp.solutions.pose
        self.pose_model = self.pose_tools.Pose(min_detection_confidence=detect_conf,
                                               min_tracking_confidence=track_conf)

    def check(self, video_file: str):
        '''Open given video file,
        play video file with poses on top.'''
        capture = cv2.VideoCapture(video_file)
        with self.pose_model as PM:
            while capture.isOpened():
                _, frame = capture.read()

                # Recolor out image from BGR to RGB. MediaPose takes in RGB
                # and be default, cv2 outputs as a BGR.
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False # Prevent video feed from being written to when converting

                ## Make Detection. Get detections and store in `results`
                results = PM.process(image)

                ## Recolor back to BGR to show on the feed
                image.flags.writeable = True # Allow the current frame to be written over, now that we have detections
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                ### Extract Landmarks if present
                try:
                    landmarks = results.pose_landmarks.landmark

                    #### Get Coordinates
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                    #### Calculate angle
                    angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)

                    #### Visualize our value to the screen
                    cv2.putText(image, str(angle),
                                tuple(np.multiply(left_elbow, [capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                                            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)]).astype(int)),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                
                except:
                    pass


                ## Render our detections onto the image
                mp_drawing.draw_landmarks(image,
                                        results.pose_landmarks, # Pass specific landmark coordinates
                                        mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) # Pass connections(right shoulder --> right elbow)

                cv2.imshow('Mediapipe Feed', image)

                # For exiting the live feed
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            capture.release()
            cv2.destroyAllWindows()

    def _calculate_angle(a,b,c):
        """
        Calculate angle between three joints
        a: first joint: default = 11(left shoulder)
        b: second joint: default = 13(left elbow)
        c: third joint: default = 15 (left wrist)
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        # Convert to radians to be able to calculate our angle
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*(180.0/np.pi))

        # Prevent angle above 180, because we are only human
        if angle > 180:
            angle = 360 - 180
        
        return angle
    
if __name__ == '__main__':
    
    # Init our class
    AI = form_checker()
    directory = "form_checker/web_scraper/reddit_videos/"
    # iterate over files in
    # that directory
    for filename in os.listdir(directory):
        fullfile_path = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(fullfile_path):
            AI.check(fullfile_path)
    
