import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd


class form_checker():
    '''
    Class that collects data from multiple deadlifting videos and
    creates a pandas dataframe for each individual frame that includes:
    - file name
    - Pose locations
    - Lifting stage label (Provided by user)
    - Form label (Provided by user)

    
    '''
    def __init__(self,
                 detect_conf: int = 0.5,
                 track_conf: int = 0.5):
        
        '''Initialize our class that holds our Pose model parameters.'''

        self.pose_drawing = mp.solutions.drawing_utils
        self.pose_tools = mp.solutions.pose
        self.pose_model = self.pose_tools.Pose(min_detection_confidence=detect_conf,
                                               min_tracking_confidence=track_conf)
        
        # initialize dataframe for data storage
        pose_name_array = [pose_name.name for _, pose_name in enumerate(self.pose_tools.PoseLandmark)] + ["video_file", "lifting_stage", "form_check"]
        self.dataframe = pd.DataFrame(columns=pose_name_array)

    def check(self, video_file: str):
        '''Open given video file,
        play video file with poses on top.'''
        capture = cv2.VideoCapture(video_file)
        print(f"Video Loaded")
        while capture.isOpened():
            success, frame = capture.read()
            if success:
                # Recolor out image from BGR to RGB. MediaPose takes in RGB
                # and be default, cv2 outputs as a BGR.
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False # Prevent video feed from being written to when converting

                ## Make Detection. Get detections and store in `results`
                results = self.pose_model.process(image)

                ## Recolor back to BGR to show on the feed
                image.flags.writeable = True # Allow the current frame to be written over, now that we have detections
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                ### Extract Landmarks if present
                try:
                    landmarks = results.pose_landmarks.landmark

                    #### Get Coordinates
                    left_shoulder = [landmarks[self.pose_tools.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.pose_tools.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[self.pose_tools.PoseLandmark.LEFT_ELBOW.value].x, landmarks[self.pose_tools.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[self.pose_tools.PoseLandmark.LEFT_WRIST.value].x, landmarks[self.pose_tools.PoseLandmark.LEFT_WRIST.value].y]
                
                    #### Calculate angle
                    angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)

                    #### Visualize our value to the screen
                    cv2.putText(image, str(angle),
                                tuple(np.multiply(left_elbow, [capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                                            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)]).astype(int)),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                
                except:
                    pass


                ## Render our detections onto the image
                self.pose_drawing.draw_landmarks(image,
                                        results.pose_landmarks, # Pass specific landmark coordinates
                                        self.pose_tools.POSE_CONNECTIONS,
                                        self.pose_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                        self.pose_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) # Pass connections(right shoulder --> right elbow)

                cv2.imshow('Mediapipe Feed', image)

                # For exiting the live feed
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                print("Video Ended")
                capture.release()
                cv2.destroyAllWindows()
                break


    def collect_data(self,
                     video_directory: str):
        '''
        This function will open our video file and append a new row
        to the dataframe
        '''

        # iterate over files in
        # that directory
        for filename in os.listdir(video_directory):
            file_path = os.path.join(video_directory, filename)
            abspath = os.path.abspath(file_path)
            # checking if it is a file
            if os.path.isfile(abspath):
                
                # Open video file
                capture = cv2.VideoCapture(abspath)

                while capture.isOpened():
                    success, frame = capture.read()
                    if success:
                        # Recolor out image from BGR to RGB. MediaPose takes in RGB
                        # and be default, cv2 outputs as a BGR.
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False # Prevent video feed from being written to when converting

                        ## Make Detection. Get detections and store in `results`
                        results = self.pose_model.process(image)

                        ## Recolor back to BGR to show on the feed
                        image.flags.writeable = True # Allow the current frame to be written over, now that we have detections
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                        ### Extract Landmarks if present
                        # try:
                        # Get all joint positions
                        landmarks = results.pose_landmarks.landmark
                            
                        ## Render our detections onto the image
                        self.pose_drawing.draw_landmarks(image,
                                                results.pose_landmarks, # Pass specific landmark coordinates
                                                self.pose_tools.POSE_CONNECTIONS,
                                                self.pose_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                                self.pose_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) # Pass connections(right shoulder --> right elbow)

                        ## Ask user for labeling:

                        ## DOWN, UP, or LIFTING?

                        ## GOOD_FORM or BAD_FORM

                        ## Append all info to dataframe
                        
                        # except:
                        #     pass


                        

                        cv2.imshow('Mediapipe Feed', image)

                        # For exiting the live feed
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
                    else:
                        print("Video Ended")
                        capture.release()
                        cv2.destroyAllWindows()
                        break


    def _calculate_angle(self, a,b,c):
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
    AI.collect_data("form_checker/web_scraper/reddit_videos/")
