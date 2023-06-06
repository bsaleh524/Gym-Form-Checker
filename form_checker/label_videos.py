import cv2
import numpy as np
import os

class video_labeler():

    def __init__(self,
                 label_map = {
                    ord('1'): 'UP',
                    ord('2'): 'DOWN',
                    ord('3'): 'LIFTING',
                    ord('4'): 'GOOD',
                    ord('5'): 'BAD',
                    ord('s'): 'SKIP'
                    }):
        
        # Create dictionary to map key codes to labels
        self.label_map = label_map

    def labelvid(self,
                 video_path: str,
                 label_path: str):

        # Initialize video capture
        cap = cv2.VideoCapture(video_path)

        # Check if video capture is successful
        if not cap.isOpened():
            print("Error opening video file")
            exit()

        # Create empty list to store labeled frames
        labeled_frames = []

        # Process frames
        frame_id = 0
        while cap.isOpened():
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break

            # Display frame
            cv2.imshow('Video', frame)

            # Collect labels for the current frame
            labels_up_down_lifting = []
            labels_good_bad = []
            while True:
                key = cv2.waitKey(0)
                if key in self.label_map:
                    label = self.label_map[key]
                    if label == 'SKIP':
                        print(f"Frame {frame_id} skipped")
                        break
                    if label in ['UP', 'DOWN', 'LIFTING']:
                        if label not in labels_up_down_lifting:
                            labels_up_down_lifting.append(label)
                            print(f"Label {label} added for Frame {frame_id}")
                        else:
                            print("You can only select one label between UP, DOWN, LIFTING.")
                    elif label in ['GOOD', 'BAD']:
                        if label not in labels_good_bad:
                            labels_good_bad.append(label)
                            print(f"Label {label} added for Frame {frame_id}")
                        else:
                            print("You can only select one label between GOOD and BAD.")
                elif key == 27:  # Press ESC to exit
                    break

            # Save labeled frame if it has the required labels
            if len(labels_up_down_lifting) == 1 and len(labels_good_bad) == 1:
                labeled_frames.append((frame_id, labels_up_down_lifting[0], labels_good_bad[0]))

            frame_id += 1

        # Release video capture and close OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

        # Save labeled frames in Imagenet 1.0 format
        with open(label_path, 'w') as f:
            for frame_id, label1, label2 in labeled_frames:
                line = f"{frame_id} {label1} {label2}\n"
                f.write(line)
        
        print(f"Labeled frames saved to {label_path}")

if __name__ == '__main__':
    
    # Init our class
    video_directory = "form_checker/web_scraper/reddit_videos/"
    AI = video_labeler()
    
    # Begin labeling each video
    for filename in os.listdir(video_directory):
        file_path = os.path.join(video_directory, filename)
        abspath = os.path.abspath(file_path)
        # checking if it is a file
        if os.path.isfile(abspath):
            # CHeck if label file already exists
            label_file = os.path.join(video_directory,
                                      "labels/",
                                      filename.split('.mp4')[0] + ".txt")
            
            if not os.path.isfile(os.path.abspath(label_file)):
                AI.labelvid(abspath, label_file)
# Check if this works