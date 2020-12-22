import sys
import os
import os.path as osp
import pandas as pd
import time
import torch
import cv2
import numpy as np

from face_detector.face_detector import FaceDetector
from fake_predictor.predictor import Predictor


def get_face(img, landmarks, face_scale=1.0):
    l_x = landmarks[0][0]
    t_y = landmarks[0][1]
    r_x = landmarks[0][2]
    b_y = landmarks[0][3]

    center_x, center_y = (l_x + r_x) / 2, (t_y + b_y) / 2

    face_size = face_scale * max(r_x - l_x, b_y - t_y)

    new_l_x = max(int(center_x - face_size / 2), 0)
    new_r_x = min(int(center_x + face_size / 2), img.shape[1] - 1)
    new_t_y = max(int(center_y - face_size / 2), 0)
    new_b_y = min(int(center_y + face_size / 2), img.shape[0] - 1)

    face = img[new_t_y: new_b_y, new_l_x: new_r_x]

    return face


def load_file_path(test_data_path):
    test_files = []
    filenames = os.listdir(test_data_path)
    for filename in filenames:
        test_file = os.path.join(test_data_path, filename)
        test_files.append(test_file)
    return test_files


def submit(predictions, test_files):
    test_files = [osp.basename(filepath) for filepath in test_files]
    submission_df = pd.DataFrame({"filename": test_files, "label": predictions})
    submission_df.to_csv("/output/submission.csv", index=False)


    # submission_df.to_csv("/data3/fanhongxing/GeekPwn2020/output/submissions.csv", index=False)


def sample_frame_from_video(path, num_frames, jitter=0, seed=None):
    """Reads frames that are always evenly spaced throughout the video

    Arguments:
        path: the video file
        num_frames: how many frames to read, -1 means the entire video
                    (warning: this will take up a lot of memory!)
        jitter: if not 0, adds small random offsets to the frame indic
                this is useful so we don't always land on even or odd fram
        seed: random seed for jittering; if you set this to a fixed va
              you probably want to set it only on the first video 
    """      
    assert num_frames > 0

    capture = cv2.VideoCapture(path)
    frame_count = int(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) / 1)
    if frame_count <= 0: return None

    frame_idxs = np.linspace(0, frame_count - 1, num_frames, endpoint=True, dtype=np.int)
    if jitter > 0:
        np.random.seed(seed)
        jitter_offsets = np.random.randint(-jitter, jitter, len(frame_idxs))
        frame_idxs = np.clip(frame_idxs + jitter_offsets, 0, frame_count - 1)

    frames = read_frames_at_indices(path, capture, frame_idxs)
    capture.release()
    return frames


def read_frames_at_indices(path, capture, frame_idxs):
    try:
        frames = []
        idxs_read = []
        for frame_idx in range(frame_idxs[0], frame_idxs[-1] + 1):
            # Get the next frame, but don't decode if we're not using it.
            ret = capture.grab()
            if not ret:
                print("Error grabbing frame %d from movie %s" % (frame_idx, path))
                break

                                         
            # Need to look at this frame?
            current = len(idxs_read)
            if frame_idx == frame_idxs[current]:
                ret, frame = capture.retrieve()
                if not ret or frame is None:
                    print("Error retrieving frame %d from movie %s" % (frame_idx, path))
                    break

                #frame = self._postprocess_frame(frame)
                frames.append(frame)
                idxs_read.append(frame_idx)

        if len(frames) > 0:
            return np.stack(frames), idxs_read
        else:
            print("No frames read from movie %s" % path)
            return None
    except:
        print("Exception while reading movie %s" % path)
        return None                                                                              


def predict_on_image(face_detector, predictor, file_path):
    # Read image
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    # Run face detection

    boxes = face_detector.detect(img)
    if boxes is None or len(boxes) == 0:
        return 0.5

    if boxes[0][0] < 0: boxes[0][0] = 0
    if boxes[0][1] < 0: boxes[0][1] = 0
    if boxes[0][2] > img.shape[1] - 1: boxes[0][2] = img.shape[1] - 1
    if boxes[0][3] > img.shape[0] - 1: boxes[0][3] = img.shape[0] - 1
    
    # Run fake prediction
    face_img = img[int(boxes[0][1]): int(boxes[0][3]), int(boxes[0][0]): int(boxes[0][2])]

    prediction = predictor.predict(face_img)
    if prediction > 0.95:
        prediction = 0.95
    elif prediction < 0.05:
        prediction = 0.05
    
    print(file_path,prediction)

    return prediction


def predict_on_video(face_detector, predictor1, file_path, frames_per_video):
    # Sample frames from video
    frames, frame_idx = sample_frame_from_video(file_path, num_frames=frames_per_video)   
    
    frame_predictions = []
    for i, frame in enumerate(frames):
        # Run face detection
        boxes = face_detector.detect(frame)
        if boxes is None or len(boxes) == 0:
            frame_predictions.append(0.5)
            continue

        im_height, im_width, _ = frame.shape

        face_img = get_face(frame, boxes)

        prediction = predictor1.predict(face_img)
        frame_predictions.append(prediction)
    
    if frames is None:
        return 0.5
    prediction = np.median(frame_predictions)
    if prediction > 0.95:
        prediction = 0.95
    elif prediction < 0.05:
        prediction = 0.05
    return prediction


def main(argv):
    """Fake detector
    
    Usage: python test.py [test_data_path]

    It reads video or image files from [test_data_path] and then make predictions on them.
    Prediction results will be wtitten to a submission.csv file.
    For each filename in the test set, you must predict a probability for the label variable. 
    The file should contain a header and have the following format:
    
    filename,label
    10000.mp4,0
    10001.jpg,0.5
    10002.mp4,1
    etc.
    
    """
    # import datetime
    # starttime = datetime.datetime.now()

    if(len(argv) != 2):
        print('Usage: %s [test_data_path]' % argv[0])
        return

    # Set parameters
    test_data_path = argv[1]
    face_detector_model_path = './weights/mobilenet0.25_Final.pth'
    pic_fake_predictor_model_path = './weights/5GAN1024png15000_xception.ckpt'
    # pic_fake_predictor_model_path = './weights/2_efficient_EndEpoch.ckpt'
    video_fake_predictor_model_path = './weights/2_efficient_EndEpoch.ckpt'
    frames_per_video = 5

    # Build models 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    face_detector = FaceDetector(face_detector_model_path, device = device)
    pic_predictor = Predictor(pic_fake_predictor_model_path, device = device, arch = 'xception')
    # pic_predictor = Predictor(pic_fake_predictor_model_path, device = device, arch = 'efficientnet-b3')
    video_predictor = Predictor(video_fake_predictor_model_path, device = device, arch = 'efficientnet-b3')

    # Run prediction
    predictions = []    
    test_files = load_file_path(test_data_path)
    for file_path in test_files:
        print('Processing {}'.format(file_path))
        if file_path.endswith('.mp4'):
            prediction = predict_on_video(face_detector, video_predictor, file_path, frames_per_video)
            predictions.append(prediction)
        elif file_path.split('.')[-1] in ['png','jpg']:
            prediction = predict_on_image(face_detector, pic_predictor, file_path)
            predictions.append(prediction)
        

    # Write submission.csv
    #submit(predictions, test_files)


    # endtime = datetime.datetime.now()
    # print((endtime - starttime).seconds)

if __name__ == '__main__':
    main(sys.argv)