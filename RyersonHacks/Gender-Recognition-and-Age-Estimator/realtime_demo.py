"""
Face detection
"""
import cv2
import os
from time import sleep
import numpy as np
import argparse
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
import pandas as pd
import datetime


class FaceCV(object):
    
    today = str(datetime.date.today())

    count_0_9 = 0
    count_10_19 = 0
    count_20_29 = 0
    count_30_39 = 0
    count_40_49 = 0
    count_50_59 = 0
    count_60_69 = 0
    count_70_79 = 0
    count_80 = 0

    count_F = 0
    count_M = 0

    df1 = pd.DataFrame(columns = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'])
    df2 = pd.DataFrame(columns = ['Female', 'Male'])
  
    """
    Singleton class for face recongnition task
    """

    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceCV, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, face_size=64):
        self.face_size = face_size
        self.model = WideResNet(face_size, depth=depth, k=width)()
        model_dir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
        fpath = get_file('weights.18-4.06.hdf5',
                         '/Users/lauradang/RyersonHacks/Gender-Recognition-and-Age-Estimator/pretrained_models/weights.18-4.06.hdf5',
                         cache_subdir=model_dir)
        self.model.load_weights(fpath)

    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def crop_face(self, imgarray, section, margin=40, size=64):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w,h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    def detect_face(self):
        
        face_cascade = cv2.CascadeClassifier('/Users/lauradang/RyersonHacks/Gender-Recognition-and-Age-Estimator/pretrained_models/haarcascade_frontalface_alt.xml')

        # 0 means the default video capture device in OS
        video_capture = cv2.VideoCapture(0)
        # infinite loop, break by key ESC
        while True:

            try:
                if not video_capture.isOpened():
                    sleep(5)
                # Capture frame-by-frame
                ret, frame = video_capture.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=10,
                    minSize=(self.face_size, self.face_size)
                )
                if faces is not ():
                    
                    # placeholder for cropped faces
                    face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
                    for i, face in enumerate(faces):
                        face_img, cropped = self.crop_face(frame, face, margin=40, size=self.face_size)
                        (x, y, w, h) = cropped
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                        face_imgs[i,:,:,:] = face_img
                
                    if len(face_imgs) > 0:
                        # predict ages and genders of the detected faces
                        results = self.model.predict(face_imgs)
                        predicted_genders = results[0]
                        ages = np.arange(0, 101).reshape(101, 1)
                        predicted_ages = results[1].dot(ages).flatten()
                    
                    # draw results
                    for i, face in enumerate(faces):
                        label = "{}, {}".format(int(predicted_ages[i]),
                                                "F" if predicted_genders[i][0] > 0.5 else "M")
                        
                        self.draw_label(frame, (face[0], face[1]), label)

                        data = label.split(", ")

                        if data[1] == 'F':
                            self.count_F += 1
                            if int(data[0]) >= 0 and int(data[0]) <= 9:
                                self.count_0_9 += 1
                            elif int(data[0]) >= 10 and int(data[0]) <= 19:
                                self.count_10_19 += 1
                            elif int(data[0]) >= 20 and int(data[0]) <= 29:
                                self.count_20_29 += 1
                            elif int(data[0]) >= 30 and int(data[0]) <= 39:
                                self.count_30_39 += 1
                            elif int(data[0]) >= 40 and int(data[0]) <= 49:
                                self.count_40_49 += 1
                            elif int(data[0]) >= 50 and int(data[0]) <= 59:
                                self.count_50_59 += 1
                            elif int(data[0]) >= 60 and int(data[0]) <= 69:
                                self.count_60_69 += 1
                            elif int(data[0]) >= 70 and int(data[0]) <= 79:
                                self.count_70_79 += 1
                            elif int(data[0]) >= 80:
                                self.count_80 += 1
                        elif data[1] == 'M':
                            self.count_M += 1
                            if int(data[0]) >= 0 and int(data[0]) <= 9:
                                self.count_0_9 += 1
                            elif int(data[0]) >= 10 and int(data[0]) <= 19:
                                self.count_10_19 += 1
                            elif int(data[0]) >= 20 and int(data[0]) <= 29:
                                self.count_20_29 += 1
                            elif int(data[0]) >= 30 and int(data[0]) <= 39:
                                self.count_30_39 += 1
                            elif int(data[0]) >= 40 and int(data[0]) <= 49:
                                self.count_40_49 += 1
                            elif int(data[0]) >= 50 and int(data[0]) <= 59:
                                self.count_50_59 += 1
                            elif int(data[0]) >= 60 and int(data[0]) <= 69:
                                self.count_60_69 += 1
                            elif int(data[0]) >= 70 and int(data[0]) <= 79:
                                self.count_70_79 += 1
                            elif int(data[0]) >= 80:
                                self.count_80 += 1

                    cv2.imshow('Keras Faces', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
                    break
            except:
                break

        self.df1.loc[self.today, '0-9'] = self.count_0_9
        self.df1.loc[self.today, '10-19'] = self.count_10_19
        self.df1.loc[self.today, '20-29'] = self.count_20_29
        self.df1.loc[self.today, '30-39'] = self.count_30_39
        self.df1.loc[self.today, '40-49'] = self.count_40_49
        self.df1.loc[self.today, '50-59'] = self.count_50_59
        self.df1.loc[self.today, '60-69'] = self.count_60_69
        self.df1.loc[self.today, '70-79'] = self.count_70_79
        self.df1.loc[self.today, '80+'] = self.count_80

        self.df2.loc[self.today, 'Female'] = self.count_F
        self.df2.loc[self.today, 'Male'] = self.count_M

        self.df1.to_excel('/Users/lauradang/RyersonHacks/data/age.xlsx')
        self.df2.to_excel('/Users/lauradang/RyersonHacks/data/gender.xlsx')

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    depth = args.depth
    width = args.width

    face = FaceCV(depth=depth, width=width)

    face.detect_face()

if __name__ == "__main__":
    main()
    
    





