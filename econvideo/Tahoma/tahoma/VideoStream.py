import cv2
import os
import numpy as np
import torch

class VideoStream:
    def __init__(self, list_of_img_size, list_of_preprocessor, db_path, list_of_mean, list_of_std, interval):
        self.db_path = db_path
        self.sizes = list_of_img_size
        self.preprocessors = list_of_preprocessor
        self.size_to_cap = {} #size to videocapture object
        self.list_of_mean = []
        self.list_of_std = []
        self.interval = interval
        for size in self.sizes:
            key = 'x'.join([str(a) for a in size])
            self.list_of_mean.append(list_of_mean[key])
            self.list_of_std.append(list_of_std[key])
            if key not in self.size_to_cap:
                self.size_to_cap[key] = None
        path = os.path.join(self.db_path, 'x'.join([str(a) for a in list_of_img_size[0]]))
        self.video_names = os.listdir(path)
        # print('video_names:', self.video_names)
        self.video_index = 0

    def __iter__(self):
        for key in self.size_to_cap.keys():
            self.size_to_cap[key] = cv2.VideoCapture(os.path.join(self.db_path, key, self.video_names[self.video_index]))
            # print(self.video_names[self.video_index])
        # self.video_index += 1
        return self

    def _get_new_caps(self):
        self.video_index += 1
        for key in self.size_to_cap.keys():
            self.size_to_cap[key].release()
            if self.video_index < len(self.video_names):
                self.size_to_cap[key] = cv2.VideoCapture(os.path.join(self.db_path, key, self.video_names[self.video_index]))
                # print(self.video_names[self.video_index])
            else: 
                raise StopIteration()
    
    def _preprocess(self, frame, preprocessor, mean, std, idx):
        frame = frame.astype(np.float)
        # Standardize
        if preprocessor == 'ColorImage':
            for i in range(0, 3):
                arr = frame[:, :, i]
                new_arr = (arr - mean[i]) / std[i]
                frame[:, :, i] = new_arr
        elif preprocessor == 'BWImage':
            arr = frame[:, :, 0]
            new_arr = (arr - mean[3]) / std[3]
            frame[:, :, 0] = new_arr
        elif preprocessor == 'BlueChannel':
            arr = frame[:, :, 0]
            new_arr = (arr - mean[0]) / std[0]
            frame[:, :, 0] = new_arr
        elif preprocessor == 'GreenChannel':
            arr = frame[:, :, 0]
            new_arr = (arr - mean[1]) / std[1]
            frame[:, :, 0] = new_arr
        else:
            arr = frame[:, :, 0]
            new_arr = (arr - mean[2]) / std[2]
            frame[:, :, 0] = new_arr
               
        # Transpose
        frame = frame.transpose(2, 0, 1)
        return torch.from_numpy(frame).float().unsqueeze(0)

    def __next__(self):
        size_to_image = {}
        while len(size_to_image.keys()) == 0:
            for key in self.size_to_cap.keys():
                cap = self.size_to_cap[key]
                if not cap.isOpened():
                    # print('cap is not opened')
                    self._get_new_caps()
                    break
                ret, frame = cap.read()
                if ret == True:
                    size_to_image[key] = frame
                else:
                    # print('get new caps')
                    self._get_new_caps()
                    break
        # print (size_to_image)
        
        frame_list = [size_to_image['x'.join([str(a) for a in size])] for size in self.sizes]

        for i in range(len(frame_list)):
            if self.preprocessors[i] == 'BWImage':
                frame_list[i] = cv2.cvtColor(frame_list[i], cv2.COLOR_BGR2GRAY)
                frame_list[i] = np.expand_dims(frame_list[i], axis=2)
            elif self.preprocessors[i] == 'BlueChannel':
                frame_list[i] = frame_list[i][:, :, 0]
                frame_list[i] = np.expand_dims(frame_list[i], axis=2)
            elif self.preprocessors[i] == 'GreenChannel':
                frame_list[i] = frame_list[i][:, :, 1]
                frame_list[i] = np.expand_dims(frame_list[i], axis=2)
            elif self.preprocessors[i] == 'RedChannel':
                frame_list[i] = frame_list[i][:, :, 2]
                frame_list[i] = np.expand_dims(frame_list[i], axis=2)
            frame_list[i] = self._preprocess(frame_list[i], self.preprocessors[i], self.list_of_mean[i], self.list_of_std[i], i)
        return frame_list
        

if __name__ == '__main__':
    my_stream = VideoStream(list_of_img_size=[224, 224, 224], list_of_preprocessor=['ColorImage', 'ColorImage', 'ColorImage'], db_path='/z/analytics/resized_videos/')
    stream_it = iter(my_stream)
    i = 0
    for frames in stream_it:
        for j, frame in enumerate(frames):
            cv2.imwrite("/z/analytics/test/{}_{}.jpg".format(i, j), frame)
        i += 1