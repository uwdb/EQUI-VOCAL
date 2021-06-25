import json
import os

import cv2
import numpy as np

class VideoCapture(object):
    def __init__(self, dir_name, index_fname=None):
        self.dir_name = dir_name
        # FIXME: horribad hack
        if index_fname is None:
            if dir_name[-1] == '/':
                dir_name = dir_name[:-1]
            index_fname = dir_name + '.json'
        if not os.path.isfile(index_fname):
            self.create_index(index_fname)
        self.read_index(index_fname)

        self._decoder_ind = 0
        self._dec_frame_ind = 0
        self._frame_ind = 0
        self._cap = cv2.VideoCapture(self.fnames[0])

    def _set_fnames(self):
        fnames = os.listdir(self.dir_name)
        fnames.sort(key=lambda x: int(x.split('.')[0]))
        fnames = list(map(lambda x: os.path.join(self.dir_name, x), fnames))
        self.fnames = fnames

    def create_index(self, index_fname):
        def get_nb_frames(fname):
            cap = cv2.VideoCapture(fname)
            nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return nb_frames
        def get_height_width(fname):
            cap = cv2.VideoCapture(fname)
            _, frame = cap.read()
            cap.release()
            return frame.shape[0:2]
        def get_fps(fname):
            cap = cv2.VideoCapture(fname)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return fps

        self._set_fnames()
        height, width = get_height_width(self.fnames[0])
        fps = get_fps(self.fnames[0])
        nb_frames = list(map(get_nb_frames, self.fnames))
        cum_frames = list(map(int, np.cumsum(nb_frames)))

        json_dict = {'height': height, 'width': width, 'fps': fps,
                     'dir_name': self.dir_name, 'cum_frames': cum_frames}
        with open(index_fname, 'w') as outfile:
            json.dump(json_dict, outfile)

    def read_index(self, index_fname):
        with open(index_fname, 'r') as infile:
            json_dict = json.load(infile)
        self.__dict__.update(json_dict)
        self._set_fnames()

    def read(self):
        ret, frame = self._cap.read()
        if ret:
            self._frame_ind += 1
            self._dec_frame_ind += 1
        else:
            self._decoder_ind += 1
            if self._decoder_ind < len(self.fnames):
                self._cap.release()
                self._cap = cv2.VideoCapture(self.fnames[self._decoder_ind])
                ret, frame = self._cap.read()
        return ret, frame

    def set(self, prop_id, value):
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            new_dec_ind = 0
            while self.cum_frames[new_dec_ind] < value:
                new_dec_ind += 1

            self._decoder_ind = new_dec_ind
            self._frame_ind = int(value)
            self._dec_frame_ind = int(value) - self.cum_frames[new_dec_ind - 1] \
                if new_dec_ind != 0 else int(value)
            self._cap.release()
            self._cap = cv2.VideoCapture(self.fnames[self._decoder_ind])
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._dec_frame_ind)
        else:
            raise NotImplementedError

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            return self._frame_ind
        elif prop_id == cv2.CAP_PROP_FPS:
            return self._cap.get(prop_id)
        elif prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return self.cum_frames[-1]
        else:
            raise NotImplementedError

    def release(self):
        pass
