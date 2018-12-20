import numpy as np
from skimage.io import imread

from keras.utils import Sequence

from python_pfm import readPFM


def crop_img(img, start_h ,start_w, new_h, new_w):
    return img[start_h:start_h+new_h, start_w:start_w+new_w]

class Dataloader(Sequence):
    def __init__(self, params, files_lists):
        self.mode = params.mode
        self.files_list = files_lists
        self.datapath = params.datapath
        self.batch_size = params.batch_size
        self.crop_height = params.crop_height
        self.crop_width = params.crop_width

        self.li_list, self.ri_list, self.ld_list = [] , [] , []

        count = 0
        with open(self.files_list) as f:
            for line in f:
                    count = count + 1
                    a = line.split()
                    self.li_list.append(a[0])
                    self.ri_list.append(a[1])
                    self.ld_list.append(a[2])
        # store numner of dataset samples
        self.length = count

    def __len__(self):
        # number of steps or iterations per epoch
        return self.length//self.batch_size

    def __getitem__(self,index):
            input_x, target_y = [], []
            for i in range(index * self.batch_size, min((index + 1) * self.batch_size, self.length)):
                img_left = imread(self.datapath + self.li_list[i])
                img_right = imread(self.datapath + self.ri_list[i])
                disp_left = readPFM(self.datapath + self.ld_list[i])

                height = img_left.shape[0]
                width = img_left.shape[1]

                if self.mode is not 'test':
                    limit_h = height - self.crop_height
                    limit_w = width - self.crop_width
                    offset_h = np.random.randint(low=0, high=limit_h + 1)
                    offset_w = np.random.randint(low=0, high=limit_w + 1)

                    img_left = crop_img(img_left, offset_h, offset_w, self.crop_height, self.crop_width)
                    img_right = crop_img(img_right, offset_h, offset_w, self.crop_height, self.crop_width)
                    disp_left = crop_img(disp_left, offset_h, offset_w, self.crop_height, self.crop_width)

                img_left = (img_left/np.max(img_left)) * 2
                img_right = (img_right/np.max(img_right)) * 2
                img_left = img_left - 1
                img_right = img_right - 1
                # concatenate left and right images
                comb_img = np.dstack((img_left, img_right))

                input_x.append(comb_img)
                target_y.append(np.expand_dims(disp_left, axis=2))
            return np.array(input_x), np.array(target_y)
