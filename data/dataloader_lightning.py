import cv2, torch, torch.nn as nn
import numpy as np
import kornia as k, os, glob
import torchvision

from data.augmentation import Augmentation

import sys

class Dataset(torch.utils.data.Dataset):

    def __init__(self, opt, mode):
        print("_______ lightning loader __________")
        self.data_path = opt.Data['data_path']
        self.prefix = 'train' if mode != 'eval' else 'test'

        self.seq_length = opt.Data['sequence_length']
        self.seq_Truelength = opt.Data['sequence_Truelength']

        self.do_aug = opt.Data['aug']
        self.videos = []; self.num_frames = []
        print(f'path: {self.data_path}\n '
              f'seq l: {self.seq_length}'
              f'_________________________________')
        sectionFolders = glob.glob(self.data_path + self.prefix + "/*")
        for ib, folderPath in enumerate(sectionFolders):
            #print("  ", folderPath)
            finalFolders = glob.glob(folderPath + "/*")
            for ic, clipFolder in enumerate(finalFolders):
                #print("    ", clipFolder, "____",len(glob.glob(clipFolder+"/*.tif")) )
                for _ in range(opt.Data[f'iter_{mode}']):
                    self.videos.append(clipFolder)
                    self.num_frames.append(len(glob.glob(clipFolder+"/*.tif")))
        #print(" ")
        #print(" ")
        #print(" ")
        #print("prefix << ", self.prefix)
        #print("root << ", self.data_path)
        #print("sections << ", sectionFolders)

        #for i in range(len(self.videos)):
            #p = self.load_img_path(self.videos[i], 0)
            #img = cv2.imread(p)
            #print(self.load_img_path(self.videos[i], 0), img)


        #print(self.videos)
        #print(self.num_frames)

        #return

        #for vid in video_list:
        #    path = self.data_path + self.prefix + '/' + vid + '/frame*.tif'
        #    length = len(glob.glob(path))
        #    if length == 0:
        #        print("import", path, " _ " ,0)
        #    for _ in range(opt.Data[f'iter_{mode}']):
        #        self.videos.append(self.prefix + '/' + vid)
        #        self.num_frames.append(length)

        self.length = len(self.videos)

        if mode == 'train' and self.do_aug:
            self.aug = Augmentation(opt.Data['img_size'], opt.Data.Augmentation, gs = opt.Data.gs)
        elif opt.Data.gs:
            self.aug = torch.nn.Sequential(
                        k.Resize(size=(opt.Data['img_size'], opt.Data['img_size'])),
                        k.augmentation.Normalize(0.5, 0.5),
                        torchvision.transforms.Grayscale(3))
        else:
            self.aug = torch.nn.Sequential(
                k.Resize(size=(opt.Data['img_size'], opt.Data['img_size'])),
                k.augmentation.Normalize(0.5, 0.5))
        #print("Dataloader init done")
        print("len", self.length)

    def __len__(self):
        return self.length
    def load_img_path(self, video, frame):
        return video + "/Frame" + str(int(frame))+".tif"

    def load_img(self, video, frame):
        img = cv2.imread(self.load_img_path(video,frame))
        return k.image_to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))/255.0

    def __getitem__(self, idx):

        video  = self.videos[idx]
        frames = np.arange(0, self.num_frames[idx])

        frames = np.append(frames, [self.num_frames[idx]-1] * (self.seq_length - self.num_frames[idx]))
        #print(video, self.num_frames[idx])
        #print("get item({}) ".format(idx), "video({}) ".format(video), self.num_frames[idx])
        ## Sample random starting point in the sequence
        #print("frames")
        #print(frames, flush=True)
        #print("____________", flush=True)
        if self.seq_length == 1:
            start_rand = np.random.randint(0, len(frames) - self.seq_length + 1)
        else:
            start_rand = 0

        r = []

        for i in range(self.seq_Truelength):
            r.append(i)
        for i in range(self.seq_length-self.seq_Truelength):
            r.append(0)

        #print(r)
        #quit(10)

        seq = torch.stack([self.load_img(video, frames[i]) for i in r], dim=0)
        return {'seq': self.aug(seq)}

