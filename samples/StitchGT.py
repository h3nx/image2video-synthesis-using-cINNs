import argparse, os, glob, cv2, torch, math, imageio, lpips

import PIL.TiffImagePlugin
import numpy as np
from PIL import Image

from natsort import natsorted


folder = "D:/BTH/EXJOBB/ColabServers/Data/Lightning/test/p1/"

name = "Bolt0"
outputFolder = "D:/BTH/EXJOBB/Data/GTStitched/"



if __name__ == '__main__':
    #im = Image.open(folder+filename+".gif")
    #print("Number of frames: "+str(im.n_frames))
    #print(folder.split("/"))
    os.makedirs(outputFolder, exist_ok=True)
    image = []
    i = 0
    for imgName in natsorted(glob.glob(folder+"/"+name+"/*.tif")):
        print(imgName)
        im = Image.open(imgName)
        if i == 0:
            image = np.asarray(im)
        else:
            image = np.concatenate((image, np.asarray(im)), axis=1)
        i+=1
    imExp = Image.fromarray(np.uint8(image))
    imExp.save(outputFolder + name + ".png")

    #print(outputFolder)
    #os.makedirs(outputFolder, exist_ok=True)
#
    #image = np.asarray(im)
    #print(image.shape)
    #for i in range(start+1, end):
    #    im.seek(i)
    #    arr = np.asarray(im)
    #    image = np.concatenate((image, arr), axis=1)
    #    print(image.shape)
    #imExp = Image.fromarray(np.uint8(image))
    #imExp.save(outputFolder + filename + ".png")
#
#
    #img0 = Image.fromarray(imageio.imread(clip[0])).resize((128, 128))
    #for imageIndex, image, in enumerate(clip):
    #    images.append(Image.fromarray(imageio.imread(image)).resize((128, 128)))


