import argparse, os, glob, cv2, torch, math, imageio, lpips

import PIL.TiffImagePlugin
import numpy as np
from PIL import Image

from natsort import natsorted


folder = "D:/BTH/EXJOBB/ColabServers/Data/Lightning"
subfolders = ["train", "test"]


frame = "Frame1.tif"
outputFolder = "D:/BTH/EXJOBB/Data/GTCollection/"



if __name__ == '__main__':
    #im = Image.open(folder+filename+".gif")
    #print("Number of frames: "+str(im.n_frames))
    #print(folder.split("/"))
    os.makedirs(outputFolder, exist_ok=True)
    stack = [];
    imageRow = []
    finalImage = []
    i = 0
    u = 0
    #stackI = 0



    for subfolderName in subfolders:
        print(folder+"/"+subfolderName+"/")
        for boltName in natsorted(glob.glob(folder+"/"+subfolderName+"/*/*")):
            print("",boltName);
            stackI = 0
            for frame in natsorted(glob.glob(boltName+"/*")):
                frameImage = Image.open(frame)
                if stackI == 0:
                   stack = np.asarray(frameImage)
                else:
                   stack = np.concatenate((stack, np.asarray(frameImage)), axis=2)
                stackI+=1
                #print(frame)
            while len(frame) < 16:
                stack = np.concatenate((stack, np.asarray(frameImage)), axis=2)
            #im = Image.open(boltName+"/"+frame)
            if i == 0:
                #imageRow = np.asarray(im)
                imageRow = stack
            else:
                imageRow = np.concatenate((imageRow, stack), axis=1)
            i+=1
            if i == 10 and u == 0:
                finalImage = imageRow
                i = 0
                u+=1
            elif i == 10:
                finalImage = np.concatenate((finalImage, imageRow), axis = 0)
                i = 0
    imExp = Image.fromarray(np.uint8(finalImage))
    imExp.save(outputFolder + "testGif" + ".gif")
    #i = 0
    #for imgName in natsorted(glob.glob(folder+"/"+name+"/*.tif")):
    #    print(imgName)
    #    im = Image.open(imgName)
    #    if i == 0:
    #        image = np.asarray(im)
    #    else:
    #        image = np.concatenate((image, np.asarray(im)), axis=1)
    #    i+=1
    #imExp = Image.fromarray(np.uint8(image))
    #imExp.save(outputFolder + name + ".png")

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


