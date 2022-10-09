import argparse, os, glob, cv2, torch, math, imageio, lpips

import PIL.TiffImagePlugin
import numpy as np
from PIL import Image


#folder = "D:/BTH/EXJOBB/Data/Sample Gifs/Dataset0/"
#folder = "D:/BTH/EXJOBB/Data/Comparison/Generated/nice/"
#folder = "D:/BTH/EXJOBB/Data/questionnaire/"
folder = "D:/BTH/EXJOBB/Data/Training results/"
#filenames = ["1","2","3","4","5","6","7","8","9","10"]
filenames = ["200"]

#outputFolder = "D:/BTH/EXJOBB/Data/Comparison/SplitExamples/"+folder.split("/")[-2]+"/"
outputFolder = "D:/BTH/EXJOBB/Data/Training results/images/"


if __name__ == '__main__':
    #total = []
    for filename in filenames:
        im = Image.open(folder+filename+".gif")
        print("Number of frames: "+str(im.n_frames))
        print(folder.split("/"))

        print(outputFolder)
        os.makedirs(outputFolder, exist_ok=True)

        start = 1
        end = start+9
        im.seek(start)
        image = np.asarray(im)
        print(image.shape)
        for i in range(start+1, end):
            im.seek(i)
            arr = np.asarray(im)
            image = np.concatenate((image, arr), axis=1)
            #print(image.shape)
        imExp = Image.fromarray(np.uint8(image))
        imExp.save(outputFolder + filename + ".png")
        #total = np.concatenate((total, image), axis=2)




