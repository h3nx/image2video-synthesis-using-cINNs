import numpy as np
import os, glob
from tqdm import tqdm
from natsort import natsorted
import glob

root = "D:/BTH/EXJOBB/ColabServers/Data/Lightning/"
#modes = ['train', 'test']
modes = ['QuestionnaireComparison']



for ia, mode in enumerate(modes):

    path = root + mode +  "/"
    print(path)

    sum = glob.glob(path+"*")
    #print(sum)
    s = 0
    total = 0
    totalClips = 0
    for ib, folderPath in enumerate(sum):
        print(folderPath)
        finalFolders = glob.glob(folderPath+"/*")
        #print(finalFolders)
        for ic, clipFolder in enumerate(natsorted(finalFolders)):
            s += 1
            clip = glob.glob(clipFolder+"/*.tif")
            for imageIndex, image, in enumerate(natsorted(clip)):
                #print(imageIndex, image)
                newName = clipFolder+"/Frame{}.tif".format(imageIndex)
                if image != newName:
                    os.rename(image, newName)
            newFolderName = folderPath+"/Bolt{}".format(ic)
            if clipFolder != newFolderName:
                os.rename(clipFolder, newFolderName)
            totalClips += 1
    print("Total {} Clips: {}".format(mode, totalClips))