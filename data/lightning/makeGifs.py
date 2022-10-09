import numpy as np
import os, glob
from tqdm import tqdm
from natsort import natsorted
import glob
import imageio
from PIL import Image


root = "D:/BTH/EXJOBB/ColabServers/Data/Lightning/"
output = "D:/BTH/EXJOBB/ColabServers/Data/Lightning/samples/Comparison/Ground truth/"
modes = ['QuestionnaireComparison']
#modes = ['train', 'test']

prePad = 14
postPad = 15


for ia, mode in enumerate(modes):

    path = root + mode +  "/"
    print(path)
    os.makedirs(output, exist_ok=True)

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
            clip = natsorted(glob.glob(clipFolder+"/*.tif"))
            #print(clipFolder)
            images = []
            img0 = Image.fromarray(imageio.imread(clip[0])).resize((128, 128))
            for i in range(0,prePad):
                images.append(img0)
            for imageIndex, image, in enumerate(clip):
                images.append(Image.fromarray(imageio.imread(image)).resize((128, 128)))
            for i in range(0,16-len(clip)):
                images.append(img0)
            for i in range(0,postPad):
                images.append(img0)
            #print(len(images))
            part =  os.path.basename(folderPath)
            name = os.path.basename(clipFolder)
            outname = output+mode[0:2]+part[1:]+name+".gif"
            print(s, outname)
            imageio.mimsave(outname, images, fps=30)
