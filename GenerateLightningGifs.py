import argparse, os, glob, cv2, torch, math, imageio, lpips
from tqdm import tqdm
import kornia as k, numpy as np
from utils.auxiliaries import denorm

from get_model import Model
from utils import auxiliaries as aux
import time

fps = 30
seq_len = 16
batchSize = 6
nrOfCopies = 30

sample_set = "mountain"
img_path = "D:/BTH/EXJOBB/ColabServers/Data/Lightning/samples/"+sample_set
save_root = "./samples/Lightning/"+sample_set

#img_path = "D:/BTH/EXJOBB/Data/Comparison/Initial"
#save_root = "D:/BTH/EXJOBB/Data/Comparison/Generated/"

model_path = "D:/BTH/EXJOBB/ColabServers/Image2video/stage2_cINN/saves/Lightning/"

def getSaveFolder(path):
    finalPath = path
    i = 1
    while os.path.exists(finalPath):
        finalPath = path+str(i)
        i+=1
    return finalPath

def getImages(folder):
    print("fetching images from " + folder)
    img_suffix = ['jpg', 'png', 'jpeg', "tif"]
    img_list = []
    for suffix in img_suffix:
        img_list.extend(glob.glob(img_path + f'/*.{suffix}'))
    return img_list
def prepareImages(images, model):
    img_res = model.config.Data['img_size']
    resize = k.Resize(size=(img_res, img_res))
    normalize = k.augmentation.Normalize(0.5, 0.5)
    imgs = [resize(normalize(k.image_to_tensor(cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)) / 255.0))
            for name in images]
    return torch.cat(imgs)
def loadModel(path, seq_len):
    print("loading model from " + path)
    start = time.time()
    model = Model(path, seq_len)
    end = time.time()
    print("  ","model loaded after {}".format(end-start))
    return model

def image2Vid(imageTensor, model, batchSize, nrOfCopies):
    print("generating videos")
    length = math.ceil(imageTensor.size(0) / batchSize)
    videos = []
    start = time.time()
    avg = 0
    with torch.no_grad():
        for x in range(nrOfCopies):
            for i in range(length):
                if i < (length - 1):
                    batch = imageTensor[i * batchSize:(i + 1) * batchSize].cuda()
                else:
                    batch = imageTensor[i * batchSize:].cuda()
                videos.append(model(batch).cpu())
    end = time.time()
    print("time taken to generate({},{}) bolts: {}, avg: {}".format(imageTensor.size(0),nrOfCopies, end-start, avg))

    videos = torch.cat(videos)
    return videos

def convert_seq2gifsSingle(sequence, prePad = 14, postPad = 15):
    gifs = denorm(sequence).permute(0, 1, 3, 4, 2).detach().cpu().numpy()
    for ix in range(16, 16+postPad):
        gifs = np.concatenate((gifs, np.expand_dims(gifs[:,15,:,:,:], axis=1)), axis=1)
    for ix in range(0, prePad):
        gifs = np.concatenate((np.expand_dims(gifs[:,15+ix,:,:,:], axis=1), gifs), axis=1)
    gifs = 255 * gifs / np.max(gifs)
    return gifs

if __name__ == '__main__':

    image_paths =  getImages(img_path)
    #print(os.path.basename(your_path))

    model = loadModel(model_path, seq_len)
    images = prepareImages(image_paths, model)

    asd = image2Vid(images, model, batchSize, nrOfCopies)

    res = convert_seq2gifsSingle(asd)

    rootPath = getSaveFolder(save_root)+"/"
    os.makedirs(os.path.dirname(rootPath), exist_ok=True)

    gifIndex = 0
    print("saving gifs to "+ rootPath)
    for gif in res:
        imageIndex = math.floor(gifIndex/nrOfCopies)
        countIndex = gifIndex-(imageIndex*nrOfCopies)
        finalPath =rootPath+os.path.basename(image_paths[gifIndex%len(image_paths)]).split(".")[0]+"_"+str(fps)+"fps_"+str(gifIndex)+".gif"
        imageio.mimsave(finalPath, gif.astype(np.uint8), fps=fps)

        print(finalPath, gif.shape)
        if gifIndex != 0 and gifIndex % len(image_paths) == 0:
            print("{}/{}".format(gifIndex, len(image_paths)*nrOfCopies))
        gifIndex+=1



