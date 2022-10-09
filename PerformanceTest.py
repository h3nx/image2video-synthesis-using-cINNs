import argparse, os, glob, cv2, torch, math, imageio, lpips
from tqdm import tqdm
import kornia as k, numpy as np
from utils.auxiliaries import denorm

from get_model import Model
from utils import auxiliaries as aux
import time
from Profile import profile, get_memory


fps = 30
seq_len = 16
batchSize = 1
nrOfCopies = 1

sample_set = "dataset2"
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

    initial = 0
    totalAvg = 0
    spinAvg = 0
    initAvg = 0
    print(get_memory())

    stamp = time.time()
    m = Model(path, seq_len)
    coldModel = time.time() - stamp
    print("cold model", coldModel)
    stamp = time.time()
    b = torch.zeros(1,3,128,128).cuda()
    coldBatch = time.time() - stamp
    print("cold batch", coldBatch)
    stamp = time.time()
    m(b).cpu()
    coldRun = time.time() - stamp
    print("cold run", coldRun)


    print(get_memory())
    iterations = 10
    for i in range(iterations):
        start = time.time()
        model = Model(path, seq_len)

        spin = time.time()
        batch = torch.zeros(1,3,128,128).cuda()
        model(batch).cpu()

        end = time.time()
        print("  ",i,"model loaded after {:.2f}s model({:.2f}s) run({:.2f}s)".format(end-start, spin-start, end-spin))

        totalAvg += end-start
        spinAvg += end-spin
        initAvg += spin-start

        del batch
        del model
        torch.cuda.empty_cache()

    totalAvg /= iterations
    spinAvg /= iterations
    initAvg /= iterations


    print("  ", "Averages over i({}) total({:.2f}s) init({:.2f}s) spin({:.2f}s)".format(iterations, totalAvg, initAvg, spinAvg))
    print("  ", "Cuda init time modelLoad({:.2f}s), run({:.2f}s)".format(coldModel-initAvg, coldRun-spinAvg))

    model = Model(path, seq_len)
    spin = time.time()
    batch = torch.zeros(1, 3, 128, 128).cuda()
    model(batch).cpu()
    return model
@profile
def image2Vid(imageTensor, model, batchSize, nrOfCopies):
    print("generating videos")

    length = math.ceil(imageTensor.size(0) / batchSize)
    videos = []
    start = time.time()
    avg = 0
    with torch.no_grad():
        for x in range(nrOfCopies):
            for i in range(length):
                s = time.time()
                #batch = imageTensor[i:i+1].cuda()
                batch = imageTensor[i:i+1].cuda()
                bt = time.time() - s
                s = time.time()
                v = model(batch).cpu()
                #v = model(imageTensor[i:i+1]).cpu()
                vt = time.time() - s
                s = time.time()
                videos.append(v)
                at = time.time() - s
                print("i({:2}),batch({:2}),model({:2}),append({:2})".format(i,bt,vt,at))
                #print(i, bt, vt, at, 1/vt)
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



