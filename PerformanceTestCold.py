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
model_path = "D:/BTH/EXJOBB/ColabServers/Image2video/stage2_cINN/saves/Lightning/"


modelCold = 0
batchCold = 0
modelRunCold = 0

modelInitAvg = 0
batchAvg = 0
modelRunAvg = 0



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
def prepareImages(images, img_res):
    resize = k.Resize(size=(img_res, img_res))
    normalize = k.augmentation.Normalize(0.5, 0.5)
    imgs = [resize(normalize(k.image_to_tensor(cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)) / 255.0))
            for name in images]
    return torch.cat(imgs)
def loadModel(path, seq_len):
    print("loading model from " + path)

    totalAvg = 0
    spinAvg = 0
    initAvg = 0

    stamp = time.perf_counter()
    m = Model(path, seq_len)
    coldModel = time.perf_counter() - stamp
    print("cold model", coldModel)
    stamp = time.perf_counter()
    b = torch.zeros(1,3,128,128).cuda()
    coldBatch = time.perf_counter() - stamp
    print("cold batch", coldBatch)
    stamp = time.perf_counter()
    m(b).cpu()
    coldRun = time.perf_counter() - stamp
    print("cold run", coldRun)

    iterations = 10
    for i in range(iterations):
        start = time.perf_counter()
        model = Model(path, seq_len)

        spin = time.perf_counter()
        batch = torch.zeros(1,3,128,128).cuda()
        model(batch).cpu()

        end = time.perf_counter()
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
    spin = time.perf_counter()
    batch = torch.zeros(1, 3, 128, 128).cuda()
    model(batch).cpu()
    return model
@profile
def image2Vid(imageTensor, model, batchSize, nrOfCopies):
    print("generating videos")

    length = math.ceil(imageTensor.size(0) / batchSize)
    videos = []
    start = time.perf_counter()
    avg = 0
    with torch.no_grad():
        for x in range(nrOfCopies):
            for i in range(length):
                s = time.perf_counter()
                #batch = imageTensor[i:i+1].cuda()
                batch = imageTensor[i:i+1].cuda()
                bt = time.perf_counter() - s
                s = time.perf_counter()
                v = model(batch).cpu()
                #v = model(imageTensor[i:i+1]).cpu()
                vt = time.perf_counter() - s
                s = time.perf_counter()
                videos.append(v)
                at = time.perf_counter() - s
                print("i({:2}),batch({:2}),model({:2}),append({:2})".format(i,bt,vt,at))
                #print(i, bt, vt, at, 1/vt)
    end = time.perf_counter()
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

    # COLD INIT
    s = time.perf_counter()
    model = Model(model_path, seq_len)
    modelCold = time.perf_counter()-s
    iterations = 10
    for i in range(iterations):
        del model
        torch.cuda.empty_cache()
        s = time.perf_counter()
        model = Model(model_path, seq_len)
        mInit = time.perf_counter() - s
        modelInitAvg += mInit
        print(i, mInit, modelInitAvg)

    modelInitAvg /= iterations
    print(modelCold, modelInitAvg, modelCold-modelInitAvg)
    print("________________________________")

    images = prepareImages(image_paths, model.config.Data['img_size'])

    s = time.perf_counter()
    batch = images[0:1].cuda()
    batchCold = time.perf_counter()-s
    s = time.perf_counter()
    model(batch).cpu()
    modelRunCold = time.perf_counter() - s
    del model
    model = Model(model_path, seq_len)
    for i in range(iterations):
        del batch
        torch.cuda.empty_cache()
        s = time.perf_counter()
        batch = images[0:1].cuda()
        mBatch = time.perf_counter()-s

        s=time.perf_counter()
        model(batch).cpu()
        mRun = time.perf_counter() - s

        batchAvg += mBatch
        modelRunAvg += mRun

        print(i, mRun, modelInitAvg, mBatch, batchAvg)
    batchAvg /= iterations
    modelRunAvg /= iterations

    print(model.getPerfResAvg(), "|", model.PerfRes)
    print(model.getPerfCinnAvg(), "|", model.PerfCinn)
    print(model.getPerfDecoderAvg(), "|", model.PerfDecoder)


    f = open("D:/BTH/EXJOBB/ColabServers/Image2video/samples/coldResultsPerfCounter.csv", "a")
    f.write("{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},\n".format(
        modelCold, modelInitAvg, modelCold-modelInitAvg,
        batchCold, batchAvg, batchCold-batchAvg,
        modelRunCold, modelRunAvg, modelRunCold-modelRunAvg,
        model.getPerfFvdAvg(), model.getPerfResAvg(), model.getPerfCinnAvg(), model.getPerfDecoderAvg()
    ))
    f.close()

    #asd = image2Vid(images, model, batchSize, nrOfCopies)



