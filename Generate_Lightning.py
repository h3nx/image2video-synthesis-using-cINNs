import argparse, os, glob, cv2, torch, math, imageio, lpips
from tqdm import tqdm
import kornia as k, numpy as np

from get_model import Model
from utils import auxiliaries as aux
import time

img_suffix = ['jpg', 'png', 'jpeg', "tif"]

# setup argparser
parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=str, default="0", help="Define GPU on which to run")
parser.add_argument('-dataset', type=str, default="", help='Specify dataset')
parser.add_argument('-texture', type=str, help='Specify texture when using DTDB')
parser.add_argument('-ckpt_path', type=str, required=False, help='If ckpt outside of repo')
parser.add_argument('-seq_length', type=int, default=16)
parser.add_argument('-bs', type=int, default=6, help='Batchsize')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

#path_ds = f'{args.dataset}/{args.texture}' if args.dataset == 'DTDB' else f'{args.dataset}'
#ckpt_path = f'./models/{path_ds}/stage2/' if not args.ckpt_path else args.ckpt_path
folder = "dark/"
img_path = "D:/BTH/EXJOBB/ColabServers/Data/Lightning/samples/"+folder
ckpt_path = "D:/BTH/EXJOBB/ColabServers/Image2video/stage2_cINN/saves/Lightning/"
## get all images (jpg, png, jpeg) in folder
img_list = []
print("fetching image names")
for suffix in img_suffix:
    img_list.extend(glob.glob(img_path + f'*.{suffix}'))

#print(ckpt_path)
#print(img_list)

## Load model from config
print("loading model")
model = Model(ckpt_path, args.seq_length)
#print(model)
## Load images

print("loading images")
img_res = model.config.Data['img_size']
resize = k.Resize(size=(img_res, img_res))
normalize = k.augmentation.Normalize(0.5, 0.5)

imgs = [resize(normalize(k.image_to_tensor(cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB))/255.0))
        for name in img_list]
imgs = torch.cat(imgs)
#print(imgs)

## Generate videos
bs = args.bs
length = math.ceil(imgs.size(0)/bs)
videos = []
count = 10
start = time.time()
with torch.no_grad():
    for x in range(count):
        for i in range(length):

            if i < (length -1):
                batch = imgs[i * bs:(i + 1) * bs].cuda()
            else:
                batch = imgs[i * bs:].cuda()
            videos.append(model(batch).cpu())

#for padI in range(30-args.seq_length):


videos = torch.cat(videos)
#print(len(videos))
#print(len(videos[0]))
#for vid in videos:
#    for padI in range(30-args.seq_length):
#        vid = torch.cat(vid, vid[args.seq_length-1])
end = time.time()
print("time taken to generate({},{}) bolts: {}".format(len(img_list),count, end-start))

## Save video as gif
save_path = f'./samples/Lightning/'+folder
os.makedirs(os.path.dirname(save_path), exist_ok=True)
#print(np.shape(videos))
#gif = aux.convert_seq2gif(videos)
gif = aux.convert_seq2gifs(videos,len(img_list),count)
#print(np.shape(gif))
#gif = gif.reshape((len(img_list), count))

fps = 30
finalPath = save_path+"results_"+str(fps)+".gif"
i = 1

while os.path.exists(finalPath):
    finalPath = save_path+"results_"+str(fps)+"_"+str(i)+".gif"
    i+=1

imageio.mimsave(finalPath, gif.astype(np.uint8), fps=fps)
print(f'Animations saved in {save_path}')






'''
    try different background
    try occlusion
    try darkness
    
    create questionnaire
    
    friday 13:30-14:00
    

'''

