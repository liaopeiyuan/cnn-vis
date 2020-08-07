import torchvision as tv
import numpy as np
import seaborn as sns
from numpy.linalg import svd
import pickle
import pandas as pd
import sys
from matplotlib import pyplot as plt
from tqdm import tqdm

networks = ['resnet'+str(x) for x in [18,34,50,101,152]]

fig, ax = plt.subplots(3, 2, figsize=(20, 20))

i = 0
for NETWORK in tqdm(networks):
  #Network dependent operations
  m = getattr(tv.models, NETWORK)(pretrained=False)#tv.models.resnet50(pretrained=True)

  layers = [m.layer1, m.layer2, m.layer3, m.layer4]

  layers = [x[-1].conv2.weight.detach().numpy() for x in layers]


  #Network independent operations
  couts = [x.shape[0] for x in layers]
  layers = [x.reshape(x.shape[0], -1) for x in layers]

  layers = [np.linalg.svd(x) for x in layers]

  singular_vals = [x[1] for x in layers]

  maxs = [x.max() for x in singular_vals]

  xcoords = [np.arange(x)/x for x in couts]

  singular_vals = [x/y for x,y in zip(singular_vals,maxs)]

  l = 1
  rows = []
  for x,y in zip(xcoords, singular_vals):
    for (a,b) in zip(x,y):
        rows.append({'i/cout':a,'\lam/\lam_max':b,'stage':str(l)})
    l+=1
  df = pd.DataFrame(rows)
  print(i//2, i%2)
  sns.lineplot(x='i/cout',y='\lam/\lam_max',legend='full',hue='stage', data=df, markers=True, ax=ax[i//2][i%2])
  ax[i//2][i%2].set_title(NETWORK)
  i += 1

import torch

sd = torch.load('faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth')

print([k for k,v in sd.items()])

keys = []

for k in (sd['state_dict'].keys()):
    if 'backbone' in k and 'layer' in k and 'conv2' in k: keys.append(k)
print(keys)

keys = ['backbone.layer1.2.conv2.weight',  'backbone.layer2.3.conv2.weight', 'backbone.layer3.5.conv2.weight', 'backbone.layer4.2.conv2.weight']

layers = [sd['state_dict'][x].numpy() for x in keys]
print(layers)

#Network independent operations
couts = [x.shape[0] for x in layers]
layers = [x.reshape(x.shape[0], -1) for x in layers]

layers = [np.linalg.svd(x) for x in layers]

singular_vals = [x[1] for x in layers]

maxs = [x.max() for x in singular_vals]

xcoords = [np.arange(x)/x for x in couts]

singular_vals = [x/y for x,y in zip(singular_vals,maxs)]

l = 1
rows = []
for x,y in zip(xcoords, singular_vals):
    for (a,b) in zip(x,y):
        rows.append({'i/cout':a,'\lam/\lam_max':b,'stage':str(l)})
    l+=1
df = pd.DataFrame(rows)
print(i//2, i%2)
sns.lineplot(x='i/cout',y='\lam/\lam_max',legend='full',hue='stage', data=df, markers=True, ax=ax[i//2][i%2])
ax[i//2][i%2].set_title('resnet50_frcnn')


plt.savefig('resnets_random.pdf')


