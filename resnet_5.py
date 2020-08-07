import torchvision as tv
import numpy as np
import seaborn as sns
from numpy.linalg import svd
import pickle
import pandas as pd
import sys
from matplotlib import pyplot as plt

networks = ['resnet'+str(x) for x in [18,34,50,101,152]]

fig, ax = plt.subplots(2, 3, figsize=(30, 30))

i = 0
for NETWORK in networks:
  #Network dependent operations
  m = getattr(tv.models, NETWORK)(pretrained=True)#tv.models.resnet50(pretrained=True)
  print(m)

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
  sns.lineplot(x='i/cout',y='\lam/\lam_max',legend='full',hue='stage', data=df, markers=True, ax=ax[i%2][i//3])
  ax[i%2][i//3].set_title(NETWORK)
  i += 1

plt.savefig('resnets.pdf')



