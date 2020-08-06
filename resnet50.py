import torchvision as tv
import numpy as np
import seaborn as sns
from numpy.linalg import svd
import pickle

#Network dependent operations
m = tv.models.resnet50(pretrained=True)
print(m)

layers = [m.layer1, m.layer2, m.layer3, m.layer4]

layers = [x[-1].conv2.weight.detach().numpy() for x in layers]


#Network independent operations
couts = [x.shape[0] for x in layers]
layers = [x.reshape(x.shape[0], -1) for x in layers]
print([x.shape for x in layers])

layers = [np.linalg.svd(x) for x in layers]

singular_vals = [x[1] for x in layers]

maxs = [x.max() for x in singular_vals]

xcoords = [np.arange(x)/x for x in couts]

singular_vals = [x/y for x,y in zip(singular_vals,maxs)]
print(singular_vals)

ax = None
for x,y in zip(xcoords, singular_vals):
    if ax is None:
        ax = sns.scatterplot(x=x, y=y)
    else:
        ax = sns.scatterplot(x=x, y=y, ax=ax)
ax.get_figure().savefig('resnet50.pdf')
pickle.dump( layers, open( "resnet50_stats.p", "wb" ) )
