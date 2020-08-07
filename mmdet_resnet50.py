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


