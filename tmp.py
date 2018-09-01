import mxnet as mx
import cv2
import numpy as np
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
ctx = mx.cpu(0)


def load_test(img, short, max_size=1024, mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)):
    img = mx.nd.array(img)
    img = mx.image.resize_short(img, short)
    if isinstance(max_size, int) and max(img.shape) > max_size:
        img = timage.resize_long(img, max_size)
    orig_img = img.asnumpy().astype('uint8')
    img = mx.nd.image.to_tensor(img)
    img = mx.nd.image.normalize(img, mean=mean, std=std)
    tensors = img.expand_dims(0)
    return tensors, orig_img

url = 'https://github.com/zhanghang1989/image-data/blob/master/encoding/segmentation/ade20k/ADE_val_00001142.jpg?raw=true'
filename = 'ade20k_example.jpg'
gluoncv.utils.download(url, filename)

img = image.imread(filename)
from matplotlib import pyplot as plt
#plt.imshow(img.asnumpy())
#plt.show()    

transform_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])

img = mx.image.resize_short(img, 200)
img = transform_fn(img)
print img.shape
img = img.expand_dims(0).as_in_context(ctx)
model = gluoncv.model_zoo.get_model('psp_resnet50_ade', pretrained=True)
img = cv2.imread(filename)
img, _ = load_test(img, 150)

print('ft1')
output = model.demo(img)
predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
print('ft2')

from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg
print predict.shape
mask = get_color_pallete(predict, 'ade20k')
print np.asarray(mask.convert('RGB')).shape
import cv2
#cv2.imshow('x', np.asarray(mask.convert('RGB')))
#cv2.waitKey(0)
#print(type(mask))

mask.save('output.png')
#exit(0)

mmask = mpimg.imread('output.png')
print mmask.shape
plt.imshow(mmask)
plt.show()
