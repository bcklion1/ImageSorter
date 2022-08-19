from __future__ import print_function
import binascii
import struct
from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import glob
import colorsys
from matplotlib import image

def get_hsv(hexrgb):
    hexrgb = hexrgb.lstrip("#")   # in case you have Web color specs
    r, g, b = (int(hexrgb[i:i+2], 16) / 255.0 for i in range(0,5,2))
    return colorsys.rgb_to_hsv(r, g, b)

NUM_CLUSTERS = 5
color_list = []
# print('reading image')
for filename in glob.iglob('C:/Users/briankunak/Downloads/pict/*', recursive=True):
    im = Image.open(filename).convert("RGB")
    # im.show()
    # im = im.resize((150, 150))      # optional, to reduce time
    ar = np.asarray(im)
    shape = ar.shape
    print(shape)
    # print(shape[2])
    print("\n\n")
    ar = ar.reshape(np.product(shape[:2]), 3).astype(float)

    # print('finding clusters')
    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    # print('cluster centres:\n', codes)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

    index_max = scipy.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
    color_list.append(colour)
    print('most frequent is %s (#%s)' % (peak, colour))

# color_list = ["000050", "005000", "500000"]  # GBR
color_list.sort(key=get_hsv)
print (color_list)