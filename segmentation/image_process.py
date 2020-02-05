from __future__ import division, print_function
from cv2 import cv2
import numpy as np

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# targ = ((24, 24), (180, 180))

# for i in range(1, 5509):
#     img = cv2.imread('./realimage/'+str(i)+'_ypred.png', 0)
#     # print(img.shape)
#     img = np.pad(img, targ, mode='constant', constant_values=0)
#     print(img)
#     #cv2.imshow('sadf',img)
#     img = img.flatten()
#     print(img.shape)
#     print(rlencode(img))


import numpy as np
from PIL import Image
import os
from itertools import chain
import csv

def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def run_length(label):
    x = label.transpose().flatten()
    y = np.where(x>0.5)[0]
    if len(y)<10:# consider as empty
        return []
    z = np.where(np.diff(y)>1)[0]
    start = np.insert(y[z+1],0,y[0])
    end = np.append(y[z],y[-1])
    leng = end - start
    res = [[s+1,l+1] for s,l in zip(list(start),list(leng))]
    res = list(chain.from_iterable(res))
    return res


if __name__ == '__main__':
    input_path = './realimage/'
    # masks = [f for f in os.listdir(input_path) if f.endswith('ypred.tif')]
    # masks = sorted(masks, key=lambda s: int(s.split('_')[0])*1000 + int(s.split('_')[1]))
    
    # encodings = []
    # N = 100     # process first N masks
    # for i,m in enumerate(masks[:N]):
    #     if i % 10 == 0: print('{}/{}'.format(i, len(masks)))
    #     img = Image.open(os.path.join(input_path, m))
    #     x = np.array(img.getdata(), dtype=np.uint8).reshape(img.size[::-1])
    #     x = x // 255
    #     encodings.append(rle_encoding(x))
        
    # #check output
    # conv = lambda l: ' '.join(map(str, l)) # list -> string
    # subject, img = 1, 1
    # print('\n{},{},{}'.format(subject, img, conv(encodings[0])))

    #  # train_masks.csv:
    
    targ = ((10, 10), (90, 90))

    for i in range(1, 5509):
        img = cv2.imread('./realimage/'+str(i)+'_ypred.png', 0)
        # print(img.shape)
        img = np.pad(img, targ, mode='constant', constant_values=0)
        # print(img)
        # cv2.imwrite('test.png',img)
        # #img = img.flatten()
        # print(img.shape)
        # print(rle_encoding(img))
        mask_rle = run_length(img)
        print(str(mask_rle).replace(',', '').replace('[', '').replace(']', ''))
        mask_rle = str(mask_rle).replace(',', '').replace('[', '').replace(']', '')
        with open('submission.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, mask_rle])
