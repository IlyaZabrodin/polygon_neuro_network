from PIL import Image
import numpy as np
import os
import nirs_test

directory = 'data/'
arr_lbs = []
arr_ims = []

for im_file in os.listdir(directory):
    f = os.path.join(directory, im_file)
    img = np.array(Image.open(f), dtype='uint8')
    mark = nirs_test.process_image(f)

    arr_ims.append(img)
    X_array = np.asarray(arr_ims)

    if not mark is None:
        arr_lbs.append(mark)
        Y_array = np.asarray(arr_lbs)
        # convex=1 convac=0

    np.savez('network_dataset.npz', x=X_array, y=Y_array)

'''
with np.load("network_dataset.npz", allow_pickle=True) as fi:
    lst = fi.files
    for item in lst:
        print(item, len(fi[item]))
        print(fi[item])
'''