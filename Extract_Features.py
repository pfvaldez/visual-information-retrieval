'''
Here’s a brief explanation of what the script does:
1. It imports necessary modules and classes.
2. It creates instances of the Color and Edge classes, which are used to extract color and edge features from images, respectively.
3. It defines a function feature_extraction that takes a descriptor and an image path, reads the image, extracts features using the descriptor, and returns the image ID and the features.
4. It defines a function index_fast that takes an index file and a descriptor, gets a list of all files in the dataset, and uses multiprocessing to extract features from all images in the dataset. It then writes the image ID and the features to the index file.
5. In the main part of the script, it checks if the dataset exists. If the color index file does not exist, it opens the file, extracts color features from all images in the dataset, and writes them to the file. It does the same for the edge index file.
'''

import multiprocessing
from functools import partial
from glob import glob
from itertools import product
from os.path import basename, exists, join

import cv2

from API_Backend.Helpers.color import Color
from API_Backend.Helpers.edge import Edge
from API_Backend.Helpers.PathConfig import FilePaths

cd = Color((8, 12, 3))
# cd = Color((16, 24, 6))
ed = Edge()


def feature_extration(desc, imagePath):
    imageID = basename(imagePath)
    image = cv2.imread(imagePath)

    # describe the image
    features = desc.describe(image)

    # write the features to file
    features = [str(f) for f in features]
    return imageID, features


def index_fast(idx_file, desc):
    file_list = glob(join(FilePaths.dataset, '*'))
    count = 0

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()-2) as pool:
        func = partial(feature_extration, desc)
        results = pool.starmap(func, product(file_list))
    for imageID, features in results:
        features = [str(f) for f in features]
        idx_file.write("%s,%s\n" % (imageID, ",".join(features)))
        count += 1
        print(count, '/', len(file_list))



if __name__ == '__main__':
    assert exists(FilePaths.dataset)
    if not exists(FilePaths.color_index):
        print('extracting color features\n')
        idx_file = open(FilePaths.color_index, 'w')
        index_fast(idx_file, cd)
        idx_file.close()
    if not exists(FilePaths.edge_index):
        print('extracting edge features\n')
        idx_file = open(FilePaths.edge_index, 'a')
        index_fast(idx_file, ed)
        idx_file.close()
