import os
import numpy as np


def get_images_and_labels(paths):
    images_all = []
    labels_all = []
    for i in paths.split(':'):
        images, labels = get_images_and_labels_one(i)
        images_all.extend(images)
        labels_all.extend(labels)
    return images_all, labels_all


def get_images_and_labels_one(path):
    folders = os.listdir(path)
    images = []
    labels = []
    for i in folders:
        static_path = path + '/' + i + '/'
        if os.path.isdir(static_path):
            imgs = os.listdir(static_path)
            for j in imgs:
                if j.endswith('.png') or j.endswith('.jpg'):
                    images.append(static_path + j)
                    labels.append(i)
    return images, labels


def get_paths_and_sims(data_dir, pair_path, file_ext):
    pairs = get_pairs(pair_path)
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(data_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = os.path.join(data_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.'+file_ext)
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(data_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = os.path.join(data_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+'.'+file_ext)
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list


def get_pairs(path):
    pairs = []
    with open(path, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)
