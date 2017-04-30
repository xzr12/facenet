import os
import sys
import argparse

import tensorflow as tf
import numpy as np
import math
from sklearn import metrics
from sklearn.decomposition import PCA

import facenet
import dataset
import joint_bayesian

# import pydevd
#
# pydevd.settrace('183.172.50.223', port=10000, stdoutToServer=True, stderrToServer=True)


def generate_features(args, newPath):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Get input images and labels
            print('Input data directory: %s' % args.input_data_dir)
            if newPath == '':
                paths, labels = dataset.get_images_and_labels(args.input_data_dir)
            else:
                paths, labels = dataset.get_paths_and_sims(args.input_data_dir, args.pairs, args.file_ext)

            # Load the model
            print('Model directory: %s' % args.model_dir)
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))

            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            facenet.load_model(args.model_dir, meta_file, ckpt_file)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image_size = images_placeholder.get_shape()[1]
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Runnning forward pass on images')
            batch_size = args.batch_size
            nrof_images = len(paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches):
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            # paths, image path list
            # labels, image label list
            # emb_array, image featrues
    return emb_array, labels


def main(args):
    if args.phase == 'train':
        # Get features and label list of input data
        # len(features) = len(labels)
        print('Phase: %s' % args.phase)
        print('Generate features from input data')
        features, labels = generate_features(args, '')
        print('PCA for features of input data')
        # choose automatically for mle
        # pca = PCA(n_components='mle', svd_solver='full')
        if args.pca == features.shape[1]:
            print features.shape[1]
            embeddings = features
            return
        else:
            pca = PCA(n_components=args.pca)
            embeddings = pca.fit_transform(features)
        print('Train with joint bayesian')
        joint_bayesian.train(embeddings, labels, args.data_range)
    else:
        # Get features and issame list of input data
        # 2 * len(features) = len(labels)
        print('Phase: %s' % args.phase)
        print('Generate features from input data')
        features, labels = generate_features(args, args.pairs)
        print('PCA for features of input data')
        pca = PCA(n_components=args.pca)
        embeddings = pca.fit_transform(features)
        print('Evaluate with joint bayesian')
        tpr, fpr, accuracy = joint_bayesian.validate(embeddings, labels, args.data_range, args.folds, args.iter, args.pca)
        print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))

        auc = metrics.auc(fpr, tpr)
        print('Area Under Curve (AUC): %1.3f' % auc)

    return


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_data_dir', type=str,
        help='Path of directory of input data.')
    parser.add_argument('phase', type=str, choices=['train', 'test'],
        help='Phase of joint bayesian, train of test.')
    parser.add_argument('model_dir', type=str,
        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters')
    parser.add_argument('--data_range', type=str, default='all',
        help='Data range of train data', choices=['all', 'casia', 'facescrub'])
    parser.add_argument('--batch_size', type=int, default=100,
        help='Number of images to process in a batch in test dataset.')
    parser.add_argument('--pairs', type=str,
        help='The file containing the pairs to use for validation.',
        default='/home/xuziru/facenet/data/pairs.txt')
    parser.add_argument('--file_ext', type=str, default='png',
        help='The file extension for the LFW dataset.', choices=['png', 'jpg'])
    parser.add_argument('--iter', type=str, default='full',
        help='A, G iter num')
    parser.add_argument('--pca', type=int, help='Dimension for PCA')
    parser.add_argument('--folds', type=int, default=10,
        help='K-fold cross validation for test data.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
