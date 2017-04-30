# coding=utf-8

import numpy as np
from sklearn.model_selection import KFold
from scipy import interpolate


# Before training,the mean must be substract
def train(trainingset, label, data_range):
    # the total num of image
    n_image = len(label)
    # the dim of features
    n_dim = trainingset.shape[1]
    # filter the complicate label,for count the total people num
    classes, labels = np.unique(label, return_inverse=True)
    # the total people num
    n_class = len(classes)
    # save each people items
    cur = {}
    withinCount = 0
    # record the count of each people
    numberBuff = np.zeros(n_image)
    maxNumberInOneClass = 0
    for i in range(n_class):
        # get the item of i
        cur[i] = trainingset[labels == i]
        # get the number of the same label persons
        n_same_label = cur[i].shape[0]

        if n_same_label > 1:
            withinCount += n_same_label
        if numberBuff[n_same_label] == 0:
            numberBuff[n_same_label] = 1
            maxNumberInOneClass = max(maxNumberInOneClass, n_same_label)
    print("prepare done, maxNumberInOneClass = " + str(maxNumberInOneClass))

    u = np.zeros([n_dim, n_class])
    ep = np.zeros([n_dim, withinCount])
    nowp = 0
    for i in range(n_class):
        # the mean of cur[i]
        u[:, i] = np.mean(cur[i], 0)
        b = u[:, i].reshape(n_dim, 1)
        n_same_label = cur[i].shape[0]
        if n_same_label > 1:
            ep[:, nowp:nowp + n_same_label] = cur[i].T - b
            nowp += n_same_label

    Su = np.cov(u.T, rowvar=0)
    Sw = np.cov(ep.T, rowvar=0)
    oldSw = Sw
    SuFG = {}
    SwG = {}
    convergence = 1
    min_convergence = 1
    for l in range(500):
        F = np.linalg.pinv(Sw)
        u = np.zeros([n_dim, n_class])
        ep = np.zeros([n_dim, n_image])
        nowp = 0
        for mi in range(maxNumberInOneClass + 1):
            if numberBuff[mi] == 1:
                # G = −(mS μ + S ε )−1*Su*Sw−1
                G = -np.dot(np.dot(np.linalg.pinv(mi * Su + Sw), Su), F)
                # Su*(F+mi*G) for u
                SuFG[mi] = np.dot(Su, (F + mi * G))
                # Sw*G for e
                SwG[mi] = np.dot(Sw, G)
        for i in range(n_class):
            ##print l, i
            nn_class = cur[i].shape[0]
            # formula 7 in suppl_760
            u[:, i] = np.sum(np.dot(SuFG[nn_class], cur[i].T), 1).reshape(n_dim,)
            # formula 8 in suppl_760
            ep[:, nowp:nowp + nn_class] = cur[i].T + np.sum(np.dot(SwG[nn_class], cur[i].T), 1).reshape(n_dim, 1)
            nowp = nowp + nn_class

        Su = np.cov(u.T, rowvar=0)
        Sw = np.cov(ep.T, rowvar=0)
        convergence = np.linalg.norm(Sw - oldSw) / np.linalg.norm(Sw)
        print("Iterations-" + str(l) + ": " + str(convergence))
        if convergence < 1e-6:
            print("Convergence: " + str(l) + " " + str(convergence))
            break
        oldSw = Sw

        if convergence < min_convergence:
            min_convergence = convergence
        F = np.linalg.pinv(Sw)
        G = -np.dot(np.dot(np.linalg.pinv(2 * Su + Sw), Su), F)
        A = np.linalg.pinv(Su + Sw) - (F + G)
        save_A_G(A, G, n_dim, data_range, str(l))

    F = np.linalg.pinv(Sw)
    G = -np.dot(np.dot(np.linalg.pinv(2 * Su + Sw), Su), F)
    A = np.linalg.pinv(Su + Sw) - (F + G)
    print('data_range: %s' % data_range)
    save_A_G(A, G, n_dim, data_range, 'full')
    return


# ratio of similar,the threshold we always choose in (-1,-2)
def verify(A, G, x1, x2):
    x1.shape = (-1, 1)
    x2.shape = (-1, 1)
    ratio = np.dot(np.dot(np.transpose(x1), A), x1) + np.dot(np.dot(np.transpose(x2), A), x2) - 2 * np.dot(
        np.dot(np.transpose(x1), G), x2)
    # print ratio
    return ratio[0][0]


def save_A_G(A, G, n_dim, data_range, iter):
    np.save('joint_bayesian/A_' + data_range + '_' + str(n_dim) + '_' + iter + '.npy', A.reshape(n_dim, n_dim))
    np.save('joint_bayesian/G_' + data_range + '_' + str(n_dim) + '_' + iter + '.npy', G.reshape(n_dim, n_dim))
    return


def load_A_G(data_range, iter, n_dim):
    A = np.load('joint_bayesian/A_' + data_range + '_' + str(n_dim) + '_' + iter + '.npy')
    G = np.load('joint_bayesian/G_' + data_range + '_' + str(n_dim) + '_' + iter + '.npy')
    return A, G


def validate(features, issame_list, data_range, nrof_folds, iter, n_dim):
    print('data_range: %s' % data_range)
    A, G = load_A_G(data_range, iter, n_dim)
    ratios = calculate_ratios(A, G, features)
    # define thresholds for choosing approximate threshold for evaluate
    # thresholds = np.arange(-150, -49, 5)
    # thresholds = np.arange(-80, -59, 1)
    thresholds = np.arange(-120, -10, 1)
    tpr, fpr, accuracy = calculate_roc(thresholds, ratios, np.asarray(issame_list), nrof_folds=nrof_folds)
    return tpr, fpr, accuracy


def calculate_ratios(A, G, features):
    nrof_pairs = len(features) / 2
    ratios = []
    for i in range(nrof_pairs):
        ratios.append(verify(A, G, features[2*i], features[2*i+1]))
    return np.array(ratios)


def calculate_roc(thresholds, ratios, issame_list, nrof_folds=10):
    nrof_pairs = len(issame_list)
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)

    acc_train = np.zeros((nrof_folds, nrof_thresholds))
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[fold_idx, threshold_idx] = calculate_accuracy(threshold, ratios[train_set], issame_list[train_set])
    best_threshold_index = np.argmax(np.mean(acc_train, 0))
    print('Best threshold of ratio: %1.2f' % thresholds[best_threshold_index])
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold, ratios[test_set], issame_list[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], ratios[test_set], issame_list[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)

    return tpr, fpr, accuracy


def calculate_accuracy(threshold, ratios, actual_issame):
    predict_issame = np.greater(ratios, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/ratios.size
    return tpr, fpr, acc
