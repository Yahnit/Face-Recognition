from PIL import Image
import numpy as np
import math
import sys

def readFile(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content

def processTrainData():
    train_data = readFile(train_path)
    image_dictionary = {}
    image_class_label_dictionary = {}
    class_num = 0
    num_images = len(train_data)
    img_x, img_y = basewidth, basewidth
    X_train = np.zeros((num_images, img_x*img_y),dtype=np.float64)
    X_temp = np.zeros((num_images, img_x*img_y),dtype=np.float64)

    for line in range(num_images):
        img_class = train_data[line].split(' ')[1]
        filename = train_data[line].split(' ')[0]

        if img_class not in image_dictionary:
            image_dictionary[img_class] = {}
            image_dictionary[img_class]['images'] = []
            image_dictionary[img_class]['class_num'] = class_num
            image_class_label_dictionary[class_num] = img_class
            class_num += 1

        image = Image.open(filename).convert('L')
        image = image.resize((basewidth,basewidth), Image.BICUBIC)
        img = np.asarray(image).flatten()
        image_dictionary[img_class]['images'].append(img)
        X_train[line,:] = img

    # 2D array, entry is 1 if the image with index 'i' belongs to class 'j'
    num_classes = len(image_dictionary)
    image_class_dictionary = np.zeros((num_images, num_classes),dtype=np.float64)
    for line in range(num_images):
        img_class = train_data[line].split(' ')[1]
        class_num = image_dictionary[img_class]['class_num']
        image_class_dictionary[line][class_num] = 1

    mean_X = X_train.mean(axis=0)

    for row in range(num_images):
        X_temp[row,:] = X_train[row,:] - mean_X

    covariance_mat = np.dot(X_temp.T, X_temp)
    eigenvalues, eigenvectors, = np.linalg.eigh(covariance_mat)

    idx = eigenvalues.argsort()[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors.T[::-1]

    k_eigenvalues = sorted_eigenvalues[:TOP_K]
    k_eigenvectors = sorted_eigenvectors[:TOP_K]
    alphas = np.dot(k_eigenvectors, X_train.T)

    W = np.random.rand(TOP_K, num_classes)*0.001

    for itr in range(MAX_ITER):
        WT_X = np.dot(W.T, alphas)
        for col in range(WT_X.shape[1]):
            WT_X[:,col] -= max(WT_X[:,col])
        prob = np.exp(WT_X)
        for img in range(prob.shape[1]):
            prob[:,img] = prob[:,img]/sum(prob[:,img])

        for clss in image_dictionary:
            imgs = image_dictionary[clss]['images']
            class_num = image_dictionary[clss]['class_num']
            num_clss_images = len(imgs)
            X = np.zeros((num_clss_images, img_x*img_y),dtype=np.float64)

            for i in range(num_clss_images):
                X[i,:] = imgs[i]

            J = np.zeros((TOP_K),dtype=np.float64)
            for img in range(num_images):
                T = image_class_dictionary[img][class_num]
                J += alphas[:,img]*(T - prob[class_num][img])

            J = J*(-1)
            W[:,class_num] = W[:,class_num] - eeta*J

    return W, k_eigenvalues, k_eigenvectors, image_class_label_dictionary

def processTestData(W, k_eigenvalues, k_eigenvectors, image_class_label_dictionary):
    test_data = readFile(test_path)
    test_labels = []

    for line in range(len(test_data)):
        filename = test_data[line].split(' ')[0]
        img_x, img_y = basewidth, basewidth

        image = Image.open(filename).convert('L')
        image = image.resize((basewidth,basewidth), Image.BICUBIC)
        img = np.asarray(image).flatten()
        alphas = np.dot(k_eigenvectors,img)

        max_prob = -1
        image_class = "None"
        WT_X = np.dot(W.T, alphas)
        WT_X -= max(WT_X)
        class_prob = np.exp(WT_X)
        class_prob = class_prob/sum(class_prob)

        for clss in range(class_prob.shape[0]):
            if class_prob[clss] >= max_prob:
                max_prob = class_prob[clss]
                image_class = image_class_label_dictionary[clss]

        test_labels.append(image_class)
        print(image_class)

    return test_labels

train_path = sys.argv[1]
test_path = sys.argv[2]

TOP_K = 32
basewidth = 32
MAX_ITER = 1000
eeta = 0.000001

W, k_eigenvalues, k_eigenvectors, image_class_label_dictionary = processTrainData()
predicted_data = processTestData(W, k_eigenvalues, k_eigenvectors, image_class_label_dictionary)
