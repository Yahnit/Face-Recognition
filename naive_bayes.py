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

        image = Image.open(filename).convert('L')
        image = image.resize((basewidth,basewidth), Image.BICUBIC)
        img = np.asarray(image).flatten()
        image_dictionary[img_class]['images'].append(img)
        X_train[line,:] = img

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

    for clss in image_dictionary:
        imgs = image_dictionary[clss]['images']
        num_clss_images = len(imgs)
        image_dictionary[clss]['prob'] = num_clss_images/num_images
        X = np.zeros((num_clss_images, img_x*img_y),dtype=np.float64)
        alphas = np.zeros((num_clss_images, TOP_K),dtype=np.float64)

        for i in range(num_clss_images):
            X[i,:] = imgs[i]

        alphas = np.dot(X, k_eigenvectors.T)

        mean_class = alphas.mean(axis=0)
        variance_class = alphas.var(axis=0)

        image_dictionary[clss]['mean'] = mean_class
        image_dictionary[clss]['variance'] = variance_class
    return image_dictionary, k_eigenvalues, k_eigenvectors

def getNormalDistValue(val, mean, variance):
    ans = 1/math.sqrt(2*math.pi*variance)
    exp_val = -((val-mean)*(val-mean))/(2*variance)
    exp_val = math.exp(exp_val)
    ans *= exp_val
    return ans

def processTestData(image_dictionary, k_eigenvalues, k_eigenvectors):
    test_data = readFile(test_path)
    test_labels = []

    for line in range(len(test_data)):
        filename = test_data[line].split(' ')[0]
        img_x, img_y = basewidth, basewidth

        image = Image.open(filename).convert('L')
        image = image.resize((basewidth,basewidth), Image.BICUBIC)
        img = np.asarray(image).flatten()

        max_prob = -1
        image_class = "None"

        for clss in image_dictionary:
            mean = image_dictionary[clss]['mean']
            variance = image_dictionary[clss]['variance']
            class_prob = image_dictionary[clss]['prob']

            alphas = np.dot(k_eigenvectors,img)

            img_prob = class_prob
            img_prob = img_prob * 10**9

            for comp in range(TOP_K):
                norm_dist = getNormalDistValue(alphas[comp], mean[comp], variance[comp])
                img_prob *= norm_dist

            if(img_prob > max_prob):
                max_prob = img_prob
                image_class = clss

        test_labels.append(image_class)
        print(image_class)

    return test_labels

train_path = sys.argv[1]
test_path = sys.argv[2]

TOP_K = 32
basewidth = 32
image_dictionary , k_eigenvalues, k_eigenvectors = processTrainData()
predicted_data = processTestData(image_dictionary, k_eigenvalues, k_eigenvectors)
