"""
Author: Maniraman Periyasamy

This modulde delineates the segmentation results along with calculating the segmentation results of both zones and front.

"""

import numpy as np
import yaml
import glob
import os
import imageio
import skimage.io as io
import pandas as pd
import cv2
import argparse





######################################## To change

def lcc_mask(bin_img):
    """
    Finds the largest connected component in a binary image

    Args:
        bin_img (ndarray): Binary image

    Returns: (ndarray) mask with largest connected component

    """

    contours = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    area = []
    for i, contour in enumerate(contours):
        area.append(cv2.moments(contour)['m00'])

    sort_area = np.argsort(area)

    labeled_img = np.zeros(bin_img.shape, dtype=np.uint8)
    cv2.drawContours(labeled_img, contours, sort_area[-1], color=255, thickness=-1)

    return labeled_img




####################################################


def predAccuracy(target, prediction):

    """
    Calculate the Pixel-wise accuracy and F1 score for binary class image segmentation.

    Args:
        target (ndarray): Ground truth in the form of numpy array
        prediction (ndarray): Predictions in the form of numpy array

    Returns: (float) pixel-wise accuracy, (float) F1 Score

    """

    target = (target/255).astype(np.int64)
    prediction = (prediction / 255).astype(np.int64)


    tn, fp, fn, tp = np.bincount(target * 2 + prediction).ravel()

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total
    f1_score = (2 * tp) / ((2 * tp) + fp + fn)

    return accuracy, f1_score


def iouAccuracy(target, prediction):
    """
    Calculates the Intersection over Union metric for the binary class image segmentation

    Args:
        target (ndarray): Ground truth in the form of numpy array
        prediction (ndarray): Predictions in the form of numpy array

    Returns: (float) IoU

    """
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter", help="Enter the link to your yaml file which includes the hyperparameter",
                        default="hyperparameters/hyperparameters_reference.yaml")
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))

    with open(dir_path +"/"+ args.parameter) as f:
        hyperPar = yaml.load(f)

    # pattern or word to be found in the folder name of the results which differentiates the model from one other
    # testList = ["_augumented","_unaugumented"]
    testList = ["[1.0, 0.0]"]#,"[0.5, 0.5]"]#, "[0.8, 0.2]", "[0.5, 0.5]", "[0.2, 0.8]", "[0.0, 1.0]"]
    outpath = dir_path + "/" + hyperPar['outputPath'] + 'Data/'

    testImages = glob.glob(hyperPar["testPath"] + 'masks_lines/*.png')
    results = os.listdir(outpath)

    # read all the files in the output folder
    FilesList = []
    for path, subdirs, files in os.walk(outpath):
        for name in files:
            FilesList.append(os.path.join(path, name))
            print(os.path.join(path, name))

    generatedImages = {}
    iouResult = {}
    accuracyResult = {}
    f1_score = {}
    boundary_TPacc = {}
    boundary_FPacc = {}
    boundary_H_M = {}

    for ratio in testList:
        generatedImages[ratio] = []
        iouResult[ratio] = {}
        accuracyResult[ratio] = {}
        f1_score[ratio] = {}
        boundary_TPacc[ratio] = {}
        boundary_FPacc[ratio] = {}

    boundary_H_M["mean"] = {}
    boundary_H_M["mean_removed"] = {}
    boundary_H_M["removed"] = {}
    boundary_H_M["area"] = {}

    iouResult["Qfactor"] = {}
    accuracyResult["Qfactor"] = {}
    f1_score["Qfactor"] = {}

    iouResult["size"] = {}
    accuracyResult["size"] = {}
    f1_score["size"] = {}

    iouResult["res"] = {}
    accuracyResult["res"] = {}
    f1_score["res"] = {}

    direc = outpath + 'result/'
    if not os.path.exists(direc): os.makedirs(direc)

    # Find all the images in the output folder and group it via individual pattern or word
    for names in FilesList:
        a = np.array([word in names for word in testList])
        ind = np.where(a == True)
        if ind[0].size != 0:
            if names[-3:] == "png":
                generatedImages[testList[ind[0][0]]].append(names)
    count = 0
    test_count = 1
    for ratio in testList:
        ratioImages = generatedImages[ratio]
        for testImage in testImages:

            testInputPath = str(testImage).replace("masks_lines", "images")
            testInputPath = testInputPath.replace("_front", "")
            testInput = cv2.imread(testInputPath)

            currImage = testImage.rsplit("/", 1)[1]
            currImage = currImage.replace("_front", "")
            target = imageio.imread(testImage)
            iouResult["size"][currImage] = target.size
            accuracyResult["size"][currImage] = target.size
            f1_score["size"][currImage] = target.size

            iouResult["Qfactor"][currImage] = currImage[-5]
            accuracyResult["Qfactor"][currImage] = currImage[-5]
            f1_score["Qfactor"][currImage] = currImage[-5]

            iouResult["res"][currImage] = currImage[-8:-6]
            accuracyResult["res"][currImage] = currImage[-8:-6]
            f1_score["res"][currImage] = currImage[-8:-6]

            iouResult["res"][currImage] = iouResult["res"][currImage].replace("_", "0")
            accuracyResult["res"][currImage] = accuracyResult["res"][currImage].replace("_", "0")
            f1_score["res"][currImage] = f1_score["res"][currImage].replace("_", "0")

           # Dilate the boundaries
            edges_target_original = target
            if iouResult["res"][currImage] == "05":
                window = (60, 60)
            elif iouResult["res"][currImage] == "06":
                window = (25, 25)
            elif iouResult["res"][currImage] == "20":
                window = (15, 15)
            elif iouResult["res"][currImage] == "10":
                window = (30, 30)
            elif iouResult["res"][currImage] == "50":
                window = (6, 6)

            # convert the ground truth of fronts from grayscale to rgb image
            # this is done for differentiating the ground truth and prediction
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, window)
            edges_target = cv2.dilate(edges_target_original, kernel, iterations=1)

            edges_target_rgb = cv2.cvtColor(edges_target, cv2.COLOR_GRAY2RGB)
            edges_target_rgb[np.where((edges_target_rgb == [255, 255, 255]).all(axis=2))] = [0, 175, 0]

            testInput[np.where((edges_target_rgb == [0, 175, 0]).all(axis=2))] = [0, 175, 0]
            cv2.imwrite(direc+"original_"+currImage, testInput)

            for img in ratioImages:
                if currImage in img:

                    # read the prediction
                    prediction = imageio.imread(img)

                    # find the largest connected component
                    prediction = lcc_mask(prediction)
                    io.imsave(direc + currImage+"_predict.png", prediction)

                    # Detect the edges of the prediction
                    edges_predict = cv2.Canny(prediction, 200, 400)
                    edges_predict_dilated = cv2.dilate(edges_predict, kernel, iterations=1)

                    # calculate pixel-wise accuracy and dice coefficient for fronts
                    boundary_TPacc[ratio][currImage], boundary_FPacc[ratio][currImage] = predAccuracy(edges_target.flatten(),
                                edges_predict_dilated.flatten())

                    # calculate IoU for zones
                    iouResult[ratio][currImage] = iouAccuracy(target, prediction)

                    # calculate pixel-wise accuracy and dice coefficient for zone
                    accuracyResult[ratio][currImage], f1_score[ratio][currImage] = predAccuracy(target.flatten(),
                                                                                             prediction.flatten())


                    # dilate the prediction edges for tollerence
                    edges_predict = cv2.dilate(edges_predict, kernel, iterations=1)

                    # convert the prediction of fronts from grayscale to rgb image
                    # this is done for differentiating the ground truth and prediction

                    edges_predict_rgb = cv2.cvtColor(edges_predict, cv2.COLOR_GRAY2RGB)
                    edges_predict_rgb[np.where((edges_predict_rgb == [255, 255, 255]).all(axis=2))] = [0,0,255]
                    testInput[np.where((edges_predict_rgb == [0,0,255]).all(axis=2))] = [0,0,255]


                    edges_predict_rgb[np.where((edges_predict_rgb == [0,0,255]).all(axis=2))] = [0, 175, 0]
                    testInput[np.where((edges_predict_rgb != [0,0,0]).any(axis=2) &
                                       (edges_predict_rgb == edges_target_rgb).all(axis=2))] = [0,255,255]


                    cv2.imwrite(direc +currImage+"_bound.png", testInput)

                    count = count + 1
                    print(count)
            test_count += 1


    # write the results to csv folder.
    iou = pd.DataFrame(iouResult)
    iou.to_csv(direc + "iou.csv")

    pixPred = pd.DataFrame(accuracyResult)
    pixPred.to_csv(direc + "pixel.csv")

    f1Score = pd.DataFrame(f1_score)
    f1Score.to_csv(direc + "f1Score.csv")

    TP_score = pd.DataFrame(boundary_TPacc)
    TP_score.to_csv(direc + "TP_score.csv")

    FP_score = pd.DataFrame(boundary_FPacc)
    FP_score.to_csv(direc + "FP_n_score.csv")

