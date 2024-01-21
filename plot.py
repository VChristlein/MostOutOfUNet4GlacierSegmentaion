"""
Author: Maniraman Periyasamy

This module plots the train and validation losses which is used to determine the overfitting and convergence.
"""
import pandas as pd
import os
import numpy as np
import glob
import yaml
import re
import matplotlib
import matplotlib.pyplot as plt
import argparse
matplotlib.use('Agg')


if __name__ == '__main__':

    dir_path = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter", help="Enter the link to your yaml file which includes the hyperparameter",
                        default="hyperparameters/hyperparameters_reference.yaml")
    args = parser.parse_args()

    with open(dir_path +"/"+ args.parameter) as f:
        hyperPar = yaml.load(f)
    outpathData = dir_path + "/" + hyperPar['outputPath'] + "Data/"
    outpath = dir_path + "/" + hyperPar['outputPath']

    # find all the csv files containing the loss results.
    fileNames = [f for f in glob.glob(outpathData + "**/*.csv", recursive=True)]
    fileSort =[]
    for file in fileNames:
        m = re.search(r"\[([A-Za-z0-9., ]+)\]", file)
        fileSort.append(m.group(1))

    # pattern or word to be found in the folder name of the results which differentiates the model from one other
    #testList = ["augumented","unaugumented"]
    testList = ["1.0, 0.0"]#,"0.8, 0.2","0.5, 0.5","0.2, 0.8","0.0, 1.0"]

    # Corresponding name to be printed on the plots for each pattern or word from above.
    originalName = {"1.0, 0.0":"FCN_Adam"}#,"0.8, 0.2":"FCN_sgd","0.5, 0.5":"UnetPP_Adam","0.2, 0.8":"UnetPP_SGD","0.0, 1.0":"unet"}


    fileSort, fileNames = (list(t) for t in zip(*sorted(zip(fileSort, fileNames))))
    fileNames = np.array(fileNames)
    index = np.array([0,2,4,7,9])
    fileNameSel = fileNames

    valAcc = {}
    valErr = {}
    trnAcc = {}
    trnErr = {}

    print("\n\n")

    # consolidate all the results (from multiple days) into pd dataframe.
    for file in fileNameSel:
        df = pd.read_csv(file)
        m = re.search(r"\[([A-Za-z0-9., ]+)\]", file)
        curName = m.group(1)
        """if "_augumented" in file:
            curName = "augumented"
        elif "_unaugumented" in file:
            curName = "unaugumented"""
        k = list(df)
        if set(list(df)) == set(['val_loss', 'val_accuracy', 'loss', 'accuracy']):
            valAcc[curName] = df['val_accuracy']
            valErr[curName] = df['val_loss']
            trnAcc[curName] = df['accuracy']
            trnErr[curName] = df['loss']
        else:
            valAcc[curName] = df['val_acc']
            valErr[curName] = df['val_loss']
            trnAcc[curName] = df['acc']
            trnErr[curName] = df['loss']

        print("{} ,{:.4f} ,{:.4f} ,{:.4f} ,{:.4f} ,{:.4f} ,{:.4f} ,{:.4f} ,{:.4f}".format(
              curName,
              np.max(valAcc[curName]),np.array(valAcc[curName])[-1],
              np.min(valErr[curName]),np.array(valErr[curName])[-1],
              np.max(trnAcc[curName]),np.array(trnAcc[curName])[-1],
              np.min(trnErr[curName]),np.array(trnErr[curName])[-1]))
        print("\n")

    if not os.path.exists(outpath+ '/results'): os.makedirs(outpath+ '/results')

    # plot the loss results.
    for ratio in testList:
        fig = plt.figure()
        fig.gca().plot(np.arange(1,trnErr[ratio].size+1),trnErr[ratio],'X-', label='training loss', linewidth=1.0)
        fig.gca().plot(np.arange(1,valErr[ratio].size+1),valErr[ratio],'o-', label='validation loss', linewidth=1.0)
        #axis = fig.gca()
        fig.gca().set_ylim(top = 1.5)
        fig.gca().set_ylim(bottom=0)
        fig.gca().set_xlim(right = 120)
        fig.gca().grid(which='minor', linestyle='--')
        fig.gca().set_xlabel('epoch')
        fig.gca().set_ylabel('loss')
        fig.gca().legend(loc = "upper right", fontsize = 18)
        fig.gca().set_title(originalName[ratio], fontsize = 20)
        fig.gca().minorticks_on()
        fig.gca().grid(which='minor', linestyle='--')
        fig.tight_layout()
        fig.savefig(outpath+ '/results/'+"["+ratio+"].png",dpi = 300)
