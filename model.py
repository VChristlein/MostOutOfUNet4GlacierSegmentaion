"""
Author: Maniraman Periyasamy

This module implements various types of U-Net model and provides train and test fuctions using Keras API.
The list of models implemented are as follows :

    1) U-Net as proposed by Ronneberger et al. in paper "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    2) U-Net as suggested by Zhang et al. in paper "Automatically delineating the calving front of Jakobshavn Isbræ from multitemporal TerraSAR-X images: a deep learning approach"
    3) U-Net baseline segmentation model as given in report
    4) U-Net baseline segmentation model with various normalization layers
        a) Layer Normalization
        b) Group Normalization
        c) Instance Normalization
        d) Weight Normaliztion
    5) U-Net basline segmentation model with Dropouts instead of normalization
    6) U-Net basline segmentation model with Dropouts and Batch Normalization
    7) FCNN as given in report
    8) Nested U-Net model as suggested by Zhou et al. in paper "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"

"""
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from clr_callback import *
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import binary_crossentropy
import tensorflow_addons as tfa
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Cropping2D

from tensorflow.keras import backend as K

from pathlib import Path
import skimage.io as io
import utilities
import cv2
import sklearn.metrics
import pickle
import time


class TimedStopping(Callback):
    """
    Adapted from: https://github.com/keras-team/keras/issues/1625
    """

    def __init__(self, seconds=None, verbose=0):
        super(Callback, self).__init__()

        self.start_time = 0
        self.seconds = seconds
        self.verbose = verbose
        self.timeFlag = False

    def on_train_begin(self, logs={}):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        if time.time() - self.start_time > self.seconds:
            self.model.stop_training = True
            self.timeFlag = True
            if self.verbose:
                print('Stopping after %s seconds.' % self.seconds)

    def on_batch_end(self, batch, logs={}):
        if time.time() - self.start_time > self.seconds:
            self.model.stop_training = True
            self.timeFlag = True
            if self.verbose:
                print('Stopping after %s seconds.' % self.seconds)



class unet:
    """
    creates a U-Net model based on the parameters given in the initializer.
    This class also provides the train and test fuction which fits and predicts the U-Net model using Keras API

    Args:
        trainGenerator: Keras data generator with train dataset
        valGenerator: Keras data generator with validation dataset
        stepsPerEpoch (int): Number of train steps in an training epoch
        validationSteps (int):  Number of validation steps in an training epoch
        outputPathCheck (str):  Relative path to the output folder and the output model name.
        outputPath (str):  Relative path to the output folder
        testPath (str):  Relative path to the test data folder
        patchSize (int):  Number of pixels in one side of the square input patch
        epochs (int, default is 100):  Maximum number of epochs
        unetType (str, default is unet):  Type of net to be tested
        loss (keras loss, default is BCE):  Loss function to be used
        loss_weight (float, default is 1.0):  fraction of loos function to be considered
        metrics (list(str),  default is ['accuracy']):  List of keras metrics
        optimizer (keras optimizer, default is Adam):  Optimizer function to be used
        inputSize (tuple(int,int,int), default is (256,256,1)):  Dimensions of the input to the model.
        patience (int, default is 30):  Number of epochs to wait for validating the convergence.
        pretrainedWeights (str, default is None):  Relative path to the pre-trained weights file
        pretrainedModel (str, default is None):  Relative path to the pre-trained model file which includes the loss function and optimizer details.
        threshold (float, default is 0.5): Default threshold value to be used for binary classification if optimal threshold is not calculated
        validationPath (str, default is None): Relative path to the validation dataset
        dilationRate (int, default is None): Dialation rate for dilated convolutions
        lossType (str, default is BCE): Type of loss
        lossWeights (tuple(int,int), default is (1.0,0.0)):  Weights for BCE and Dice loss if the lossType is 'combined'
        model_train (bool, default is True): Flag to indicate whether the model is to be trained
        dropout (float, default is 0.0): Dropput rates to be used in case of dropout layer
        transferTrain (str, default is None): Type of transfer learning to be used if any
    """
    def __init__(self, trainGenerator, valGenerator, stepsPerEpoch, validationSteps, outputPathCheck, outputPath, testPath,
                 patchSize, epochs = 100, unetType = "unet", loss = "binary_crossentropy",
                 loss_weight = [1], metrics = ["accuracy"],
                 optimizer = Adam(lr = 1e-4), inputSize = (256,256,1), patience = 30,  pretrainedWeights = None, pretrainedModel = None,
                 threshold = 0.5, validationPath = None, dilationRate = None, lossType = binary_crossentropy,
                 lossWeights = [1.0,0.0], model_train = True, dropout = 0.0, transferTrain = None):


        self.__trainGenerator = trainGenerator
        self.__valGenerator = valGenerator
        self.__unetType = unetType
        self.__patchSize = patchSize
        self.__epochs = epochs
        self.__stepsPerEpoch = stepsPerEpoch
        self.__validationSteps = validationSteps
        self.__outputPathCheck = outputPathCheck
        self.__outputPath = outputPath
        self.__loss = loss
        self.__lossWeight = loss_weight
        self.__metrics = metrics
        self.__optimizer = optimizer
        self.__pretrainedWeights = pretrainedWeights
        self.__pretrainedModel = pretrainedModel
        self.__inputSize = inputSize
        self.__patience = patience
        self.__testPath = testPath
        self.__validationPath = validationPath
        self.__threshold = threshold
        self.__dilationRate = dilationRate
        self.__lossType = lossType
        self.__lossWeights = lossWeights
        self.__modelTrain = model_train
        self.__dropoutRate = dropout
        self.__transferTrain = transferTrain

        # build the U-Net
        self.buildModel()

        # add loss function and optimizer to the U-Net
        self.modelCompile()

        # add checkpoints, earlystopping and cyclic learnind rate to the model
        self.checkPoint()

        # load pre-trained weights if available. Often used to continue training on the second day.
        if not self.__pretrainedWeights is None:
           self.model.load_weights(self.__pretrainedWeights)

        # if the pre-trained model is available, load pre-trainied model.
        # Keras load_model function have bug in loading combined loss function. hence the model was loaded without
        # combined loss and then the loss function is compiled.
        # this addidtional complie makes the optimizer to loose its momentum details. Therefore it is loaded seperately.
        # to load the optimizer values, the model as to run at least one epoch. Hence it is trained for single epoch
        # and then the optimizer value is loaded.

        if not self.__pretrainedModel is None:
            if self.__lossType == "combined":
                self.model = load_model(self.__pretrainedModel, custom_objects={'loss': self.custLoss(1.0)}, compile=False)
                self.model.compile(optimizer=self.__optimizer, loss=self.custLoss(1.0), loss_weights=self.__lossWeight,
                                   metrics=self.__metrics)

                clr_triangular = CyclicLR(mode="triangular", base_lr=1e-7, max_lr=1e-2,
                                          step_size=5 * self.__stepsPerEpoch)
                timeStop = TimedStopping(seconds=1, verbose=1)
                modelCheckpoint = [timeStop, clr_triangular]

                # condition to check whether the model is going to be trained. if not loading optimizer is of no use.
                if self.__modelTrain:
                    history = self.model.fit_generator(self.__trainGenerator, steps_per_epoch=self.__stepsPerEpoch,
                                                        epochs=self.__epochs, validation_data=self.valGenerator,
                                                        validation_steps=self.__validationSteps,
                                                        callbacks=modelCheckpoint, verbose=1)

                    with open(self.__outputPath + "optimizer.pkl", 'rb') as f:
                        weight_values = pickle.load(f)
                    self.model.optimizer.set_weights(weight_values)
                self.model.load_weights(self.__pretrainedModel)

            # if the model doesnot use combined loss. simple load_model is sufficient.
            elif self.__lossType == "diceLoss":
                self.model = load_model(self.__pretrainedModel, custom_objects={'loss': self.diceLoss})
            else:
                self.model = load_model(self.__pretrainedModel)

        # if tranfer learning is used, then the corresponding layers are frozen depending on the type of transfer
        # learning as given in the report.
        if not self.__transferTrain is None:
            if self.__transferTrain == "encoder":
                for i in range(39, len(self.model.layers)):
                    self.model.layers[i].trainable = False
            elif self.__transferTrain == "decoder":
                for i in range(29):
                    self.model.layers[i].trainable = False
            elif self.__transferTrain == "full":
                for i in range(39, len(self.model.layers)):
                    self.model.layers[i].trainable = False
                for i in range(29):
                    self.model.layers[i].trainable = False
            else:
                pass


    def buildModel(self):
        """
        This function builds the model based on the hyperparameters and type of U-Net architecture required.

        Returns: None

        """
        if self.__unetType == "unet":
            self.unet()
        elif self.__unetType == "unet_Enze19_2":
            self.unet_Enze19_2()
        elif self.__unetType=="dilated_convo":
            self.dilated_convo()
        elif self.__unetType=="dilated_convo2":
            self.dilated_convo2()
        elif self.__unetType=="dilated_convo3":
            self.dilated_convo3()
        elif self.__unetType=="dilated_convo4":
            self.dilated_convo4()
        elif self.__unetType=="resBlock":
            self.resBlock()
        elif self.__unetType=="dilatedResBlock":
            self.dilatedResBlock()
        elif self.__unetType=="dilatedResBlockLayerNorm":
            self.dilatedResBlockLayerNorm()
        elif self.__unetType=="dilatedResBlockWeightNorm":
            self.dilatedResBlockWeightNorm()
        elif self.__unetType=="dilatedResBlockInstanceNorm":
            self.dilatedResBlockInstanceNorm()
        elif self.__unetType=="dilatedResBlockGroupNorm":
            self.dilatedResBlockGroupNorm()
        elif self.__unetType=="dilatedResBlockWithDropout":
            self.dilatedResBlockWithDropout()
        elif self.__unetType == "dilatedResBlockWithBNandDropout":
            self.dilatedResBlockWithBNandDropout()
        elif self.__unetType == "unetPlusPlus":
            self.unetPlusPlus()
            #self.model = Nestnet()
        elif self.__unetType == "FCN":
            self.FCN()

        # Generate a report file indicating the architecture and number of parameters.
        with open(self.__outputPath + 'report.txt', 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))


    def custLoss(self, temp=1.0):
        """
        Combined loss funtion formed by the weighted combination of BCE and Dice loss

        Returns: combined loss function

        """
        def custom_loss(y_true, y_pred):
            return self.__lossWeights[0] * binary_crossentropy(y_true, y_pred) + self.__lossWeights[1] * self.diceLoss(y_true, y_pred)
        return custom_loss

    def diceLoss(self, y_true, y_pred):
        """
        Calculates the Dice loss.

        Args:
            y_true (list(float)): Ground Truth
            y_pred (list(float)): Prediction from the model

        Returns: (float) Dice loss value

        """
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        intersection = K.sum(K.abs(y_true * y_pred))
        loss = (2. * intersection) / (K.sum(y_true) + K.sum(y_pred))
        return 1 - loss


    def modelCompile(self):
        """
        Compile the loss function and optimizer to be used
        Returns: None
        """
        if self.__lossType == "combined":
            self.model.compile(optimizer=self.__optimizer, loss=self.custLoss(1.0), loss_weights= self.__lossWeight,
                           metrics=self.__metrics)

        elif self.__lossType == "diceLoss":
            loss = self.diceLoss
            self.model.compile(optimizer=self.__optimizer, loss=loss, loss_weights=self.__lossWeight,
                               metrics=self.__metrics)
        else:
            self.model.compile(optimizer=self.__optimizer, loss=self.__lossType, loss_weights=self.__lossWeight,
                               metrics=self.__metrics)

    def checkPoint(self):

        """
        create the list of callbacks to be send during the training.
        list of callbacks impleneted are:
            - cyclic learning rate
            - timed stopping of training (23 hours restriction)
            - early stopping of the model
            - model checkpoints to be saved
        Returns: None

        """
        clr_triangular = CyclicLR(mode="triangular",base_lr=1e-7, max_lr=1e-2, step_size= 5* self.__stepsPerEpoch)
        self.timeStop = TimedStopping(seconds=72000, verbose=1)
        earlyStop = EarlyStopping(monitor= 'val_loss',patience=self.__patience,restore_best_weights=True)
        modelcheck = ModelCheckpoint(self.__outputPathCheck, monitor='val_loss', verbose=0,save_best_only=True)
        self.modelCheckpoint = [modelcheck, self.timeStop,earlyStop,clr_triangular]




    def train(self):

        """
        Fuction to train the generated model.

        Returns: (Dict) training loss and accuracy, (bool) timeFlag which indicates whether the model is stopped due to time restriction

        """

        self.History = self.model.fit_generator(self.__trainGenerator,steps_per_epoch=self.__stepsPerEpoch,
                                                epochs=self.__epochs,validation_data=self.valGenerator,
                                                validation_steps=self.__validationSteps,
                                                callbacks=self.modelCheckpoint, verbose=1)

        # save the optimizer state to continue trainig the next day.
        symbolic_weights = getattr(self.model.optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights)
        with open(self.__outputPath+"optimizer.pkl", 'wb') as f:
            pickle.dump(weight_values, f)




        return self.History, self.timeStop.timeFlag

    def predAccuracy(self, target, prediction):
        """
        Calculate the pixel-wise accuracy and F1 score (Dice coefficient)

        Args:
            target (list(float)): Ground Truth
            prediction (list(float)): Prediction from the model

        Returns: (float) accuracy, (float) F1 score

        """
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(target, prediction).ravel()
        tn, fp, fn, tp = np.float64(tn), np.float64(fp), np.float64(fn), np.float64(tp)
        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total
        f1_score = (2 * tp) / ((2 * tp) + fp + fn)
        return accuracy, f1_score

    def prediction(self, filename, threshold):
        """
        Generates the pixel-wise prediction on a image

        Args:
            filename (str): Relative path to the image to be tested
            threshold (float): Threshold value for binary classification

        Returns: (ndarray) predicted mask

        """

        img = io.imread(filename, as_gray=True)
        img = img / 255
        img_pad = cv2.copyMakeBorder(img, 0, (self.__patchSize - img.shape[0]) % self.__patchSize, 0,
                                     (self.__patchSize - img.shape[1]) % self.__patchSize, cv2.BORDER_CONSTANT)
        p_img, i_img = utilities.extract_grayscale_patches(img_pad, (self.__patchSize, self.__patchSize),
                                                           stride=(self.__patchSize, self.__patchSize))
        p_img = np.reshape(p_img, p_img.shape + (1,))

        p_img_predicted = self.model.predict(p_img)

        p_img_predicted = np.reshape(p_img_predicted, p_img_predicted.shape[:-1])
        img_mask_predicted_recons = utilities.reconstruct_from_grayscale_patches(p_img_predicted, i_img)[0]

        # unpad and normalize
        img_mask_predicted_recons_unpad = img_mask_predicted_recons[0:img.shape[0], 0:img.shape[1]]
        img_mask_predicted_recons_unpad_norm = cv2.normalize(src=img_mask_predicted_recons_unpad, dst=None, alpha=0,
                                                             beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # quantization to make the binary masks
        img_mask_predicted_recons_unpad_norm[img_mask_predicted_recons_unpad_norm < int(threshold*255)] = 0
        img_mask_predicted_recons_unpad_norm[img_mask_predicted_recons_unpad_norm >= int(threshold*255)] = 255

        return  img_mask_predicted_recons_unpad_norm

    def threshold_Check(self):
        """
        Function to find the optimal threshold for binary classification.

        Note:
            threshold in the range of [0.4, 0.75] are tested with an interval of 0.05

        Returns: None

        """
        thersholdRange = np.arange(0.4, 0.75,0.05)
        dice_mean = np.zeros((thersholdRange.shape[0], 2))
        for i in range(thersholdRange.shape[0]):
            dice_all = []
            for filename in Path(self.__validationPath, 'images').rglob('*.png'):
                predictionResult = self.prediction(filename=filename, threshold=thersholdRange[i])

                # io.imsave(Path(str(self.__outputPath), Path(filename).name), img_mask_predicted_recons_unpad_norm)

                gt_path = str(Path(self.__validationPath, 'masks_zones'))
                gt_name = filename.name
                gt = io.imread(str(Path(gt_path, gt_name)), as_gray=True)

                #print("Name:%s" % (str(filename)), np.unique(gt),np.unique(predictionResult))

                pixel, dice = self.predAccuracy(gt.flatten(), predictionResult.flatten())
                dice_all.append(dice)
                print("i:%s dice:%f" % (str(filename), dice))
            print("Iteration: %d, Avg_Dice:%f" % (i, np.mean(dice_all)))
            dice_mean[i] = np.array([np.mean(dice_all), thersholdRange[i]])

        pos = np.argmax(dice_mean, axis=0)
        self.__threshold = dice_mean[pos[0]][1]

    def test(self):

        """
        Creates the prediction mask for all the images present in test dataset.

        Returns: None

        """

        # construct the images
        count = 0
        for filename in Path(self.__testPath, 'images').rglob('*.png'):
            predictionResult = self.prediction(filename=filename, threshold=self.__threshold)
            io.imsave(Path(str(self.__outputPath), Path(filename).name), predictionResult)
            count+=1
            print(filename, count)
        # store the optimal threshold value
        with open(self.__outputPath + 'thershold.txt', 'w') as file:
            file.write(str(self.__threshold))


    def unet(self):
        """
        Constructs the U-Net model as proposed by Ronneberger et al. in paper
        "U-Net: Convolutional Networks for Biomedical Image Segmentation"
        """
        inputs = Input(self.__inputSize)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, padding='same', activation='sigmoid')(conv9)

        self.model = Model(input=inputs, output=conv10)

    def unet_Enze19_2(self):
        """
        Constructs the U-Net model as suggested by Enze Zhang et al. in paper
        "Automatically delineating the calving front of Jakobshavn Isbræ from multitemporal TerraSAR-X images: a deep learning approach"
        """

        inputs = Input(self.__inputSize)
        conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)
        conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 128

        conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)
        conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 64

        conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)
        conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # 32

        conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)
        conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)

        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  # 16

        conv5 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = LeakyReLU(alpha=.1)(conv5)
        conv5 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = LeakyReLU(alpha=.1)(conv5)

        up6 = Conv2DTranspose(256, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv5)
        merge6 = concatenate([conv4, up6], axis=3)

        conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = Conv2DTranspose(128, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv6)
        merge7 = concatenate([conv3, up7], axis=3)

        conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = Conv2DTranspose(64, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv7)
        merge8 = concatenate([conv2, up8], axis=3)

        conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        conv8 = BatchNormalization()(conv8)

        up9 = Conv2DTranspose(32, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv8)
        merge9 = concatenate([conv1, up9], axis=3)

        conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        conv9 = BatchNormalization()(conv9)

        conv10 = Conv2D(1, 3, padding='same', activation='sigmoid')(conv9)

        self.model = Model(inputs=inputs, outputs=conv10)

    def dilated_convo(self):
        """
        Constructs the baseline segmentation model with dilated bottelneck with dilation rate given in intializer
        """
        inputs = Input(self.__inputSize)
        conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)
        conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 128

        conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)
        conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 64

        conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)
        conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # 32

        conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)
        conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)

        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  # 16

        conv5 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal', dilation_rate=self.__dilationRate)(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = LeakyReLU(alpha=.1)(conv5)

        up6 = Conv2DTranspose(256, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv5)
        merge6 = concatenate([conv4, up6], axis=3)

        conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = Conv2DTranspose(128, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv6)
        merge7 = concatenate([conv3, up7], axis=3)

        conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = Conv2DTranspose(64, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv7)
        merge8 = concatenate([conv2, up8], axis=3)

        conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        conv8 = BatchNormalization()(conv8)

        up9 = Conv2DTranspose(32, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv8)
        merge9 = concatenate([conv1, up9], axis=3)

        conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        conv9 = BatchNormalization()(conv9)

        conv10 = Conv2D(1, 3, padding='same', activation='sigmoid')(conv9)

        self.model = Model(inputs=inputs, outputs=conv10)

    def dilated_convo2(self):
        """
        Constructs the baseline segmentation model with dilated bottelnecks with dilation rate of (2,2) and (4,4)
        """
        inputs = Input(self.__inputSize)
        conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)
        conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 128

        conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)
        conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 64

        conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)
        conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # 32

        conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)
        conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)

        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  # 16

        conv5_1 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(pool4)
        conv5_1 = BatchNormalization()(conv5_1)
        conv5_1 = LeakyReLU(alpha=.1)(conv5_1)

        conv5_2 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(pool4)
        conv5_2 = BatchNormalization()(conv5_2)
        conv5_2 = LeakyReLU(alpha=.1)(conv5_2)

        conv5 = Add()([conv5_1,conv5_2])

        up6 = Conv2DTranspose(256, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv5)
        merge6 = concatenate([conv4, up6], axis=3)

        conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = Conv2DTranspose(128, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv6)
        merge7 = concatenate([conv3, up7], axis=3)

        conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = Conv2DTranspose(64, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv7)
        merge8 = concatenate([conv2, up8], axis=3)

        conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        conv8 = BatchNormalization()(conv8)

        up9 = Conv2DTranspose(32, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv8)
        merge9 = concatenate([conv1, up9], axis=3)

        conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        conv9 = BatchNormalization()(conv9)

        conv10 = Conv2D(1, 3, padding='same', activation='sigmoid')(conv9)

        self.model = Model(inputs=inputs, outputs=conv10)

    def dilated_convo3(self):
        """
        Constructs the baseline segmentation model with dilated bottelnecks with dilation rate of (2,2), (4,4) and (8,8)
        """
        inputs = Input(self.__inputSize)
        conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)
        conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 128

        conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)
        conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 64

        conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)
        conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # 32

        conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)
        conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)

        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  # 16

        # BottleNeck layer

        conv5_1 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(pool4)
        conv5_1 = BatchNormalization()(conv5_1)
        conv5_1 = LeakyReLU(alpha=.1)(conv5_1)

        conv5_2 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(pool4)
        conv5_2 = BatchNormalization()(conv5_2)
        conv5_2 = LeakyReLU(alpha=.1)(conv5_2)

        conv5_3 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal', dilation_rate=(8, 8))(pool4)
        conv5_3 = BatchNormalization()(conv5_3)
        conv5_3 = LeakyReLU(alpha=.1)(conv5_3)

        conv5 = Add()([conv5_1,conv5_2,conv5_3])


        # up sampling
        up6 = Conv2DTranspose(256, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv5)
        merge6 = concatenate([conv4, up6], axis=3)

        conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = Conv2DTranspose(128, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv6)
        merge7 = concatenate([conv3, up7], axis=3)

        conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = Conv2DTranspose(64, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv7)
        merge8 = concatenate([conv2, up8], axis=3)

        conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        conv8 = BatchNormalization()(conv8)

        up9 = Conv2DTranspose(32, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv8)
        merge9 = concatenate([conv1, up9], axis=3)

        conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        conv9 = BatchNormalization()(conv9)

        conv10 = Conv2D(1, 3, padding='same', activation='sigmoid')(conv9)

        self.model = Model(inputs=inputs, outputs=conv10)

    def dilated_convo4(self):

        """
        Constructs the baseline segmentation model with dilated bottelnecks with dilation rate of (1,1), (2,2), (4,4) and (8,8)
        """

        inputs = Input(self.__inputSize)
        conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)
        conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 128

        conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)
        conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 64

        conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)
        conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # 32

        conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)
        conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)

        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  # 16

        # Bottle Neck later

        conv5_0 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal', dilation_rate=(1, 1))(pool4)
        conv5_0 = BatchNormalization()(conv5_0)
        conv5_0 = LeakyReLU(alpha=.1)(conv5_0)


        conv5_1 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(pool4)
        conv5_1 = BatchNormalization()(conv5_1)
        conv5_1 = LeakyReLU(alpha=.1)(conv5_1)

        conv5_2 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(pool4)
        conv5_2 = BatchNormalization()(conv5_2)
        conv5_2 = LeakyReLU(alpha=.1)(conv5_2)

        conv5_3 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal', dilation_rate=(8, 8))(pool4)
        conv5_3 = BatchNormalization()(conv5_3)
        conv5_3 = LeakyReLU(alpha=.1)(conv5_3)

        conv5 = Add()([conv5_0,conv5_1,conv5_2,conv5_3])


        # up sampling
        up6 = Conv2DTranspose(256, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv5)
        merge6 = concatenate([conv4, up6], axis=3)

        conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = Conv2DTranspose(128, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv6)
        merge7 = concatenate([conv3, up7], axis=3)

        conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = Conv2DTranspose(64, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv7)
        merge8 = concatenate([conv2, up8], axis=3)

        conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        conv8 = BatchNormalization()(conv8)

        up9 = Conv2DTranspose(32, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv8)
        merge9 = concatenate([conv1, up9], axis=3)

        conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        conv9 = BatchNormalization()(conv9)

        conv10 = Conv2D(1, 3, padding='same', activation='sigmoid')(conv9)

        self.model = Model(inputs=inputs, outputs=conv10)

    def resBlock(self):

        """
        Constructs the baseline segmentation model with residual bottelneck
        """

        inputs = Input(self.__inputSize)
        conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)
        conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 128

        conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)
        conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 64

        conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)
        conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # 32

        conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)
        conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)

        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  # 16

        # Resblock
        conv5_1 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal')(pool4)
        conv5_1 = BatchNormalization()(conv5_1)
        conv5_1 = LeakyReLU(alpha=.1)(conv5_1)

        conv5_2 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal')(conv5_1)
        conv5_2 = BatchNormalization()(conv5_2)
        conv5_2 = LeakyReLU(alpha=.1)(conv5_2)

        shortCut = Conv2D(512, 1, padding='same', kernel_initializer='he_normal')(conv5_1)

        conv5 = Add()([shortCut,conv5_2])
        conv5 = BatchNormalization()(conv5)
        conv5 = LeakyReLU(alpha=.1)(conv5)

        #resBlock End
        up6 = Conv2DTranspose(256, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv5)
        merge6 = concatenate([conv4, up6], axis=3)

        conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = Conv2DTranspose(128, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv6)
        merge7 = concatenate([conv3, up7], axis=3)

        conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = Conv2DTranspose(64, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv7)
        merge8 = concatenate([conv2, up8], axis=3)

        conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        conv8 = BatchNormalization()(conv8)

        up9 = Conv2DTranspose(32, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv8)
        merge9 = concatenate([conv1, up9], axis=3)

        conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        conv9 = BatchNormalization()(conv9)

        conv10 = Conv2D(1, 3, padding='same', activation='sigmoid')(conv9)

        self.model = Model(inputs=inputs, outputs=conv10)

    def dilatedResBlock(self):
        """
        Constructs the baseline segmentation model with dilated bottelneck and residual connection with dilation rate of (4,4)
        """
        inputs = Input(self.__inputSize)
        conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal', name="convo1")(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)
        conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal', name="convo2")(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 128

        conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal', name="convo3")(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)
        conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal', name="convo4")(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 64

        conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal',  name="convo5")(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)
        conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal',  name="convo6")(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # 32

        conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal',  name="convo7")(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)
        conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal',  name="convo8")(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)

        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  # 16
        #29

        # Resblock

        conv5_1 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal', dilation_rate=(4,4), name="Bconvo1")(pool4)
        conv5_1 = BatchNormalization()(conv5_1)
        conv5_1 = LeakyReLU(alpha=.1)(conv5_1)

        conv5_2 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal', dilation_rate=(4,4), name="Bconvo2")(conv5_1)
        conv5_2 = BatchNormalization()(conv5_2)
        conv5_2 = LeakyReLU(alpha=.1)(conv5_2)

        shortCut = Conv2D(512, 1, padding='same', kernel_initializer='he_normal', name="Bconvo3")(conv5_1)

        conv5 = Add()([shortCut, conv5_2])
        conv5 = BatchNormalization()(conv5)
        conv5 = LeakyReLU(alpha=.1)(conv5)

        #10

        # resBlock End
        up6 = Conv2DTranspose(256, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal',  name="Tconvo1")(conv5)
        merge6 = concatenate([conv4, up6], axis=3)

        conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal', name="deconvo1")(merge6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal', name="deconvo2")(conv6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = Conv2DTranspose(128, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal', name="Tconvo2")(conv6)
        merge7 = concatenate([conv3, up7], axis=3)

        conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal', name="deconvo3")(merge7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal', name="deconvo4")(conv7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = Conv2DTranspose(64, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal', name="Tconvo3")(conv7)
        merge8 = concatenate([conv2, up8], axis=3)

        conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal', name="deconvo5")(merge8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal', name="deconvo6")(conv8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        conv8 = BatchNormalization()(conv8)

        up9 = Conv2DTranspose(32, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal', name="Tconvo4")(conv8)
        merge9 = concatenate([conv1, up9], axis=3)

        conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal', name="deconvo7")(merge9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal', name="deconvo8")(conv9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        conv9 = BatchNormalization()(conv9)

        conv10 = Conv2D(1, 3, padding='same', activation='sigmoid', name="deconvo9")(conv9)
        #33
        self.model = Model(inputs=inputs, outputs=conv10)
        print("a")

    def dilatedResBlockLayerNorm(self):

        """
        Constructs the baseline segmentation model with optimal bottleneck and layer normalization
        """

        inputs = Input(self.__inputSize)
        conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = LayerNormalization()(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)
        conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = LayerNormalization()(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 128

        conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = LayerNormalization()(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)
        conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = LayerNormalization()(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 64

        conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = LayerNormalization()(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)
        conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = LayerNormalization()(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # 32

        conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = LayerNormalization()(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)
        conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = LayerNormalization()(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)

        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  # 16

        # Resblock

        conv5_1 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(pool4)
        conv5_1 = BatchNormalization()(conv5_1)
        conv5_1 = LeakyReLU(alpha=.1)(conv5_1)

        conv5_2 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(conv5_1)
        conv5_2 = BatchNormalization()(conv5_2)
        conv5_2 = LeakyReLU(alpha=.1)(conv5_2)

        shortCut = Conv2D(512, 1, padding='same', kernel_initializer='he_normal')(conv5_1)

        conv5 = Add()([shortCut, conv5_2])
        conv5 = BatchNormalization()(conv5)
        conv5 = LeakyReLU(alpha=.1)(conv5)

        # resBlock End
        up6 = Conv2DTranspose(256, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv5)
        merge6 = concatenate([conv4, up6], axis=3)

        conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        conv6 = LayerNormalization()(conv6)
        conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        conv6 = LayerNormalization()(conv6)

        up7 = Conv2DTranspose(128, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv6)
        merge7 = concatenate([conv3, up7], axis=3)

        conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        conv7 = LayerNormalization()(conv7)
        conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        conv7 = LayerNormalization()(conv7)

        up8 = Conv2DTranspose(64, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv7)
        merge8 = concatenate([conv2, up8], axis=3)

        conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        conv8 = LayerNormalization()(conv8)
        conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        conv8 = LayerNormalization()(conv8)

        up9 = Conv2DTranspose(32, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv8)
        merge9 = concatenate([conv1, up9], axis=3)

        conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        conv9 = LayerNormalization()(conv9)
        conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        conv9 = LayerNormalization()(conv9)

        conv10 = Conv2D(1, 3, padding='same', activation='sigmoid')(conv9)

        self.model = Model(inputs=inputs, outputs=conv10)

    def dilatedResBlockWeightNorm(self):

        """
        Constructs the baseline segmentation model with optimal bottleneck and weight normalization
        """

        inputs = Input(self.__inputSize)
        conv1 = tfa.layers.WeightNormalization(Conv2D(32, 5, padding='same', kernel_initializer='he_normal'))(inputs)
        #conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)
        conv1 = tfa.layers.WeightNormalization(Conv2D(32, 5, padding='same', kernel_initializer='he_normal'))(conv1)
        #conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 128

        conv2 = tfa.layers.WeightNormalization(Conv2D(64, 5, padding='same', kernel_initializer='he_normal'))(pool1)
        #conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)
        conv2 = tfa.layers.WeightNormalization(Conv2D(64, 5, padding='same', kernel_initializer='he_normal'))(conv2)
        #conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 64

        conv3 = tfa.layers.WeightNormalization(Conv2D(128, 5, padding='same', kernel_initializer='he_normal'))(pool2)
        #conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)
        conv3 = tfa.layers.WeightNormalization(Conv2D(128, 5, padding='same', kernel_initializer='he_normal'))(conv3)
        #conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # 32

        conv4 = tfa.layers.WeightNormalization(Conv2D(256, 5, padding='same', kernel_initializer='he_normal'))(pool3)
        #conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)
        conv4 = tfa.layers.WeightNormalization(Conv2D(256, 5, padding='same', kernel_initializer='he_normal'))(conv4)
        #conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)

        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  # 16

        # Resblock

        conv5_1 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(pool4)
        conv5_1 = BatchNormalization()(conv5_1)
        conv5_1 = LeakyReLU(alpha=.1)(conv5_1)

        conv5_2 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(conv5_1)
        conv5_2 = BatchNormalization()(conv5_2)
        conv5_2 = LeakyReLU(alpha=.1)(conv5_2)

        shortCut = Conv2D(512, 1, padding='same', kernel_initializer='he_normal')(conv5_1)

        conv5 = Add()([shortCut, conv5_2])
        conv5 = BatchNormalization()(conv5)
        conv5 = LeakyReLU(alpha=.1)(conv5)

        # resBlock End
        up6 = Conv2DTranspose(256, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv5)
        merge6 = concatenate([conv4, up6], axis=3)

        conv6 = tfa.layers.WeightNormalization(Conv2D(256, 5, padding='same', kernel_initializer='he_normal'))(merge6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        #conv6 = BatchNormalization()(conv6)
        conv6 = tfa.layers.WeightNormalization(Conv2D(256, 5, padding='same', kernel_initializer='he_normal'))(conv6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        #conv6 = BatchNormalization()(conv6)

        up7 = Conv2DTranspose(128, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv6)
        merge7 = concatenate([conv3, up7], axis=3)

        conv7 = tfa.layers.WeightNormalization(Conv2D(128, 5, padding='same', kernel_initializer='he_normal'))(merge7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        #conv7 = BatchNormalization()(conv7)
        conv7 = tfa.layers.WeightNormalization(Conv2D(128, 5, padding='same', kernel_initializer='he_normal'))(conv7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        #conv7 = BatchNormalization()(conv7)

        up8 = Conv2DTranspose(64, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv7)
        merge8 = concatenate([conv2, up8], axis=3)

        conv8 = tfa.layers.WeightNormalization(Conv2D(64, 5, padding='same', kernel_initializer='he_normal'))(merge8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        #conv8 = BatchNormalization()(conv8)
        conv8 = tfa.layers.WeightNormalization(Conv2D(64, 5, padding='same', kernel_initializer='he_normal'))(conv8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        #conv8 = BatchNormalization()(conv8)

        up9 = Conv2DTranspose(32, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv8)
        merge9 = concatenate([conv1, up9], axis=3)

        conv9 = tfa.layers.WeightNormalization(Conv2D(32, 5, padding='same', kernel_initializer='he_normal'))(merge9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        #conv9 = BatchNormalization()(conv9)
        conv9 = tfa.layers.WeightNormalization(Conv2D(32, 5, padding='same', kernel_initializer='he_normal'))(conv9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        #conv9 = BatchNormalization()(conv9)

        conv10 = tfa.layers.WeightNormalization(Conv2D(1, 3, padding='same', activation='sigmoid'))(conv9)

        self.model = Model(inputs=inputs, outputs=conv10)

    def dilatedResBlockInstanceNorm(self):

        """
        Constructs the baseline segmentation model with optimal bottleneck and instance normalization
        """

        inputs = Input(self.__inputSize)
        conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = tfa.layers.InstanceNormalization()(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)
        conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = tfa.layers.InstanceNormalization()(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 128

        conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = tfa.layers.InstanceNormalization()(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)
        conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = tfa.layers.InstanceNormalization()(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 64

        conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = tfa.layers.InstanceNormalization()(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)
        conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = tfa.layers.InstanceNormalization()(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # 32

        conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = tfa.layers.InstanceNormalization()(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)
        conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = tfa.layers.InstanceNormalization()(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)

        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  # 16

        # Resblock

        conv5_1 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(pool4)
        conv5_1 = BatchNormalization()(conv5_1)
        conv5_1 = LeakyReLU(alpha=.1)(conv5_1)

        conv5_2 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(conv5_1)
        conv5_2 = BatchNormalization()(conv5_2)
        conv5_2 = LeakyReLU(alpha=.1)(conv5_2)

        shortCut = Conv2D(512, 1, padding='same', kernel_initializer='he_normal')(conv5_1)

        conv5 = Add()([shortCut, conv5_2])
        conv5 = BatchNormalization()(conv5)
        conv5 = LeakyReLU(alpha=.1)(conv5)

        # resBlock End
        up6 = Conv2DTranspose(256, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv5)
        merge6 = concatenate([conv4, up6], axis=3)

        conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        conv6 = tfa.layers.InstanceNormalization()(conv6)
        conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        conv6 = tfa.layers.InstanceNormalization()(conv6)

        up7 = Conv2DTranspose(128, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv6)
        merge7 = concatenate([conv3, up7], axis=3)

        conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        conv7 = tfa.layers.InstanceNormalization()(conv7)
        conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        conv7 = tfa.layers.InstanceNormalization()(conv7)

        up8 = Conv2DTranspose(64, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv7)
        merge8 = concatenate([conv2, up8], axis=3)

        conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        conv8 = tfa.layers.InstanceNormalization()(conv8)
        conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        conv8 = tfa.layers.InstanceNormalization()(conv8)

        up9 = Conv2DTranspose(32, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv8)
        merge9 = concatenate([conv1, up9], axis=3)

        conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        conv9 = tfa.layers.InstanceNormalization()(conv9)
        conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        conv9 = tfa.layers.InstanceNormalization()(conv9)

        conv10 = Conv2D(1, 3, padding='same', activation='sigmoid')(conv9)

        self.model = Model(inputs=inputs, outputs=conv10)

    def dilatedResBlockGroupNorm(self):

        """
        Constructs the baseline segmentation model with optimal bottleneck and Group normalization
        """
        inputs = Input(self.__inputSize)
        conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = tfa.layers.GroupNormalization()(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)
        conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = tfa.layers.GroupNormalization()(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 128

        conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = tfa.layers.GroupNormalization()(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)
        conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = tfa.layers.GroupNormalization()(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 64

        conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = tfa.layers.GroupNormalization()(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)
        conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = tfa.layers.GroupNormalization()(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # 32

        conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = tfa.layers.GroupNormalization()(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)
        conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = tfa.layers.GroupNormalization()(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)

        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  # 16

        # Resblock

        conv5_1 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(pool4)
        conv5_1 = BatchNormalization()(conv5_1)
        conv5_1 = LeakyReLU(alpha=.1)(conv5_1)

        conv5_2 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(conv5_1)
        conv5_2 = BatchNormalization()(conv5_2)
        conv5_2 = LeakyReLU(alpha=.1)(conv5_2)

        shortCut = Conv2D(512, 1, padding='same', kernel_initializer='he_normal')(conv5_1)

        conv5 = Add()([shortCut, conv5_2])
        conv5 = BatchNormalization()(conv5)
        conv5 = LeakyReLU(alpha=.1)(conv5)

        # resBlock End
        up6 = Conv2DTranspose(256, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv5)
        merge6 = concatenate([conv4, up6], axis=3)

        conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        conv6 = tfa.layers.GroupNormalization()(conv6)
        conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        conv6 = tfa.layers.GroupNormalization()(conv6)

        up7 = Conv2DTranspose(128, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv6)
        merge7 = concatenate([conv3, up7], axis=3)

        conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        conv7 = tfa.layers.GroupNormalization()(conv7)
        conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        conv7 = tfa.layers.GroupNormalization()(conv7)

        up8 = Conv2DTranspose(64, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv7)
        merge8 = concatenate([conv2, up8], axis=3)

        conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        conv8 = tfa.layers.GroupNormalization()(conv8)
        conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        conv8 = tfa.layers.GroupNormalization()(conv8)

        up9 = Conv2DTranspose(32, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv8)
        merge9 = concatenate([conv1, up9], axis=3)

        conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        conv9 = tfa.layers.GroupNormalization()(conv9)
        conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        conv9 = tfa.layers.GroupNormalization()(conv9)

        conv10 = Conv2D(1, 3, padding='same', activation='sigmoid')(conv9)

        self.model = Model(inputs=inputs, outputs=conv10)



    def dilatedResBlockWithDropout(self):

        """
        Constructs the baseline segmentation model with optimal bottleneck and Dropout layer instead of Batch normalization
        """

        inputs = Input(self.__inputSize)
        conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Dropout(rate=self.__dropoutRate)(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)
        conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = Dropout(rate=self.__dropoutRate)(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 128

        conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Dropout(rate=self.__dropoutRate)(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)
        conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = Dropout(rate=self.__dropoutRate)(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 64

        conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Dropout(rate=self.__dropoutRate)(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)
        conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = Dropout(rate=self.__dropoutRate)(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # 32

        conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Dropout(rate=self.__dropoutRate)(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)
        conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = Dropout(rate=self.__dropoutRate)(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)

        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  # 16

        # Resblock

        conv5_1 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(pool4)
        conv5_1 = BatchNormalization()(conv5_1)
        conv5_1 = LeakyReLU(alpha=.1)(conv5_1)

        conv5_2 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(conv5_1)
        conv5_2 = BatchNormalization()(conv5_2)
        conv5_2 = LeakyReLU(alpha=.1)(conv5_2)

        shortCut = Conv2D(512, 1, padding='same', kernel_initializer='he_normal')(conv5_1)

        conv5 = Add()([shortCut, conv5_2])
        conv5 = BatchNormalization()(conv5)
        conv5 = LeakyReLU(alpha=.1)(conv5)

        # resBlock End
        up6 = Conv2DTranspose(256, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv5)
        merge6 = concatenate([conv4, up6], axis=3)

        conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        conv6 = Dropout(rate=self.__dropoutRate)(conv6)
        conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        conv6 = Dropout(rate=self.__dropoutRate)(conv6)

        up7 = Conv2DTranspose(128, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv6)
        merge7 = concatenate([conv3, up7], axis=3)

        conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        conv7 = Dropout(rate=self.__dropoutRate)(conv7)
        conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        conv7 = Dropout(rate=self.__dropoutRate)(conv7)

        up8 = Conv2DTranspose(64, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv7)
        merge8 = concatenate([conv2, up8], axis=3)

        conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        conv8 = Dropout(rate=self.__dropoutRate)(conv8)
        conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        conv8 = Dropout(rate=self.__dropoutRate)(conv8)

        up9 = Conv2DTranspose(32, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv8)
        merge9 = concatenate([conv1, up9], axis=3)

        conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        conv9 = Dropout(rate=self.__dropoutRate)(conv9)
        conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        conv9 = Dropout(rate=self.__dropoutRate)(conv9)

        conv10 = Conv2D(1, 3, padding='same', activation='sigmoid')(conv9)

        self.model = Model(inputs=inputs, outputs=conv10)


    def dilatedResBlockWithBNandDropout(self):
        """
        Constructs the baseline segmentation model with optimal bottleneck and batch normalization followed by a dropout
        """
        inputs = Input(self.__inputSize)
        conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)
        conv1 = Dropout(rate=self.__dropoutRate)(conv1)
        conv1 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=.1)(conv1)
        conv1 = Dropout(rate=self.__dropoutRate)(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 128

        conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)
        conv2 = Dropout(rate=self.__dropoutRate)(conv2)
        conv2 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=.1)(conv2)
        conv2 = Dropout(rate=self.__dropoutRate)(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 64

        conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)
        conv3 = Dropout(rate=self.__dropoutRate)(conv3)
        conv3 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=.1)(conv3)
        conv3 = Dropout(rate=self.__dropoutRate)(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # 32

        conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)
        conv4 = Dropout(rate=self.__dropoutRate)(conv4)
        conv4 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=.1)(conv4)
        conv4 = Dropout(rate=self.__dropoutRate)(conv4)

        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  # 16

        # Resblock

        conv5_1 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(pool4)
        conv5_1 = BatchNormalization()(conv5_1)
        conv5_1 = LeakyReLU(alpha=.1)(conv5_1)

        conv5_2 = Conv2D(512, 5, padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(conv5_1)
        conv5_2 = BatchNormalization()(conv5_2)
        conv5_2 = LeakyReLU(alpha=.1)(conv5_2)

        shortCut = Conv2D(512, 1, padding='same', kernel_initializer='he_normal')(conv5_1)

        conv5 = Add()([shortCut, conv5_2])
        conv5 = BatchNormalization()(conv5)
        conv5 = LeakyReLU(alpha=.1)(conv5)

        # resBlock End
        up6 = Conv2DTranspose(256, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv5)
        merge6 = concatenate([conv4, up6], axis=3)

        conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Dropout(rate=self.__dropoutRate)(conv6)
        conv6 = Conv2D(256, 5, padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = LeakyReLU(alpha=.1)(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Dropout(rate=self.__dropoutRate)(conv6)

        up7 = Conv2DTranspose(128, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv6)
        merge7 = concatenate([conv3, up7], axis=3)

        conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Dropout(rate=self.__dropoutRate)(conv7)

        conv7 = Conv2D(128, 5, padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = LeakyReLU(alpha=.1)(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Dropout(rate=self.__dropoutRate)(conv7)

        up8 = Conv2DTranspose(64, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv7)
        merge8 = concatenate([conv2, up8], axis=3)

        conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Dropout(rate=self.__dropoutRate)(conv8)

        conv8 = Conv2D(64, 5, padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = LeakyReLU(alpha=.1)(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Dropout(rate=self.__dropoutRate)(conv8)

        up9 = Conv2DTranspose(32, 5, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv8)
        merge9 = concatenate([conv1, up9], axis=3)

        conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Dropout(rate=self.__dropoutRate)(conv9)

        conv9 = Conv2D(32, 5, padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = LeakyReLU(alpha=.1)(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Dropout(rate=self.__dropoutRate)(conv9)

        conv10 = Conv2D(1, 3, padding='same', activation='sigmoid')(conv9)

        self.model = Model(inputs=inputs, outputs=conv10)

    def basic_block(self, input_tensor, filters_size, kernel_size=5):

        x = Conv2D(filters_size, (kernel_size, kernel_size),
                   kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=.1)(x)
        x = Dropout(0.1)(x)
        x = Conv2D(filters_size, (kernel_size, kernel_size),
                   kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=.1)(x)
        x = Dropout(0.1)(x)

        return x

    def unetPlusPlus(self):

        """
        Constructs the Nested U-Net model as suggested by Zhou et al. in paper
        "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"
        """

        filters_size = [32, 64, 128, 256, 512]

        inputs = Input(self.__inputSize)

        conv1_1 = self.basic_block(inputs, filters_size=filters_size[0])
        pool1 = MaxPooling2D((2, 2))(conv1_1)

        conv2_1 = self.basic_block(pool1, filters_size=filters_size[1])
        pool2 = MaxPooling2D((2, 2))(conv2_1)

        up1_2 = Conv2DTranspose(filters_size[0], 5, strides=(2, 2), padding='same')(conv2_1)
        conv1_2 = concatenate([up1_2, conv1_1], axis=3)
        conv1_2 = self.basic_block(conv1_2, filters_size=filters_size[0])

        conv3_1 = self.basic_block(pool2, filters_size=filters_size[2])
        pool3 = MaxPooling2D((2, 2))(conv3_1)

        up2_2 = Conv2DTranspose(filters_size[1], 5, strides=(2, 2), padding='same')(conv3_1)
        conv2_2 = concatenate([up2_2, conv2_1], axis=3)
        conv2_2 = self.basic_block(conv2_2, filters_size=filters_size[1])

        up1_3 = Conv2DTranspose(filters_size[0], 5, strides=(2, 2), padding='same')(conv2_2)
        conv1_3 = concatenate([up1_3, conv1_1, conv1_2], axis=3)
        conv1_3 = self.basic_block(conv1_3, filters_size=filters_size[0])

        conv4_1 = self.basic_block(pool3, filters_size=filters_size[3])
        pool4 = MaxPooling2D((2, 2))(conv4_1)

        up3_2 = Conv2DTranspose(filters_size[2], 5, strides=(2, 2), padding='same')(conv4_1)
        conv3_2 = concatenate([up3_2, conv3_1], axis=3)
        conv3_2 = self.basic_block(conv3_2, filters_size=filters_size[2])

        up2_3 = Conv2DTranspose(filters_size[1],5, strides=(2, 2), padding='same')(conv3_2)
        conv2_3 = concatenate([up2_3, conv2_1, conv2_2], axis=3)
        conv2_3 = self.basic_block(conv2_3, filters_size=filters_size[1])

        up1_4 = Conv2DTranspose(filters_size[0],5, strides=(2, 2), padding='same')(conv2_3)
        conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], axis=3)
        conv1_4 = self.basic_block(conv1_4, filters_size=filters_size[0])

        conv5_1 = self.basic_block(pool4, filters_size=filters_size[4])

        up4_2 = Conv2DTranspose(filters_size[3], 5, strides=(2, 2), padding='same')(conv5_1)
        conv4_2 = concatenate([up4_2, conv4_1], axis=3)
        conv4_2 = self.basic_block(conv4_2, filters_size=filters_size[3])

        up3_3 = Conv2DTranspose(filters_size[2], 5, strides=(2, 2), padding='same')(conv4_2)
        conv3_3 = concatenate([up3_3, conv3_1, conv3_2], axis=3)
        conv3_3 = self.basic_block(conv3_3, filters_size=filters_size[2])

        up2_4 = Conv2DTranspose(filters_size[1], 5, strides=(2, 2), padding='same')(conv3_3)
        conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], axis=3)
        conv2_4 = self.basic_block(conv2_4, filters_size=filters_size[1])

        up1_5 = Conv2DTranspose(filters_size[0], 5, strides=(2, 2), padding='same')(conv2_4)
        conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], axis=3)
        conv1_5 = self.basic_block(conv1_5, filters_size=filters_size[0])

        output = Conv2D(1, (1, 1), activation='sigmoid', name='output_4',
                                  kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)


        self.model = Model(inputs=inputs, outputs=output)

    def FCN(self):

        """
        Constructs the FCNN model as suggested in the report
        """



        inputs = Input(self.__inputSize)

        # Block 1
        x = Conv2D(64, 5, activation='relu', padding='same')(inputs)
        x = Conv2D(64, 5, activation='relu', padding='same')(x)
        x1 = MaxPooling2D((2, 2), strides=(2, 2))(x)

        # Block 2
        x = Conv2D(128, 5, activation='relu', padding='same')(x1)
        x = Conv2D(128, 5, activation='relu', padding='same')(x)
        x2 = MaxPooling2D((2, 2))(x)

        # Block 3
        x = Conv2D(128, 5, activation='relu', padding='same')(x2)
        x = Conv2D(128, 5, activation='relu', padding='same')(x)
        x = Conv2D(256, 5, activation='relu', padding='same')(x)
        x3 = MaxPooling2D((2, 2))(x)

        # Block 4
        x = Conv2D(256, 5, activation='relu', padding='same')(x3)
        x = Conv2D(256, 5, activation='relu', padding='same')(x)
        x = Conv2D(512, 5, activation='relu', padding='same')(x)
        x4 = MaxPooling2D((2, 2))(x)

        # Block 5
        x = Conv2D(512, 5, activation='relu', padding='same')(x4)
        x = Conv2D(512, 5, activation='relu', padding='same')(x)
        x = Conv2D(256, 5, activation='relu', padding='same')(x)
        x5 = MaxPooling2D((2, 2), strides=(2, 2))(x)

        o = x5
        o = Conv2D(256, 7, activation='relu', padding='same')(o)
        o = Dropout(0.5)(o)
        o = Conv2D(256, 1, activation='relu', padding='same')(o)
        o = Dropout(0.5)(o)

        o = Conv2D(2, 1, activation='relu',padding='same')(o)
        o = Conv2DTranspose(2, kernel_size=5, strides=2, use_bias=False, padding='same')(
            o)
        #o = Cropping2D(((1, 1), (1, 1)))(o)

        o2 = x4
        o2 = Conv2D(2, 1, activation='relu', padding='same')(o2)

        o = Add()([o, o2])
        o = Conv2DTranspose(2, kernel_size=5, strides=2, use_bias=False, padding='same')(o)
        #o = Cropping2D(((1, 1), (1, 1)))(o)

        o3 = x3
        o3 = Conv2D(2, (1, 1), activation='relu',padding='same')(o3)

        o = Add()([o3, o])
        o = Conv2DTranspose(2, kernel_size=16, strides=8, use_bias=False, padding='same')(o)
        #o = Cropping2D(((4, 4), (4, 4)))(o)
        o = Conv2D(1, 1, padding='same', activation='sigmoid')(o)

        self.model = Model(inputs=inputs, outputs=o)




    @property
    def trainGenerator(self):
        return self.__trainGenerator

    @trainGenerator.setter
    def trainGenerator(self, trainGenerator):
        self.__trainGenerator = trainGenerator

    @property
    def valGenerator(self):
        return self.__valGenerator

    @valGenerator.setter
    def valGenerator(self, valGenerator):
        self.__valGenerator = valGenerator

    @property
    def optimizer(self):
        return self.__optimizer
    @optimizer.setter
    def optimizer(self, optimizer):
        self.__optimizer = optimizer
        self.modelCompile()

    @property
    def loss(self):
        return self.__loss
    @loss.setter
    def loss(self, loss):
        self.__loss = loss
        self.modelCompile()

    @property
    def metrics(self):
        return self.__metrics

    @metrics.setter
    def metrics(self, metrics):
        self.__metrics = metrics
        self.modelCompile()

    @property
    def unetType(self):
        return self.__unetType

    @unetType.setter
    def unetType(self, unetType):
        self.__unetType = unetType
        self.buildModel()
        self.modelCompile()

    @property
    def inputSize(self):
        return self.__inputSize

    @inputSize.setter
    def inputSize(self, inputSize):
        self.__inputSize = inputSize
        self.buildModel()
        self.modelCompile()

    @property
    def epochs(self):
        return self.__epochs

    @epochs.setter
    def epochs(self, epochs):
        self.__epochs = epochs

    @property
    def stepsPerEpoch(self):
        return self.__stepsPerEpoch

    @stepsPerEpoch.setter
    def stepsPerEpoch(self, stepsPerEpoch):
        self.__stepsPerEpoch = stepsPerEpoch

    @property
    def validationSteps(self):
        return self.__validationSteps

    @validationSteps.setter
    def validationSteps(self, vaidationSteps):
        self.__validationSteps = vaidationSteps

    @property
    def pretrainedWeights(self):
        return self.__pretrainedWeights

    @pretrainedWeights.setter
    def pretrinedWeights(self, pretrainedWeights):
        self.__pretrainedWeights = pretrainedWeights
        self.model.load_weights(self.__pretrainedWeights)



