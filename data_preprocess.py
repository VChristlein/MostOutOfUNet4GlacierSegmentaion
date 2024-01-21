"""
Author: Maniraman Periyasamy

This module generates the train, validation and test dataset.

The raw input images are first splited into train, validation and test dataset and then pre-processed using bilateral and CLAHE filter.
The pre-processed image is augumented eight folds using geomentric transformations and then patched into small patches in case of train and validation data.
Test data is saved as such.
"""

import numpy as np
from pathlib import Path
import os
import os.path
import utilities
import time
import cv2
import pandas as pd
from scipy.ndimage import rotate




def generateData(filePath='../Dataset3', folder = 'data3_final_256'):
    """
    This function performs the following functionality
        - Splits the data into train, validation and test set
        - Pre-process the data
        - Auguments the data

    Note:
        Data Augumentation and pre-processing are hardcoded into the function. Hence, it has to manually modified if required.

    Args:
        filePath (str): Relative path to the raw input images.
        folder (str): name of the folder to be generated where all the train, test and validation dataset will be saved.

    Returns: None

    """

    # parameters for the function
    data_count = 0
    data_all = []
    data_names = []
    fileSize = {}
    resolution_dict = {}
    unique_resolution  = []
    testSize = 30
    valSize = 30
    NA_count = 0
    PATCH_SIZE = 256

    for filename in Path(filePath).rglob('*.png'):

        # decides whether to use NA class or not.
        if str(filename.resolve()).find('_zones.png') == -1:
            data = cv2.imread(str(filename),0)

            print(data.shape)

            if data.shape in fileSize.keys():
                fileSize[data.shape] = fileSize[data.shape]+1
            else:
                fileSize[data.shape] = 1
            data_all.append(data)

            data_names.append(filename.as_posix())
            data_count += 1

            # calculates the unique spatial resolutions in the image data
            if str(filename)[-10:]=="_front.png":
                if str(filename)[-14:-12] not in unique_resolution:
                    unique_resolution.append(str(filename)[-14:-12])
                if str(data.shape) in resolution_dict.keys():
                    resolution_dict[str(data.shape)].append(str(filename)[-14:-12])
                else:
                    resolution_dict[str(data.shape)] = []
                    resolution_dict[str(data.shape)].append(str(filename)[-14:-12])
        else:
            NA_count +=1

    print(data_count)
    resolution_count_pd = pd.DataFrame(columns=unique_resolution, index=list(resolution_dict.keys()))
    for col in resolution_count_pd.columns:
        resolution_count_pd[col].values[:]=0

    for key in resolution_dict.keys():
        for elem in resolution_dict[key]:
            resolution_count_pd.loc[key][elem] += 1

    # to sort all images in the list according to alphabet order
    resolution_count_pd.to_csv("resolution_count.csv")
    data_names, data_all = (list(t) for t in zip(*sorted(zip(data_names, data_all))))

    images_names, images_names_id = [], []
    masks_line_names, masks_line_names_id = [], []
    masks_zone_names, masks_zone_names_id = [], []
    for i in range(len(data_names)):
        if "_front.png" in data_names[i]:
            masks_line_names.append(data_names[i])
            masks_line_names_id.append(i)
        elif "_zones_NA.png" in data_names[i]:
            masks_zone_names.append(data_names[i])
            masks_zone_names_id.append(i)
        else:
            images_names.append(data_names[i])
            images_names_id.append(i)


    # splits the data into train test and validation set
    from sklearn.model_selection import train_test_split

    data_idx = np.arange(len(images_names_id))
    train_idx, test_idx = train_test_split(data_idx, test_size=testSize,
                                           random_state=1)  # 30 images are chosen as the test images
    train_idx, val_idx = train_test_split(train_idx, test_size=valSize,
                                          random_state=1)  # 30 of training data as validation data


    # writes the list of images in each set into csv files for future reference
    pd.DataFrame({"images":np.array(images_names)[train_idx], "masks":np.array(masks_zone_names)[train_idx],
                  "lines": np.array(masks_line_names)[train_idx]}).to_csv("train_images.csv")
    pd.DataFrame({"images": np.array(images_names)[val_idx], "masks": np.array(masks_zone_names)[val_idx],
                  "lines": np.array(masks_line_names)[val_idx]}).to_csv("validation_images.csv")
    pd.DataFrame({"images": np.array(images_names)[test_idx], "masks": np.array(masks_zone_names)[test_idx],
                  "lines": np.array(masks_line_names)[test_idx]}).to_csv("test_images.csv")

    START = time.time()


    # train path
    if not os.path.exists(str(Path(folder + '/train/images'))): os.makedirs(
        str(Path(folder+ '/train/images')))
    if not os.path.exists(str(Path(folder + '/train/masks_zones'))): os.makedirs(
        str(Path(folder+ '/train/masks_zones')))
    if not os.path.exists(str(Path(folder + '/train/masks_lines'))): os.makedirs(
        str(Path(folder+ '/train/masks_lines')))

    STRIDE_train = (PATCH_SIZE, PATCH_SIZE)
    patch_counter_train = 0
    tot = len(train_idx)
    count = 0
    for i in train_idx:
        masks_zone_tmp = data_all[masks_zone_names_id[i]]
        masks_zone_tmp[masks_zone_tmp <= 127] = 0
        masks_zone_tmp[masks_zone_tmp > 127] = 255


        # Pre-processing
        img_bilateral = cv2.bilateralFilter(data_all[images_names_id[i]], 10, 95, 75)  # bilateral filter
        img_CLAHE = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(5, 5)).apply(
            img_bilateral)  # CLAHE adaptive contrast enhancement

        # Zero-padding in case of mis-match in patch size.


        img_CLAHE = cv2.copyMakeBorder(img_CLAHE, 0, (PATCH_SIZE - img_CLAHE.shape[0]) % PATCH_SIZE, 0,
                                   (PATCH_SIZE - img_CLAHE.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT)

        masks_zone_tmp = cv2.copyMakeBorder(masks_zone_tmp, 0, (PATCH_SIZE - masks_zone_tmp.shape[0]) % PATCH_SIZE, 0,
                                  (PATCH_SIZE - masks_zone_tmp.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT)

        maskLine = cv2.copyMakeBorder(data_all[masks_line_names_id[i]], 0,
                                  (PATCH_SIZE - data_all[masks_line_names_id[i]].shape[0]) % PATCH_SIZE, 0,
                                  (PATCH_SIZE - data_all[masks_line_names_id[i]].shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT)

        # Data Augumentation
        for rot in [0,90,180,270]:


            image = rotate(img_CLAHE, rot, reshape=True)
            mask = rotate(masks_zone_tmp, rot, reshape=True)
            line = rotate(maskLine, rot, reshape=True)


            for flag in range(1):
                if flag == 1:
                    image = np.fliplr(image)
                    mask = np.fliplr(mask)
                    line = np.fliplr(line)



                p_masks_zone, i_masks_zone = utilities.extract_grayscale_patches(mask, (PATCH_SIZE, PATCH_SIZE),
                                                                                 stride=STRIDE_train)
                p_img, i_img = utilities.extract_grayscale_patches(image, (PATCH_SIZE, PATCH_SIZE),
                                                                   stride=STRIDE_train)
                p_masks_line, i_masks_line = utilities.extract_grayscale_patches(line,
                                                                                 (PATCH_SIZE, PATCH_SIZE),
                                                                                 stride=STRIDE_train)

                for j in range(p_masks_zone.shape[0]):
                    if np.count_nonzero(p_masks_zone[j]) / (PATCH_SIZE * PATCH_SIZE) >= 0 and np.count_nonzero(
                            p_masks_zone[j]) / (PATCH_SIZE * PATCH_SIZE) <= 1:
                        cv2.imwrite(str(
                            Path(folder+ '/train/images/' + str(patch_counter_train) + '.png')),
                            p_img[j])
                        cv2.imwrite(str(Path(folder+ '/train/masks_zones/' + str(
                               patch_counter_train) + '.png')),
                           p_masks_zone[j])
                        cv2.imwrite(
                           str(Path(folder+ '/train/masks_lines/' + str(
                               patch_counter_train) + '.png')),
                           p_masks_line[j])
                        patch_counter_train += 1

        count = count +1
        print("completed {} of {}".format(count,tot))




    #####
    # validation path
    if not os.path.exists(str(Path(folder+ '/val/images'))): os.makedirs(
        str(Path(folder+ '/val/images')))
    if not os.path.exists(str(Path(folder+ '/val/masks_zones'))): os.makedirs(
        str(Path(folder+ '/val/masks_zones')))
    if not os.path.exists(str(Path(folder+ '/val/masks_lines'))): os.makedirs(
        str(Path(folder+ '/val/masks_lines')))

    STRIDE_val = (PATCH_SIZE, PATCH_SIZE)
    patch_counter_val = 0
    tot = len(val_idx)
    count = 0
    for i in val_idx:
        masks_zone_tmp = data_all[masks_zone_names_id[i]]
        masks_zone_tmp[masks_zone_tmp <= 127] = 0
        masks_zone_tmp[masks_zone_tmp > 127] = 255


        # Data Pre-processing

        img_bilateral = cv2.bilateralFilter(data_all[images_names_id[i]], 10, 95, 75)  # bilateral filter
        img_CLAHE = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(5, 5)).apply(
            img_bilateral)  # CLAHE adaptive contrast enhancement


        # Zero padding
        img_CLAHE = cv2.copyMakeBorder(img_CLAHE, 0, (PATCH_SIZE - img_CLAHE.shape[0]) % PATCH_SIZE, 0,
                                       (PATCH_SIZE - img_CLAHE.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT)
        masks_zone_tmp = cv2.copyMakeBorder(masks_zone_tmp, 0, (PATCH_SIZE - masks_zone_tmp.shape[0]) % PATCH_SIZE, 0,
                                            (PATCH_SIZE - masks_zone_tmp.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT)
        maskLine = cv2.copyMakeBorder(data_all[masks_line_names_id[i]], 0,
                                      (PATCH_SIZE - data_all[masks_line_names_id[i]].shape[0]) % PATCH_SIZE, 0,
                                      (PATCH_SIZE - data_all[masks_line_names_id[i]].shape[1]) % PATCH_SIZE,
                                      cv2.BORDER_CONSTANT)


        # Data Augumentation
        for rot in [0,90,180,270]:

            image = rotate(img_CLAHE, rot, reshape=True)
            mask = rotate(masks_zone_tmp, rot, reshape=True)
            line = rotate(maskLine, rot, reshape=True)

            for flag in range(1):
                if flag == 1:
                    image = np.fliplr(image)
                    mask = np.fliplr(mask)
                    line = np.fliplr(line)

                p_masks_zone, i_masks_zone = utilities.extract_grayscale_patches(mask, (PATCH_SIZE, PATCH_SIZE),
                                                                                 stride=STRIDE_train)
                p_img, i_img = utilities.extract_grayscale_patches(image, (PATCH_SIZE, PATCH_SIZE),
                                                                   stride=STRIDE_train)
                p_masks_line, i_masks_line = utilities.extract_grayscale_patches(line,
                                                                                 (PATCH_SIZE, PATCH_SIZE),
                                                                                 stride=STRIDE_train)

                if patch_counter_val == 1649:
                    pass
                for j in range(p_masks_zone.shape[0]):
                    if np.count_nonzero(p_masks_zone[j]) / (PATCH_SIZE * PATCH_SIZE) > 0 and np.count_nonzero(
                            p_masks_zone[j]) / (PATCH_SIZE * PATCH_SIZE) < 1:
                        cv2.imwrite(str(Path(folder+ '/val/images/' + str(patch_counter_val) + '.png')),
                                    p_img[j])
                        cv2.imwrite(
                            str(Path(folder+ '/val/masks_zones/' + str(patch_counter_val) + '.png')),
                            p_masks_zone[j])
                        cv2.imwrite(
                            str(Path(folder+ '/val/masks_lines/' + str(patch_counter_val) + '.png')),
                            p_masks_line[j])
                        patch_counter_val += 1
                        # store the name of the file that the patch is from in a list as well

        count = count + 1
        print("completed {} of {}".format(count, tot))



    #####
    # test path
    if not os.path.exists(str(Path(folder+ '/test/images'))): os.makedirs(
        str(Path(folder+ '/test/images')))
    if not os.path.exists(str(Path(folder+ '/test/masks_zones'))): os.makedirs(
        str(Path(folder+ '/test/masks_zones')))
    if not os.path.exists(str(Path(folder+ '/test/masks_lines'))): os.makedirs(
        str(Path(folder+ '/test/masks_lines')))

    STRIDE_test = (PATCH_SIZE, PATCH_SIZE)

    tot = len(test_idx)
    count = 0

    for i in test_idx:
        masks_zone_tmp = data_all[masks_zone_names_id[i]]
        masks_zone_tmp[masks_zone_tmp <= 127] = 0
        masks_zone_tmp[masks_zone_tmp > 127] = 255

        # Data Pre-processing
        img_bilateral = cv2.bilateralFilter(data_all[images_names_id[i]], 10, 95, 75)  # bilateral filter
        img_CLAHE = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(5, 5)).apply(
            img_bilateral)  # CLAHE adaptive contrast enhancement


        cv2.imwrite(str(Path(folder+ '/test/images/' +str(count)+"_"+ Path(images_names[i]).name)),
                    img_CLAHE)
        cv2.imwrite(str(Path(folder+ '/test/masks_zones/'+str(count)+"_" + Path(masks_zone_names[i]).name)),
                    masks_zone_tmp)
        cv2.imwrite(str(Path(folder+ '/test/masks_lines/'+str(count)+"_" + Path(masks_line_names[i]).name)),
                    data_all[masks_line_names_id[i]])

        count = count + 1
        print("completed {} of {}".format(count, tot))


    END = time.time()
    print(END - START)

#generateData()