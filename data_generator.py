"""
Author: Maniraman Periyasamy

This module generates the keras data generator object which is to be used for training the U-Net model
"""



from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def adjustData(img,mask,flag_multi_class,num_class):
    """
    Adjust the masks to binary or multiclass problem.

    Args:
        img (ndarray): image
        mask (ndarray): mask for the image.
        flag_multi_class (bool): Flag indicating whether the data is a binary class data or not
        num_class (int): Number of classes in the dataset

    Returns: tuple(image, mask)

    """
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    """
    Generate the keras Data Generator object for the given dataset

    Args:
        batch_size (int): Batch size
        train_path (str): Relative path to the data
        image_folder (str): Name of the folder where images are saved in the train_path
        mask_folder (str): Name of the folder where masks are saved in the train_path
        aug_dict (dict): Transformation properties
        image_color_mode (str, default is 'grayscal'): Colorspace of the image
        mask_color_mode (str, default is 'grayscal'): Colorspace of the mask
        image_save_prefix (str, default is 'image'): Prefix to be used for the image name
        mask_save_prefix (str, default is 'mask'): Prefix to be used for the mask name
        flag_multi_class (bool, default is False): Flag indicating whether the data is a binary class data or not
        num_class (int, default is 2): Number of classes in the dataset
        save_to_dir (str, default is None): Directory where the sample is to be saved for viewing
        target_size (tuple(int, int), default is (256, 256)): Required size of the images.
        seed (int): Random seed to be used.

    Returns: None

    """

    if aug_dict == None:
        image_datagen = ImageDataGenerator()
        mask_datagen = ImageDataGenerator()
    else:
        image_datagen = ImageDataGenerator(**aug_dict)
        mask_datagen = ImageDataGenerator(**aug_dict)
        
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)







