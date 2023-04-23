import numpy as np
from PIL import Image
from os.path import join
from functools import partial
from collections import Counter
import os

#TODO: Raising the dimension
def flatten(x):
    original_shape = x.shape
    return x.flatten(), partial(np.reshape, newshape=original_shape)

def mapping(img_flatten_array,list_classes):
    '''
    This function Create links to pixels and categories
    e.g. The image contains pixel values{0,255,40,20,190}
         The image is devided into 5 classes[0,1,2,3,4]
         Then replace the pixel values in the image with the category


    Parameters:
        img_flatten_array：Convert the image to 'P' and then pull it into one dimension
        list_classes：The list of Classes from 0 to n-1 classes

    Returns:
        Image dimension after substitution with category is 1

    '''

    img_set = list(set(img_flatten_array))
    
    dictionary = dict(zip(img_set, list_classes))
    list_numpy_array = img_flatten_array.tolist()
    classes_img = list()
    for index in range(len(list_numpy_array)):
        temp = dictionary[list_numpy_array[index]]
        classes_img.append(temp)
    classes_img = np.array(classes_img)

    return classes_img

def get_dicr(gt_list,pred_list,root,num_classes):

    #TODO: CREATE THE LIST OF CLASSES
    list_classes =list()
    for index in range(num_classes):
        list_classes.append(index)

    gt_path = list()
    #TODO: get the path of ground truth file
    for index in range(len(gt_list)):
        temp = os.path.join(root,'drive_gt',gt_list[index])
        gt_path.append(temp)
    

    pred_path = list()
    #TODO: get the path of prediction file
    for index in range(len(pred_list)):
        temp = os.path.join(root,'drive_gt_test',pred_list[index])
        pred_path.append(temp)
    
    #TODO: MAKING THE PICTURE TO BATCH
    gt_batch = list()

    for index in range(len(gt_path)):
        temp = np.array(Image.open(gt_path[index]).convert('P'))
        gt_batch.append(temp)
   

    pred_batch = list()
    for index in range(len(pred_path)):
        temp = np.array(Image.open(pred_path[index]).convert('P'))
        pred_batch.append(temp)
    
    list_intersections = list()
    list_andSet = list()
    for pred_label, gt_label in zip(pred_batch, gt_batch):

        pred_label,pred_unflatten = flatten(pred_label)
        gt_label,gt_unflatten = flatten(gt_label)

        pred_label = mapping(pred_label,list_classes)
        gt_label = mapping(gt_label,list_classes)

        pred_label = pred_unflatten(pred_label)
        gt_label = gt_unflatten(gt_label)

        intersections = dict(Counter(list((gt_label-pred_label).flatten())))[0]
        list_intersections.append(intersections)
        # print(len(list(pred_label.flatten())))
        list_andSet.append(len(list(pred_label.flatten()))+len(list(gt_label.flatten())))
        

    Dice = (sum(list_intersections))/(sum(list_andSet)-sum(list_intersections))

    return Dice

