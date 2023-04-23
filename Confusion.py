import numpy as np
import os 
from PIL import Image
from functools import partial
from collections import Counter

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


def calc_confusion_matrix(gt_list,pred_list,root,num_classes=None):
    '''
    This function calculates the confusion matrix

    Parameters:
        ge_list - list of the ground truth file name 
        pred_list - list the prediction file name
        root - root of the path to the ground truth file or prediction file
        num_classes - number of classes

    Returns:
        Returns the confusion

    '''
    
    #TODO: CREATE THE LIST OF CLASSES
    list_classes =list()
    for index in range(num_classes):
        list_classes.append(index)
    #TODO: CREATE THE CONFUSION MATRIX
    confusion = np.zeros(num_classes*num_classes,dtype=np.int64)
    #TODO: MAKING THE PICTURE TO BATCH

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
    

    pred_labels = iter(pred_batch)
    gt_labels = iter(gt_batch) 
    list_confusion =list()
    for pred_label, gt_label in zip(pred_labels, gt_labels):
        # print(len(np.array(pred_label).shape))
        
        
        if pred_label.ndim != 2 or gt_label.ndim != 2:
            raise ValueError('ndim of labels should be two.')
 
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should be same.')

        pred_label,pred_unflatten= flatten(pred_label)  
        gt_label,gt_unflatten = flatten(gt_label)

        pred_label = mapping(pred_label, list_classes)
        gt_label = mapping(gt_label,list_classes)
       
        # Dynamically expand the confusion matrix if necessary.
        lb_max = np.max((pred_label, gt_label))
        # print(lb_max)
        if lb_max >= num_classes:    #如果分类数大于预设的分类数目，则扩充一下。
            expanded_confusion = np.zeros((lb_max + 1, lb_max + 1), dtype=np.int64)
            expanded_confusion[0:num_classes, 0:num_classes] = confusion
 
            num_classes = lb_max + 1
            confusion = expanded_confusion

        pred_label = pred_unflatten(pred_label)
        gt_label = gt_unflatten(gt_label)

        # Count statistics from valid pixels

        mask = (gt_label >= 0) & (gt_label < num_classes)
       
        confusion = np.bincount(
            num_classes * gt_label[mask].astype(int) + pred_label[mask],
            minlength=num_classes ** 2).reshape((num_classes, num_classes))
        
        list_confusion.append(confusion)
 
    for iter_ in (pred_labels, gt_labels):
        # This code assumes any iterator does not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')
    
    confusion = sum(list_confusion)
        
 
    return confusion


 



    
