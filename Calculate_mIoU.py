import numpy as np
from PIL import Image
from os.path import join
from functools import partial
import os


def calculate_mIOU(gt_list,pred_list,root='', num_classes = int, name_classes = list()):
    '''
    This function calculates the mIOU to metric the results of the segmentation

    Parameters:
        ge_list - list of the ground truth file name 
        pred_list - list the prediction file name
        root - root of the path to the ground truth file or prediction file
        num_classes - number of classes
        name_classes - Each classes' name

    Returns:
        Returns the mIOU

    '''
    name_classes = np.array(name_classes,dtype=np.str_)
    hist = np.zeros((num_classes,num_classes)) # Confusion Matrix

    gt_path = list()
    #TODO: get the path of ground truth file
    for index in range(len(gt_list)):
        temp = join(root,'drive_gt',gt_list[index])
        gt_path.append(temp)

    pred_path = list()
    #TODO: get the path of prediction file
    for index in range(len(pred_list)):
        temp = join(root,'drive_gt_test',pred_list[index])
        pred_path.append(temp)
    
    #TODO: Create the list of classes
    list_classes =list()
    for index in range(num_classes):
        list_classes.append(index)

    list_confusion = list()

    #TODO: Calculate the every classes's IoU and mIoU
    for index in range(len(gt_path)):
        pred = np.array(Image.open(pred_path[index]).convert('P'))
       
        pred,unflatten_pred = flatten(pred)
        

        label = np.array(Image.open(gt_path[index]).convert('P'))
        label,unflatten_label = flatten(label)
       
        pred= mapping(pred, list_classes)
        label = mapping(label,list_classes)
        
        if len(label) != len(pred):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                                                                              len(pred.flatten()), gt_path[index],
                                                                              pred_path[index]))
            continue
        
        pred = unflatten_pred(pred)
        label = unflatten_label(label)

      
        confusion = fast_hist(label,pred,num_classes)
        list_confusion.append(confusion)
        

        if index > 0 and index % 10 == 0:
            print('{:d} / {:d}: {:0.2f}'.format(index, len(gt_path), 100 * np.mean(per_class_iu(sum(list_confusion)))))

    mIoUs = per_class_iu(sum(list_confusion))

    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))

    return mIoUs


def fast_hist(a, b, n):
    #print(set(a))
    k = (a >= 0) & (a < n)
    #print(k.shape)
    #print(len(np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2)))
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    
    
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

#TODO: Raising the dimension
def flatten(x):
    original_shape = x.shape
    return x.flatten(), partial(np.reshape, newshape=original_shape)



