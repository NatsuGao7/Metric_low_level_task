from Confusion import calc_confusion_matrix
import numpy as np
import os

pred_list = os.listdir('')
gt_list = os.listdir('')
name_classes =['Background', 'White', 'Green', 'Red', 'Blue']  
ROOT_PATH = r''

num_classes=5

confusion = calc_confusion_matrix(gt_list,pred_list,ROOT_PATH,num_classes)
 

def accuracy(confusion):
    '''
    Accuracy = (TP+TN)/(TP+TN+FP+FN)
    '''
    return np.sum(confusion.diagonal())/np.sum(confusion)



def precision(confusion,name_classes):

    '''
    Percision = TP/(TP+FP)

    Parameters:

    name_classes - every classes's name
    '''
    list_persion = confusion.diagonal()/np.sum(confusion,axis=1)
    list_name_persion = list()

    for index in range(len(name_classes)):
        temp = name_classes[index]+'_pc'
        list_name_persion.append(temp)
    
    dictionary_pc = dict(zip(list_name_persion,  list_persion))

    return dictionary_pc

 
def recall(confusion,name_classes):
    """
    Recall(Sensitivity) = TP/(TP+FN)

    Parameters:

    name_classes - every classes's name

    """
    list_recall = confusion.diagonal()/np.sum(confusion,axis=0)

    list_name_recall = list()

    for index in range(len(name_classes)):
        temp = name_classes[index]+'_rc'
        list_name_recall.append(temp)

    dictionary_rc = dict(zip(list_name_recall, list_recall ))

    return dictionary_rc



def specificity(confusion,name_classes):
    """
    Specificity = TN/(TN+FP)

    Parameters:

    name_classes - every classes's name

    """
    list_specificity = confusion.diagonal()/np.sum(confusion,axis=0)

    list_name_specificity = list()

    for index in range(len(name_classes)):
        temp = name_classes[index]+'_sc'
        list_name_specificity.append(temp)

    dictionary_sc = dict(zip(list_name_specificity,  list_specificity ))
    return  dictionary_sc 


def F1score(PC,RC,name_classes):
    """
    F1 score = 2*(precision*recall)/(precision+recall)

    Parameters:
    param PC: precision rate
    param RC: recall rate

    """
    list_f1 = list()
    for pc, rc in zip(PC,RC):
        temp = 2*(pc*rc)/(pc+rc)
        list_f1.append(temp)

    list_name_f1 = list()

    for index in range(len(name_classes)):
        temp = name_classes[index]+'_f1'
        list_name_f1.append(temp)
    
    dictionary_f1 = dict(zip(list_name_f1,  list_f1  ))
    
    return dictionary_f1





