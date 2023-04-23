import matplotlib.pyplot as plt
import numpy as np
from Confusion import calc_confusion_matrix
import os

 
 
pred_list = os.listdir('')
gt_list = os.listdir('')

ROOT_PATH = r''

num_classes=5

confusion = calc_confusion_matrix(gt_list,pred_list,ROOT_PATH,num_classes)


plt.imshow(confusion, cmap=plt.cm.Reds)
 
#TODO: If you have chinese words
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False
 

plt.colorbar()        

 

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion metrix of RITE Dataset')                            
 

classes=['Background', 'White', 'Green', 'Red', 'Blue']            
indices = range(len(confusion))                
 
plt.xticks(indices, classes, rotation=45)   
plt.yticks(indices, classes)
 

for i in range(len(confusion)):
    for j in range(len(confusion)):
        plt.text(j, i, confusion[i][j], 
                 fontsize=15,
                 horizontalalignment="center",  
                 verticalalignment="center",    
                 color="white" if confusion[i, j] > confusion.max()/2. else "black") 
 

plt.show()
 
