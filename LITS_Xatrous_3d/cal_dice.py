import numpy as np 
import os
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
from skimage import io

img_path = 'D:/LITS_Rst/LITS_280_lx_atrous/exp1/imgs/'
file_names = os.listdir(img_path)
img_num = len(file_names)
height = 280
width = 280

tumor_label = 0
tumor_pred = 1
tumor_labPred = 0
liver_label = 0
liver_pred = 0
liver_labPred = 0
for img_i in range(0, img_num, 2):
	label_img = io.imread(img_path+file_names[img_i])
	pred_img = io.imread(img_path+file_names[img_i+1])

	for i in range(0, height):
		for j in range(0, width):
			if (label_img[i, j, 1] == 0):
				tumor_label = tumor_label+1
			if (pred_img[i, j, 1] == 0):
				tumor_pred = tumor_pred+1
			if  (label_img[i, j, 1] == 0 and  pred_img[i, j, 1] == 0):
				tumor_labPred = tumor_labPred+1
			if (label_img[i, j, 1] == 69):
				liver_label = liver_label+1
			if (pred_img[i, j, 1] == 69):
				liver_pred = liver_pred+1
			if  (label_img[i, j, 1] == 69 and  pred_img[i, j, 1] == 69):
				liver_labPred = liver_labPred+1

tumor_dice_coe = 2*tumor_labPred/(tumor_label + tumor_pred)
liver_dice_coe = 2*liver_labPred/(liver_label + liver_pred)

print("liver_dice:", liver_dice_coe)
print("tumor_dice:", tumor_dice_coe)
with open('FCNX_atrous.txt', 'a+') as resltFile:
	resltFile.write("liver_dice:  %.3f \n" %(liver_dice_coe))
	resltFile.write("tumor_dice:  %.3f \n" %(tumor_dice_coe))
