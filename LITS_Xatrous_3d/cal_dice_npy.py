import numpy as np
import os

npy_path = 'D:/LITS_Rst/LITS_280_lx_atrous/exp4/npys/'
file_names = os.listdir(npy_path)
npy_num = len(file_names)

liver_label = 0
liver_pred = 0
liver_labPred = 0
for img_i in range(0, npy_num, 2):
    label_batch = np.load(npy_path + file_names[img_i])
    pred_batch = np.load(npy_path + file_names[img_i+1])
    liver_label = liver_label + np.count_nonzero(label_batch == 1)
    liver_pred = liver_pred + np.count_nonzero(pred_batch == 1)

    label_bool = (label_batch == 1)
    pred_bool = (pred_batch == 1)
    common = np.logical_and(label_bool, pred_bool)
    liver_labPred = liver_labPred + np.count_nonzero(common == True)

liver_dice_coe = 2*liver_labPred/(liver_label + liver_pred)
print("liver_dice:", liver_dice_coe)
with open('FCNX_atrous.txt', 'a+') as resltFile:
	resltFile.write("exp4_liver_dice:  %.3f \n" %(liver_dice_coe))
