import numpy as np
import os

npy_path = 'D:/IESLES_Rst/CT_128/exp4/npys/'
exp_path = 'D:/IESLES_Rst/CT_128/exp4/'
file_names = os.listdir(npy_path)
npy_num = len(file_names)

liver_label = 0
liver_pred = 0
liver_labPred = 0
label_all = []
pred_all = []
for img_i in range(0, npy_num, 2):
    label_batch = np.load(npy_path + file_names[img_i])
    pred_batch = np.load(npy_path + file_names[img_i+1])
    print(np.count_nonzero(label_batch == 2))
    print(np.count_nonzero(pred_batch == 2))
    liver_label = liver_label + np.count_nonzero(label_batch == 1)
    liver_pred = liver_pred + np.count_nonzero(pred_batch == 1)

    label_bool = (label_batch == 1)
    pred_bool = (pred_batch == 1)
    common = np.logical_and(label_bool, pred_bool)
    liver_labPred = liver_labPred + np.count_nonzero(common == True)

liver_dice_coe = 2*liver_labPred/(liver_label + liver_pred)
print("lesion_dice:", liver_dice_coe)
print("lesion_label", liver_label)
print("lesion_pred", liver_pred)
print("lesion_labPred", liver_labPred)
with open(exp_path + 'FCNX_atrous.txt', 'a+') as resltFile:
	resltFile.write("lesion_dice:  %.3f \n" %(liver_dice_coe))



# --------------another way to cal dice, not fast as above
# for img_i in range (0, npy_num, 2):
#     label_batch = np.load(npy_path + file_names[img_i])
#     pred_batch = np.load(npy_path + file_names[img_i + 1])
#
#     if img_i == 0:
#         label_all = label_batch
#         pred_all = pred_batch
#     else:
#         label_all = np.concatenate((label_all, label_batch), axis=2)
#         pred_all = np.concatenate((pred_all, pred_batch), axis=2)
#
# inse = np.sum(label_all * pred_all, axis= (0,1,2))
# ll = np.sum(label_all, axis= (0,1,2))
# rr = np.sum(pred_all, axis= (0,1,2))
# dice = (2.0*inse + 1e-5) / (ll + rr + 1e-5)
# dice = np.mean(dice)
# print(dice)
