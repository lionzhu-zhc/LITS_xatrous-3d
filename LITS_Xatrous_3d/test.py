import  os
IMGDEPTH = 24
testPath = 'E:/Lianxin_LITS/lxData_rs_600_cut_280/test_npy/'

list1 = ['11', 10]
print(type(list1))

test_dirs = os.listdir(testPath + '/vol/')

if not(os.path.exists(testPath + '/vol/')):
    print("ol")

for tDir in test_dirs:
    name_pre = tDir[:-4]
    file_split = name_pre.split('-')
    a = int(file_split[-1]) * IMGDEPTH
    print(type(file_split))