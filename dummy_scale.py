import os
from stat import S_IRWXU, S_IRWXG

# Path data.
path = 'C:/Users/harsh/ivadomed/'
folder = 'data_example_spinegeneric_10x/'
num = 99

# Creating false data to trick ivadomed into thinking data set is bigger; copies one sample's contents for num additional samples.
for i in range(2, num + 1):
    copied_dir = path + folder + f'sub-unf01 - Copy ({i})/anat'

    t1wjson = copied_dir + f'/sub-unf0{i}_T1w.json'
    os.rename(copied_dir + '/sub-unf01_T1w.json',
              t1wjson)

    t1wniigz = copied_dir + f'/sub-unf0{i}_T1w.nii.gz'
    os.rename(copied_dir + '/sub-unf01_T1w.nii.gz',
              t1wniigz)

    t2starjson = copied_dir + f'/sub-unf0{i}_T2star.json'
    os.rename(copied_dir + '/sub-unf01_T2star.json',
              t2starjson)

    t2starniigz = copied_dir + f'/sub-unf0{i}_T2star.nii.gz'
    os.rename(copied_dir + '/sub-unf01_T2star.nii.gz',
              t2starniigz)

    t2wjson = copied_dir + f'/sub-unf0{i}_T2w.json'
    os.rename(copied_dir + '/sub-unf01_T2w.json',
              t2wjson)

    t2wniigz = copied_dir + f'/sub-unf0{i}_T2w.nii.gz'
    os.rename(copied_dir + '/sub-unf01_T2w.nii.gz',
              t2wniigz)

    copied_folder = path + folder + f'sub-unf01 - Copy ({i})'
    os.chmod(copied_folder, S_IRWXU | S_IRWXG)
    new_dir = path + folder + f'sub-unf0{i}'
    os.rename(copied_folder,
              new_dir)

    print(f'Done changing file names in {copied_folder}.')

# Creates derivatives/labels for num additional samples.
deriv = path + folder + 'derivatives/labels/'
for i in range(2, num + 1):

    copied_dir = deriv + f'sub-unf01 - Copy ({i})/anat'

    t1w_seg = copied_dir + f'/sub-unf0{i}_T1w_seg-manual.nii.gz'
    os.rename(copied_dir + '/sub-unf01_T1w_seg-manual.nii.gz',
              t1w_seg)

    t2star_seg = copied_dir + f'/sub-unf0{i}_T2star_seg-manual.nii.gz'
    os.rename(copied_dir + '/sub-unf01_T2star_seg-manual.nii.gz',
              t2star_seg)

    t2w_csfseg = copied_dir + f'/sub-unf0{i}_T2w_csfseg-manual.nii.gz'
    os.rename(copied_dir + '/sub-unf01_T2w_csfseg-manual.nii.gz',
              t2w_csfseg)

    t2w_seg = copied_dir + f'/sub-unf0{i}_T2w_seg-manual.nii.gz'
    os.rename(copied_dir + '/sub-unf01_T2w_seg-manual.nii.gz',
              t2w_seg)

    copied_folder = deriv + f'sub-unf01 - Copy ({i})'
    os.chmod(copied_folder, S_IRWXU | S_IRWXG)
    new_dir = deriv + f'sub-unf0{i}'
    os.rename(copied_folder,
              new_dir)

