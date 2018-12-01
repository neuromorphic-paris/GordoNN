import random
import glob
import math

def get_filenames_dataset(number_of_files = -1, train_test_ratio = 0.75):
    used_classes = ['off', 'on']
    folders = ['/home/marcorax/Desktop/off_aedats', '/home/marcorax/Desktop/on_aedats']
    
    filenames_train = []
    filenames_test = []
    class_train = []
    class_test = []

    for i in range(len(used_classes)):
        aedats_in_folder = glob.glob(folders[i] + '/*.aedat')
        print ('No. of files of class'), used_classes[i], ': ', len(aedats_in_folder)

        if number_of_files > 0:
            print ('Func:get_filenames_dataset(): Getting', number_of_files, 'files from the', used_classes[i], 'folder')
            aedats_in_folder = random.sample(aedats_in_folder, number_of_files)
        elif number_of_files > len(aedats_in_folder):
            print ('Func:get_filenames_dataset(): Error: the number of files selected is bigger than the number of .aedat file in the folder. Getting the whole dataset')

        aedats_for_training = int(math.ceil(len(aedats_in_folder)*train_test_ratio))
        aedats_for_testing = len(aedats_in_folder) - aedats_for_training

        random.shuffle(aedats_in_folder)

        for ind_train in range(aedats_for_training):
            filenames_train.append(aedats_in_folder[ind_train])
            class_train.append(i)
        for ind_test in range(aedats_for_training, len(aedats_in_folder)):
            filenames_test.append(aedats_in_folder[ind_test])
            class_test.append(i)        

    return filenames_train, class_train, filenames_test, class_test