""" Generates the Corrupted Data for CIFAR10 Data Set """

homepath = '/home/jfernando/'
projectpath = homepath + 'PycharmProjects/Deep_Learning_Final_Project/'
sourcepath = projectpath + 'Generate_Data/'
datapath = sourcepath + 'CIFAR10_Data/Old_CIFAR10/'
outputpath = sourcepath + 'CIFAR10_Data/'

import pickle
import numpy as np
import os


def stack_data(pathlist, datapath, outputpath, prefix = "Train"):
    data = None
    labels = None
    for apath in pathlist:
        with open(datapath+apath, 'rb') as f:
                # unpickling from python 2
            unpick = pickle._Unpickler(f)
            unpick.encoding = 'latin1'
            temp_dict = unpick.load()
            temp_data = temp_dict['data']
            temp_labels = temp_dict['labels']

            if data is None:
                data = temp_data
            else:
                data = np.vstack([data, temp_data])

            if labels is None:
                labels = temp_labels
            else:
                labels = np.concatenate([labels, temp_labels])

    pickle.dump(data, open(outputpath+"CIFAR10_"+prefix+"_Data", "wb"))
    pickle.dump(np.array(labels), open(outputpath+"CIFAR10_"+prefix+"_Labels", "wb"))


if __name__ == "__main__":
    myfiles = np.sort(os.listdir(datapath))
    train_files = myfiles[0:5]
    test_files = [myfiles[5]]
    stack_data(train_files, datapath, outputpath, "Train")
    stack_data(test_files, datapath, outputpath, "Test")

