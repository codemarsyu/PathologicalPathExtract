__author__ = 'ruoyu'
import caffe
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import lmdb
import random
from PIL import Image

class SavetoDB:
    def __init__(self, pathfile, pathDB):
        self.file_path = pathfile
        self.DB_path = pathDB

    def save_to_lmdb(self):
        #work_path = os.path.dirname(os.path.abspath(__file__))
        #file_path = work_path + '/ADC'
        if not os.path.exists(self.DB_path + '/lmdb/train'):
            os.makedirs(self.DB_path + '/lmdb/train')
        if not os.path.exists(self.DB_path + '/lmdb/test'):
            os.makedirs(self.DB_path + '/lmdb/test')
        db_train = lmdb.open(self.DB_path + '/lmdb/train', map_size=int(1e12))
        db_test = lmdb.open(self.DB_path + '/lmdb/test', map_size=int(1e12))

        # read all the subfolder names in ADC
        for files in os.walk(self.file_path + '/patch_positive'):
            #print("dir: %s" % dir)
            img_list_pos = files[2]
            break

        for files in os.walk(self.file_path + '/patch_negative'):
            #print("dir: %s" % dir)
            img_list_neg = files[2]
            break


        # select part of folders for training and the rest for testing
        size_pos = img_list_pos.__len__()
        size_neg = img_list_neg.__len__()

        index_pos = range(int(size_pos))
        index_neg = range(int(size_neg))
        random.shuffle(index_pos)
        random.shuffle(index_neg)
        train_pos_pool = index_pos[0:int(np.ceil(size_pos*0.9))]                   # training set, 0.9 is the ratio of sampels for training
        test_pos_pool = index_pos[int(np.ceil(size_pos*0.9)):size_pos+1]           # testing set, the rest 0.1 is for testing
        train_neg_pool = index_neg[0:int(np.ceil(size_neg*0.9))]
        test_neg_pool = index_neg[int(np.ceil(size_neg*0.9)):size_neg+1]
        count_neg = 0
        count_pos = 0
        with db_train.begin(write=True) as train_txn:
            for patch_ind in train_pos_pool:
                patch_name = img_list_pos[patch_ind]
                patch_dir = self.file_path + '/patch_positive/' + patch_name
                image = plt.imread(patch_dir).transpose((2, 0, 1))
                if not image.shape[1] == 20 & image.shape[2] == 20:
                    continue
                # Load image into datum object
                datum = caffe.io.array_to_datum(image.astype(float), 1)
                train_txn.put('%08d_%s' % (count_pos, 'pos'), datum.SerializeToString())
                count_pos += 1

            for patch_ind in train_neg_pool:
                patch_name = img_list_neg[patch_ind]
                patch_dir = self.file_path + '/patch_negative/' + patch_name
                image = plt.imread(patch_dir).transpose((2, 0, 1))
                if not image.shape[1] == 20 & image.shape[2] == 20:
                    continue
                # Load image into datum object
                datum = caffe.io.array_to_datum(image.astype(float), 0)
                train_txn.put('%08d_%s' % (count_neg, 'neg'), datum.SerializeToString())
                count_neg += 1
        print('Finish saving training data to %s' % self.DB_path + '/lmdb/train')

        count_neg = 0
        count_pos = 0
        with db_test.begin(write=True) as test_txn:
            for patch_ind in test_pos_pool:
                patch_name = img_list_pos[patch_ind]
                patch_dir = self.file_path + '/patch_positive/' + patch_name
                image = plt.imread(patch_dir).transpose((2, 0, 1))
                if not image.shape[1] == 20 & image.shape[2] == 20:
                    continue
                # Load image into datum object
                datum = caffe.io.array_to_datum(image.astype(float), 1)
                test_txn.put('%08d_%s' % (count_pos, 'pos'), datum.SerializeToString())
                count_pos += 1

            for patch_ind in test_neg_pool:
                patch_name = img_list_neg[patch_ind]
                patch_dir = self.file_path + '/patch_negative/' + patch_name
                image = plt.imread(patch_dir).transpose((2, 0, 1))
                if not image.shape[1] == 20 & image.shape[2] == 20:
                    continue
                # Load image into datum object
                datum = caffe.io.array_to_datum(image.astype(float), 0)
                test_txn.put('%08d_%s' % (count_neg, 'neg'), datum.SerializeToString())
                count_neg += 1
        print('Finish saving training data to %s' % self.DB_path + '/lmdb/test')

        print('Finish saving all data to lmdb format at %s' % self.DB_path + '/lmdb')


