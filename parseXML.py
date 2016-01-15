__author__ = 'ruoyu'
# this file includes the functions to parse a XML for contours, build up groudtruth mask, and extract patches and pack it.
from lxml import etree
import os, glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from os import path
from Save_to_lmdb import SavetoDB

def parseXML(filepath):
     with open(filepath,'r') as file:
        tree = etree.parse(file)
        boundary = [[(point.attrib['X'], point.attrib['Y']) for point in subtree.xpath('.//Vertices/Vertex')] for subtree in tree.xpath('//Region')]
        return boundary

def makecontour(img, boundary):
    # draw the contours on original image
    [cv2.polylines(img, [np.array(sparsecontour, np.int32).reshape((-1,1,2))],True,(255, 0, 0),5) for sparsecontour in boundary]
    return img

def makemask(img, boundary):
    # fill the inside of contours given by boundary, return the mask
    mask = np.zeros(img.shape, np.uint8)
    [cv2.fillPoly(mask, [np.array(sparsecontour, np.int32).reshape(sparsecontour.__len__(),2)], (255,0,0)) for sparsecontour in boundary]
    plt.imshow(mask)
    plt.show()
    return mask

def isPositive(point, width, mask):
    thred = 0.5
    if np.sum(mask[point[0]: point[0] + width[0][1] + width[0][0], point[1]: point[1] + width[1][1] + width[1][0], 0]) > thred*255*width[0][1]*width[0][0]:
        return True
    else:
        return False

def genePatch(img, point, width):
    patch = img[point[0]: point[0] + width[0][1] + width[0][0], point[1]: point[1] + width[1][1] + width[1][0], :]
    return patch

def randompatch(img, mask, para, ind_fig):
    # padding the mask and figure
    img_padded = np.zeros(para['size'], dtype=int)
    for ind in range(3):
        img_padded[:, :, ind] = np.pad(img[:, :, ind], para['pad_width'], mode='symmetric')
    mask_padded = np.zeros(para['size'], dtype=int)
    for ind in range(3):
        mask_padded[:, :, ind] = np.pad(mask[:, :, ind], para['pad_width'], mode='symmetric')

    patch_dir = para['folder'] + '/patch_positive'
    if not os.path.exists(patch_dir):               # check if this directory has existed, if not, create it with the name
        os.makedirs(patch_dir)

    image_id = 0
    cordrow_pos, cordcolumn_pos = np.where(mask[:, :, 0] > 0)
    cordrow_neg, cordcolumn_neg = np.where(mask[:, :, 0] == 0)
    if cordrow_pos.size != cordcolumn_pos.size | cordrow_pos.size < para['num_positive']:
        return
    else:
        sequence = range(cordrow_pos.size)
        np.random.shuffle(sequence)
        for index in sequence[:para['num_positive']]:
            if isPositive((cordrow_pos[index], cordcolumn_pos[index]), para['mask_width'], mask_padded):
                patch = genePatch(img_padded, (cordrow_pos[index], cordcolumn_pos[index]), para['mask_width'])
                patch_name = 'pos_'+str(ind_fig) + '_' + str(image_id) + '.jpg'
                if not path.exists(patch_dir + '/' + patch_name):      # if this path has existed (patch has been saved) do nothing, otherwise, save it
                    Image.fromarray(patch.astype(np.uint8)).save(patch_dir + '/' + patch_name)
                    image_id += 1

    patch_dir = para['folder'] + '/patch_negative'
    if not os.path.exists(patch_dir):               # check if this directory has existed, if not, create it with the name
        os.makedirs(patch_dir)

    image_id = 0
    if cordrow_neg.size != cordcolumn_neg.size | cordrow_neg.size < para['num_negative']:
        return
    else:
        sequence = range(cordrow_neg.size)
        np.random.shuffle(sequence)
        for index in sequence[:para['num_negative']]:
            if not isPositive((cordrow_neg[index], cordcolumn_neg[index]), para['mask_width'], mask_padded):
                patch = genePatch(img_padded, (cordrow_neg[index], cordcolumn_neg[index]), para['mask_width'])
                patch_name = 'neg_'+str(ind_fig) + '_' + str(image_id) + '.jpg'
                if not path.exists(patch_dir + '/' + patch_name):      # if this path has existed (patch has been saved) do nothing, otherwise, save it
                    Image.fromarray(patch.astype(np.uint8)).save(patch_dir + '/' + patch_name)
                    image_id += 1

def pad_para(size, work_path):
    pad_width = ((10, 10), (10, 10))                    # padding width on 2D image, ((height),(width)),
    mask_width = ((10, 10), (10, 10))
    size_padded = list()
    size_padded.append(size[0] + pad_width[0][0] + pad_width[0][1])   # size of image after padding
    size_padded.append(size[1] + pad_width[1][0] + pad_width[1][1])
    size_padded.append(size[2])
    size_padded = tuple(size_padded)
    num_positive = 1000
    num_negative = 1000
    para = {'num_positive': num_positive, 'num_negative': num_negative, 'size': size_padded, 'pad_width': pad_width, 'mask_width': mask_width, 'folder': work_path}
    return para

def plotcontour(img, boundary, work_path, index_fig):
    img_contour = makecontour(img, boundary)  # draw the ROI on the image
    ROI_path = work_path[:-5] +'/ROI'
    if not os.path.exists(ROI_path):               # check if this directory has existed, if not, create it with the name
       os.makedirs(ROI_path)
    img_name = ROI_path + '/ROI_labeled_' + str(index_fig) + '.jpg'
    Image.fromarray(img_contour.astype(np.uint8)).save(img_name) # save the image with labeld ROI ground truth


def pickpatch(work_path):
    index_fig = 0
    for file in glob.glob(work_path + '/*.xml'):
        # read XML file
        boundary = parseXML(file)
        imgformat = 'jpg'
        imagepath = file[:-3] + imgformat
        img = plt.imread(imagepath)
        size = img.shape

        mask = makemask(img, boundary)
        para = pad_para(size, work_path+'/patch')
        randompatch(img, mask, para, index_fig)
        index_fig += 1

    path = work_path + '/patch'
    LMDBDB = SavetoDB(path, path) # generate a instance of SavetoDB using the path for directory to file and database
    LMDBDB.save_to_lmdb() # save to database




if __name__ == '__main__':
    work_path = os.path.dirname(os.path.abspath(__file__))
    work_path +='/data'
    pickpatch(work_path)

