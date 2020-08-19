import cv2
import numpy as np
from skimage.exposure import rescale_intensity


def searchRightSide(bloc, img, row, col, bindex, searchspace, dmeasure, errors):
    begin = max(col-1, bindex)
    end = min(img.shape[1]-bindex, col + searchspace)

    for j in range(begin, end):
        bl = img[row - bindex : row + bindex + 1, j - bindex : j + bindex + 1]
        errors[j - begin] = dmeasure(bl, bloc)

    return begin - bindex + np.argmin(errors[:end-begin])


def searchLeftSide(bloc, img, row, col, bindex, searchspace, dmeasure, errors):
    end = min(col+1, img.shape[1]-bindex) 
    begin = max(bindex, col - searchspace)

    for j in range(begin, end):
        bl = img[row - bindex : row + bindex + 1, j - bindex : j + bindex + 1]
        errors[j - begin] = dmeasure(bl.astype(np.float), bloc)

    return begin - bindex + np.argmin(errors[:end-begin])


def dmap(leftside, rightside, bindex, dmeasure, searchspace, searchside):
    dss = np.zeros(leftside.shape, dtype=np.float)
    errors = np.zeros((leftside.shape[1]))
    nleftside = cv2.copyMakeBorder(leftside, bindex, bindex, bindex, bindex, cv2.BORDER_REPLICATE)
    nrightside = cv2.copyMakeBorder(rightside, bindex, bindex, bindex, bindex, cv2.BORDER_REPLICATE)

    for i in range(bindex, nleftside.shape[0]-bindex):
        if(i % 10 == 0):
            print(i, '/', nleftside.shape[0]-bindex)
        for j in range(bindex, nleftside.shape[1]-bindex):
            block = nleftside[i-bindex:i+bindex+1 , j-bindex: j+bindex+1]
            indexu = searchside(block.astype(np.float), nrightside, i, j, bindex, searchspace, dmeasure, errors)
            dss[i-bindex, j-bindex] =  abs((j-bindex) - indexu) 

    output = rescale_intensity(dss, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    return output