import cv2
import disp
import measures
import matplotlib.pyplot as plt
import time

# HYPER PARAMETERS
WINDOW_SIZE = 7
PADDING_SIZE = (WINDOW_SIZE - 1) // 2

# READ IMAGES
# Image from Camera 1
RIGHTIMG = cv2.imread('2006data/aloe/view1.png', 0)
# Image from Camera 2
LEFTIMG = cv2.imread('2006data/aloe/view5.png', 0)


# To measure Time
startTime = time.time()

# PARAMETRES :
# IMG_SOURCE, IMG_TO_SEARCH_IN, PADDING_SIZE, MEASURE_TYPE, SEARCH_SPACE, SEARCH_SIDE
disparity_image = disp.dmap(LEFTIMG, RIGHTIMG, PADDING_SIZE, measures.ssd, 80, disp.searchRightSide)
#disparity_image = disp.dmap(RIGHTIMG, LEFTIMG, PADDING_SIZE, measures.ssd, 100, disp.searchLeftSide)

print('Time Elapsed : ', time.time() - startTime)

plt.imshow(disparity_image, cmap='gray')
plt.show()