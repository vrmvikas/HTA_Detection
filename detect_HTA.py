from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-ti','--test_image_path', help='Path to test image - PCB Under Inspection')
parser.add_argument('-gt','--groundtruth_path', help='Path to test image - PCB Under Inspection')

args = parser.parse_args()

testImagePath = args.test_image_path
GT_path = args.groundtruth_path
testImage, GT, BinImage, contours = Get_Contours_Of_SusRegions(testImagePath, GT_path, BinThreshold= 100, MIN_AREA_THRESHOLD = 300)
All_distances = SiameseNetwork(contours, testImage, GT)

BoundryColor = (255,0,0)
threshold = 17
Num_Of_Detections = 5

cleanContours = [contours[i] for i in np.argsort(All_distances)[::-1][:Num_Of_Detections]]    
modifiedImage = Draw_Rectangle_Using_xywh(testImage, cleanContours, thickness=5, color=BoundryColor)

# Draw red contour on GT image and save it
plt.imsave("op1.jpg", modifiedImage)

# Draw red contour on white background and save it
plt.imsave("op2.jpg", Draw_Rectangle_Using_xywh(np.ones_like(GT)*255, cleanContours, thickness=5, color=BoundryColor))

# To run this file:
# python detect_HTA.py -ti PUI_image.jpg -gt GT_image.jpg