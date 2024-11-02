import numpy as np
from PIL import Image
import copy, cv2
import matplotlib.pyplot as plt
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import transforms


device = 'cuda'
ResModel = resnet50(weights=ResNet50_Weights.DEFAULT)
ResModel.fc = torch.nn.Identity()
ResModel.to(device)
ResModel.eval()
import tqdm
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, numpy_list):
        self.numpy_list = numpy_list

    def __len__(self):
        return len(self.numpy_list)

    def __getitem__(self, idx):
        numpy_array = self.numpy_list[idx]
        # tensor = torch.from_numpy(numpy_array).float()
        return numpy_array


transform = transforms.Compose([
    transforms.Resize((224,224)),       # Can be made better
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def calculate_iou(box1, box2):
    # determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # compute the area of intersection rectangle
    intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

    # compute the area of both the prediction and ground-truth rectangles
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    # return the intersection over union value
    return iou

def cleanContoursV2(contours, MIN_AREA_THRESHOLD = 30):
    contours = list(contours)
    newContours = [cv2.boundingRect(contour) for contour in contours]
    all_w = np.array(newContours)[:,2]
    all_h = np.array(newContours)[:,3]
    areas = all_w * all_h
    
    idx_list = np.where(areas < MIN_AREA_THRESHOLD)[0].tolist()
    newContoursCleaned = [ele for idx, ele in enumerate(newContours) if idx not in idx_list]
    newContours = copy.deepcopy(newContoursCleaned)        

    print("After eliminating small bboxes:\t\t", len(newContoursCleaned),end='\n\n')
    # print(np.unique(areas, return_counts=True))
    
    cleanContours = copy.deepcopy(newContoursCleaned)

    boxContours = [(x, y, x+w, y+h) for (x,y,w,h) in newContours]
    for i in tqdm.trange(len(boxContours), desc = "Merging overlaping bboxes:\t"):
        for j in range(i+1, len(boxContours)):
            thisdata = calculate_iou(boxContours[i], boxContours[j])
            if thisdata > 0.000:
                if newContours[i][2] < newContours[j][2]:
                    try:
                        cleanContours.remove(newContours[i])
                    except:
                        pass
                else:
                    try:
                        cleanContours.remove(newContours[j])                
                    except:
                        pass
    return cleanContours


def align_images(image, template, maxFeatures=500, keepPercent=0.2, debug=False):
    # Convert both the input image and template to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and extract local invariant descriptors
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)

    # Match keypoints using a brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descsA, descsB)

    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Keep only the top matches
    numKeep = int(len(matches) * keepPercent)
    matches = matches[:numKeep]

    # Extract the matched keypoints
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    for (i, m) in enumerate(matches):
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # Compute the homography matrix
    (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)

    # Align the images
    aligned = cv2.warpPerspective(image, H, (template.shape[1], template.shape[0]))
    # aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    return aligned

def Get_Contours_Of_SusRegions(PUI_path, GT_path, BinThreshold=100, MIN_AREA_THRESHOLD = 30):
    testImage = cv2.imread(PUI_path)
    GT = cv2.imread(GT_path)

    testImage = align_images(testImage, GT)
    
    image1 = copy.deepcopy(GT)
    image2 = copy.deepcopy(testImage)
    diff = cv2.absdiff(image1, image2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)       # Convert the difference image to grayscale

    # Threshold the difference image
    _, thresh = cv2.threshold(gray, BinThreshold, 255, cv2.THRESH_BINARY)


    # Find contours in the threshold image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # boxesOnly = np.ones_like(thresh) *255
    print("Initial no. of bboxes found:\t\t", len(contours))
    
    contours_new = cleanContoursV2(contours, MIN_AREA_THRESHOLD)
    print("After merging overlapping bboxes:\t", len(contours_new),end='\n\n')
    
    return cv2.cvtColor(testImage, cv2.COLOR_BGR2RGB), cv2.cvtColor(GT, cv2.COLOR_BGR2RGB), thresh, contours_new

def Draw_Rectangle_Using_xywh(image, contours, thickness = 2, color=(255,0,0)):
    # import cv2

    # # Load your image (replace 'path_to_image' with the actual path)
    # image = cv2.imread('path_to_image')

    # Define the starting and ending coordinates of the rectangle
    localimage = copy.deepcopy(image)
    for (x,y,w,h) in contours:
        start_point = (x, y)  
        end_point = (x + w, y + h)  

        # Specify the color (BGR format) and thickness of the rectangle
        # color = (0, 255, 0)  # Blue color (you can change this)
        # thickness = 2  # Thickness of the rectangle border

        # Draw the rectangle on the image
        localimage = cv2.rectangle(localimage, start_point, end_point, color, thickness)
    return localimage

def CropHTAandShow(GT, testImage, contours_new, distances = None):
    image1 = GT
    for i, contour in enumerate(contours_new):
        # (x, y, w, h) = cv2.boundingRect(contour)
        (x, y, w, h) = contour
        if w<7 or h<7:
            continue
        # print(cv2.boundingRect(contour))
        
        plt.subplot(221)
        plt.imshow(image1[y:y+h, x:x+w])
        plt.subplot(223)
        plt.imshow(testImage[y:y+h, x:x+w])
        
        
        bordersize = 50
        
        plt.subplot(222)
        plt.imshow(image1[y-bordersize:y+h+bordersize, x-bordersize:x+w+bordersize])
        plt.subplot(224)
        plt.imshow(testImage[y-bordersize:y+h+bordersize, x-bordersize:x+w+bordersize])
        # print(distances)
        plt.show()
        if distances:
            print("Distance is: ",distances[i])
        print("*"*30)    
      
def SiameseNetwork(contours, image, GT):
    bordersize = 0
    batch_size = 32

    PUI_crops_list = [transform(Image.fromarray(image[y-bordersize:y+h+bordersize, x-bordersize:x+w+bordersize])) for x,y,w,h in contours]
    PUI_crops_dataset = CustomDataset(PUI_crops_list)
    PUI_dataloader = DataLoader(PUI_crops_dataset, shuffle=False, batch_size = batch_size)

    GT_crops_list = [transform(Image.fromarray(GT[y-bordersize:y+h+bordersize, x-bordersize:x+w+bordersize])) for x,y,w,h in contours]
    GT_crops_dataset = CustomDataset(GT_crops_list)
    GT_dataloader = DataLoader(GT_crops_dataset, shuffle=False, batch_size = batch_size)

    All_distances = []
    for PUI_batch, GT_batch in tqdm.tqdm(zip(PUI_dataloader, GT_dataloader), total = len(PUI_dataloader), desc="Siamese Network Working\t\t"):
        PUI_batch, GT_batch = PUI_batch.to(device), GT_batch.to(device)
        PUI_crop_emb = ResModel(PUI_batch).detach()
        GT_crop_emb = ResModel(GT_batch).detach()
        distance = (PUI_crop_emb - GT_crop_emb).pow(2).sum(dim=1).sqrt()
        All_distances += distance.tolist()

    return All_distances    #, contours, image, GT


