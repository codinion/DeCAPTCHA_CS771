""" 
This script contains functions for extracting training features from input training images
Author : Abir Mukherjee
Email : abir.mukherjee0595@gmail.com
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from collections import deque
import glob


# This function implements the Histogram of Gradients(HoG) feature extractor.
# It takes as input a single image, and return the HoG features.
def extractHOG(img):
    #Gradients of the image are calculated
    img=np.float32(img)/255.0
    dix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    diy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(dix, diy, angleInDegrees=True)

    #A Histogram of the gradients is calculated in 64x64 blocks
    hist=np.zeros((8,8,9))
    for i in np.arange(64):
        for j in np.arange(64):
            hr=np.int32(np.floor(i/8))
            hc=np.int32(np.floor(j/8))
            ang=angle[i,j]%180
            bno=np.int32(np.floor(ang/20))
            nbno=(bno+1)%9
            rem=ang%20
            w0=(20-rem)/20
            w1=rem/20
            hist[hr,hc,bno]+=w0*mag[i,j]
            hist[hr,hc,nbno]+=w1*mag[i,j]

    #The histograms are transformed into HoG features.
    features=np.zeros((1764,))
    featidx=0
    for i in np.arange(7):
        for j in np.arange(7):
            tmp=np.zeros((36,))
            tmp[0:9]=np.array(hist[i,j,:])
            tmp[9:18]+=hist[i,j+1,:]
            tmp[18:27]+=hist[i+1,j,:]
            tmp[27:36]+=hist[i+1,j+1,:]
            normv=np.linalg.norm(tmp)
            if(normv!=0):
                tmp=tmp/normv
            features[featidx:featidx+36]=tmp
            featidx+=36       
    return features


# This function processes the training images. 
# Returns a list of binary images of individual alphabets and the areas of their bounding boxes from the training image.
def preprocessTrainingImage(imgName):
    img = cv2.imread(imgName)
    (rows,cols,ch)=img.shape
    # Create a mask having 0's on pixels having colours equal to the background color.
    # Background is assumed to be the color of the corner pixel
    # This is mask is used to extract the non-background pixels of the image, then convert it to grayscale
    masktf=(img==img[0,0].tolist())
    masktf=(np.invert(np.all(masktf,axis=2))).astype(np.int)
    mask=np.repeat(masktf[:,:,np.newaxis],3,axis=2)
    maskedImg=np.asarray(img*mask,dtype=np.uint8)
    grayImg = cv2.cvtColor(maskedImg, cv2.COLOR_BGR2GRAY)

    # Use a 3x3 kernel to perform erosion over the images.
    # This is performed to remove the line artifacts from the images.
    # Eroded image is then adaptively thresholded and gaussian blurred to smooth it out.
    kernel = np.ones((3,3),np.uint8)
    erodedImg = cv2.erode(grayImg,kernel,iterations = 3)
    thresholdedImg = cv2.adaptiveThreshold(erodedImg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,3)
    blurredImg=cv2.GaussianBlur(thresholdedImg,(3,3),0)

    # Edge detection is now performed over the obtained image.
    # Canny edge detection algorithm is used.
    # Obtained edges are then dilated and blurred for smoothing.
    # Again erosion is performed for removing artifacts, and a copy is made for contouring.
    edges = cv2.Canny(blurredImg,0,0)
    edges = cv2.dilate(edges,kernel,iterations = 2)
    edges=cv2.GaussianBlur(edges,(1,1),0)
    edges=cv2.erode(edges,kernel,iterations = 2)
    edged=edges.copy()

    # Contouring is done in order to find the letters in the image.
    # Contours are sorted in order of decreasing area, and the top 4 are selected.
    # Only contours having area greater than 100 are retained from the top 4 to remove contours formed by artifacts
    # Bounding boxes are formed around the contours of the identified letters, and are displayed.
    contours,_=cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contourList=sorted(contours,key=cv2.contourArea,reverse=True)[:4]
    trueContours=list(filter(lambda x: cv2.contourArea(x)>100 , contourList))
    boundingRects=np.zeros((len(trueContours),4),dtype=np.uint)
    areaArr=np.zeros((len(trueContours),),dtype=np.float32)
    for i,j in zip(trueContours,np.arange(len(trueContours))):
        x,y,w,h= cv2.boundingRect(i)
        boundingRects[j]=[x,y,w,h]
        areaArr[j]=cv2.contourArea(i)
    order=boundingRects[:,0].argsort()
    boundingRects=boundingRects[order]
    areaArr=areaArr[order]

    # The letters corresponding to the bounding box regions are extracted from the image containing edges.
    alphabetList=[]
    for alp in boundingRects:
        rx,ry,rw,rh=alp
        oneL=edges[ry:(ry+rh),rx:(rx+rw)]
        oneL=np.invert(oneL)
        oneL = cv2.resize(oneL,(64,64), interpolation = cv2.INTER_CUBIC)
        alphabetList.append(oneL)
    
    return alphabetList,areaArr


def preprocessReferenceImage(imgName,lowerAngle,upperAngle):
    img = cv2.imread(imgName)
    (rows,cols,ch)=img.shape
    # The input image is rotated at different angles between lowerAngle and upperAngle to generate training data for different angles of rotation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rotatedAlphs=[]
    rotatedAreas=[]
    for deg in range(lowerAngle,upperAngle,10):
        M = cv2.getRotationMatrix2D((cols/2,rows/2),deg,1)
        gray = cv2.warpAffine(gray,M,(cols,rows),borderValue=(255,255,255))
        # Edge detection is performed, followed by contouring
        # The bounding box of the largest contour is selected, and the corresponding region is extracted.
        edged=np.invert(gray.copy())
        contours, hierarchy=cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cnt=sorted(contours,key=cv2.contourArea,reverse=True)[:1]
        contArea=cv2.contourArea(cnt[0])
        x,y,w,h= cv2.boundingRect(cnt[0])
        oneL=gray[y:(y+h),x:(x+w)]
        kernel = np.ones((2,2),np.uint8)
        oneL = cv2.dilate(oneL,kernel,iterations = 2)
        oneL = cv2.resize(oneL,(64,64), interpolation = cv2.INTER_CUBIC)
        rotatedAlphs.append(oneL)
        rotatedAreas.append(contArea)
    return rotatedAlphs,rotatedAreas




if __name__=="__main__":
    trainPath = glob.glob("dataset/*.png")
    refPath = glob.glob("reference/*.png")
    trainData=[]

    #Generate features from the training dataset
    for imgname in trainPath:
        alphList=list((imgname.split('/')[1]).split('.')[0]) 
        extractedAlphs,boundArea=preprocessTrainingImage(imgname)
        for alphImg,alphTxt,alphArea in zip(extractedAlphs,alphList,boundArea):
            feat=extractHOG(alphImg)
            tdPoint=np.hstack((feat,[alphArea],[ord(alphTxt)-ord('A')]))
            trainData.append(tdPoint)

    # Generate features from the reference dataset
    for imgname in refPath:
        ltr=ord((imgname.split('/')[1]).split('.')[0])-ord('A')
        ulim=50
        llim=-40
        if (ltr+ord('A'))==ord('Z') or (ltr+ord('A'))==ord('N'):
            ulim=30
            llim=-20
        orientedAlphs,orientedAreas=preprocessReferenceImage(imgname,llim,ulim)
        for alphImg,alphArea in zip(orientedAlphs,orientedAreas):
            feat=extractHOG(alphImg)
            tdPoint=np.hstack((feat,[alphArea],[ltr]))
            trainData.append(tdPoint)
    np.savetxt('data.out', np.array(trainData), delimiter=',') 
