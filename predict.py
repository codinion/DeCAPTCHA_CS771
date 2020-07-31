""" 
This script contains functions for predicting the number of alphabets and the code in the given test images.
This script is called by the eval script.
Author : Abir Mukherjee
Email : abir.mukherjee0595@gmail.com
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle 
import cv2
from matplotlib import pyplot as plt
import glob
from skimage.feature import hog

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



# This function processes the test image.
# Returns a list of binary images of individual alphabets and the areas of their bounding boxes from the test image.
def preprocessTestImage(img):
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

def decaptcha( filenames ):
    clf = pickle.load(open('model', 'rb'))
    numChars=[]
    codes=[]
    cv_img=[]
    codeList=[]
    for imgname in filenames:
        img = cv2.imread(imgname)
        cv_img.append(img)
    for img in cv_img:
        print(np.shape(img))
        testAlphs,testAreas=preprocessTestImage(img)
        for alp,arr in zip(testAlphs,testAreas):
            feat=extractHOG(alp)
            codeList.append(np.array(np.hstack((feat,[arr]))))
        numChars.append(len(testAreas))
    ch=clf.predict(codeList)
    ch[:]+=ord('A')
    ch=[chr(ch[i]) for i in np.arange(len(ch))]
    idx=0
    for nc in numChars:
        code=''.join(ch[idx:idx+nc])
        codes.append(code)
        idx=idx+nc
    numChars=np.array(numChars)
    return (numChars, codes)