import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle 
import cv2
from matplotlib import pyplot as plt
import glob
from skimage.feature import hog


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
        (rows,cols,ch)=img.shape
        masktf=(img==img[0,0].tolist())
        masktf=(np.invert(np.all(masktf,axis=2))).astype(np.int)
        mask=np.repeat(masktf[:,:,np.newaxis],3,axis=2)
        nbimg=np.asarray(img*mask,dtype=np.uint8)
        gray = cv2.cvtColor(nbimg, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(gray,kernel,iterations = 3)
        ret3,th3 = cv2.threshold(erosion,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        th3=cv2.GaussianBlur(th3,(1,1),0)
        edges = cv2.Canny(th3,0,0)
        edges = cv2.dilate(edges,kernel,iterations = 2)
        edges=cv2.GaussianBlur(edges,(1,1),0)
        edges=cv2.erode(edges,kernel,iterations = 1)
        edged=edges.copy()
        _, contours, _ =cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cnt=sorted(contours,key=cv2.contourArea,reverse=True)[:4]
        cnt=list(filter(lambda x: cv2.contourArea(x)>1000 , cnt))
        testArr=np.zeros((len(cnt),4),dtype=np.uint)
        areaArr=np.zeros((len(cnt),),dtype=np.float32)
        for i,j in zip(cnt,np.arange(len(cnt))):
            x,y,w,h= cv2.boundingRect(i)
            testArr[j]=[x,y,w,h]
            areaArr[j]=cv2.contourArea(i)
        order=testArr[:,0].argsort()
        testArr=testArr[order]
        areaArr=areaArr[order]
        alphs=[]
        for alp,arr in zip(testArr,areaArr):
            rx,ry,rw,rh=alp
            oneL=edges[ry:(ry+rh),rx:(rx+rw)]
            oneL=np.invert(oneL)
            oneL = cv2.resize(oneL,(64,64), interpolation = cv2.INTER_CUBIC)
            feat, _ = hog(oneL, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=False)
            codeList.append(np.array(np.hstack((feat,[arr]))))
        numChars.append(len(cnt))
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