# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 19:32:17 2021

@author: XuPeng
"""

import cv2
import numpy as np
cv2.ocl.setUseOpenCL(False)
from matplotlib import pyplot as plt
import random
import math

# 画线函数
def draw(out,pt1,pt2):
    cv2.line(output, (int(pt1[0]),int(pt1[0])), (int(pt2[1]),int(pt2[1])), (255, 0, 0))
    
    
# K-means 聚类
# in:二维数据点 xMax,yMax：边界最大值（图像尺寸）
def Kmeans(input,k,xMax,yMax):
    # 加上分类信息
    keyPoint = [[0 for x in range(3)] for y in range(len(input))] 
    for i in range(len(keyPoint)):
        keyPoint[i][0] = input[i][0]
        keyPoint[i][1] = input[i][1]
        keyPoint[i][2] = 999
    # 初始化 k 个中心点 
    center = [[0 for x in range(3)] for y in range(k)] 
    #radious = [0 for x in range(k)]
    for i in range(k):
        center[i][0] = random.randint(0,xMax)
        center[i][1] = random.randint(0,yMax)
    
    # 停止迭代的三个条件
    time = 0 # 迭代次数
    timeMax = 30
    changed = 0 # 重新分配
    a = 0.01 # 最小移动与图像尺度的比例
    move = 0 # 所有类中心移动距离小于moveMax
    moveMax = a*xMax
    
    # 未到最大迭代次数
    while time < timeMax:
        time = time + 1
        # 计算每个点的最近分类
        for i in range(len(keyPoint)):
            dis = -1
            for j in range(k):
                x = keyPoint[i][0]- center[j][0]
                y = keyPoint[i][1]- center[j][1]
                disTemp = x*x + y*y
                # 更新当前最近分类并标记
                if (disTemp < dis) | (dis == -1):
                    dis = disTemp
                    keyPoint[i][2] = j
        # 更新类中心点坐标
        for i in range(k):
            xSum = 0
            ySum = 0
            num = 0
            for j in range(len(keyPoint)):
                if keyPoint[j][2] == i:
                    xSum = xSum + keyPoint[j][0]
                    ySum = ySum + keyPoint[j][1]
                    num = num + 1
            if num != 0:
                center[i][0] = xSum/num
                center[i][1] = ySum/num
     # 记录每个分类的点数量  
    for i in range(len(keyPoint)):
        center[keyPoint[i][2]][2] = center[keyPoint[i][2]][2] + 1
    return center



if __name__ == '__main__':
    
    # 选择 
    # 0:输出SIFT匹配结果
    # 1:输出差异识别结果
    # 2:用圆形圈出来
    func = 2
    
    
    # 载入图像
    img1 = cv2.imread('./datahomo6/img1l.png')
    img2 = cv2.imread('./datahomo6/img2d.png')
    
    sift = cv2.xfeatures2d.SIFT_create()
    
    # 检测关键点
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    # 关键点匹配
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 6)
    search_params = dict(checks = 10)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1,des2,k=2)
    
    good = []
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            good.append(m)
    
    # 把good中的左右点分别提出来找单应性变换
    pts_src = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    pts_dst = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    # 单应性变换
    M, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC,5.0)
    
    
    # 输出SIFT匹配结果 ********************************************************
    if func == 0:
        
        # 输出图片初始化
        height = max(img1.shape[0], img2.shape[0])
        width = img1.shape[1] + img1.shape[1]
        output = np.zeros((height, width, 3), dtype=np.uint8)
        output[0:img1.shape[0], 0:img1.shape[1]] = img1
        output[0:img2.shape[0], img2.shape[1]:] = img2[:]
        
        # 把点画出来
        _1_255 = np.expand_dims( np.array( range( 0, 256 ), dtype='uint8' ), 1 )
        _colormap = cv2.applyColorMap(_1_255, cv2.COLORMAP_HSV)
        
        for i in range(len(mask)):
            
            left = pts_src[i][0]
            right = pts_dst[i][0]
            colormap_idx = int( (left[0] - img1.shape[1]*.5 + left[1] - img1.shape[0]*.5) * 256. / (img1.shape[0]*.5 + img1.shape[1]*.5) )
            
            if mask[i] == 1:
                color = tuple( map(int, _colormap[ colormap_idx,0,: ]) )
                if i%2 == 0:
                    cv2.circle(output, (int(pts_src[i][0][0]),int(pts_src[i][0][1])),2,color, 2)
                    cv2.circle(output, (int(pts_dst[i][0][0])+img1.shape[1],int(pts_dst[i][0][1])),2,color, 2)
                    cv2.line(output, (pts_src[i][0][0],pts_src[i][0][1]), (int(pts_dst[i][0][0]+img1.shape[1]),pts_dst[i][0][1]), color, 1, 0)
    
        # 最终结果输出
        a = 1
        outputN = cv2.resize(output,(int(img1.shape[1]*2*a), int(img1.shape[0]*a)), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('show', outputN)
        k = cv2.waitKey(0)
        if k ==27:
            cv2.destroyAllWindows() 
    
    
    # 输出差异识别结果 ********************************************************
    if (func == 1)|(func == 2):
        
		# M矩阵中xy方向的偏移量
        dX = M[0][2] # x方向 负为左比右小
        dY = M[1][2] # y方向 负为左比右小
        
		# 图像的长宽
        height,width,channel = img1.shape    
        
        # 设定关键点的尺度
        size = int(width * 0.01)
        
        # 自动选择采样点的位置范围
        xMinLeft = width
        xMaxLeft = 0
        yMinLeft = height
        yMaxLeft = 0
        xMinRight = width
        xMaxRight = 0
        yMinRight = height
        yMaxRight = 0
        
        # 用当前匹配成功的点集分析合适的检测范围
        for i in range(len(pts_src)):
            if mask[i] == 1:
                if pts_src[i][0][1] < yMinLeft:
                    yMinLeft = pts_src[i][0][1]
                if pts_src[i][0][1] > yMaxLeft:
                    yMaxLeft = pts_src[i][0][1]    
                if pts_src[i][0][0] < xMinLeft:
                    xMinLeft = pts_src[i][0][0]
                if pts_src[i][0][0] > xMaxLeft:
                    xMaxLeft = pts_src[i][0][0]
        for i in range(len(pts_dst)):
            if mask[i] == 1:
                if pts_dst[i][0][1] < yMinRight:
                    yMinRight = pts_dst[i][0][1]
                if pts_dst[i][0][1] > yMaxRight:
                    yMaxRight = pts_dst[i][0][1]    
                if pts_dst[i][0][0] < xMinRight:
                    xMinRight = pts_dst[i][0][0]
                if pts_dst[i][0][0] > xMaxRight:
                    xMaxRight = pts_dst[i][0][0]
        
        # 转换为int型
        if xMinLeft > xMinRight:
            xMin = int(xMinLeft)
        else:
            xMin = int(xMinRight)
        if xMaxLeft < xMaxRight:
            xMax = int(xMaxLeft)
        else:
            xMax = int(xMaxRight)
        if yMinLeft > yMinRight:
            yMin = int(yMinLeft)
        else:
            yMin = int(yMinRight)
        if yMaxLeft < yMaxRight:
            yMax = int(yMaxLeft)
        else:
            yMax = int(yMaxRight)
        
        # 检测范围确定
        interval = 2.5*size    # 监测点间隔
        searchWidth = int((xMaxLeft - xMinLeft)/interval)
        searchHeight = int((yMaxLeft - yMinLeft)/interval)
        searchNum = searchWidth * searchHeight
        demo_src = np.float32([[0] * 2] * searchNum * 1).reshape(-1,1,2)
        for i in range(searchWidth):
            for j in range(searchHeight):
                demo_src[i+j*searchWidth][0][0] = xMinLeft + i*interval
                demo_src[i+j*searchWidth][0][1] = yMinLeft + j*interval
        
     
        # 单应性变换 左图映射到右图的位置
        demo_dst = cv2.perspectiveTransform(demo_src,M)    
        
        # 把差异点画出来
        heightO = max(img1.shape[0], img2.shape[0])
        widthO = img1.shape[1] + img1.shape[1]
        output = np.zeros((heightO, widthO, 3), dtype=np.uint8)
        output[0:img1.shape[0], 0:img1.shape[1]] = img1
        output[0:img2.shape[0], img2.shape[1]:] = img2[:]
        # output2
        output2 = output
        
        # 转换成KeyPoint类型
        kp_src = [cv2.KeyPoint(demo_src[i][0][0], demo_src[i][0][1], size)
                            for i in range(demo_src.shape[0])]
        kp_dst = [cv2.KeyPoint(demo_dst[i][0][0], demo_dst[i][0][1], size)
                            for i in range(demo_dst.shape[0])]
        
        # 计算这些关键点的SIFT描述子
        keypoints_image1, descriptors_image1 = sift.compute(img1, kp_src)
        keypoints_image2, descriptors_image2 = sift.compute(img2, kp_dst)        
        
        # 差异点
        diffLeft = []
        diffRight = []
        
        # 分析差异
        for i in range(searchNum):
            
            shreshood = 470
            difference = 0
            for j in range(128):
                d = abs(descriptors_image1[i][j]-descriptors_image2[i][j])
                difference = difference + d*d
            difference = math.sqrt(difference)
            
            # 右图关键点位置不超出范围
            if (demo_dst[i][0][1]>= 0) & (demo_dst[i][0][0] >= 0):
                if difference <= shreshood:
                    cv2.circle(output, (demo_src[i][0][0],demo_src[i][0][1]),1, (0, 255, 0), 2)
                    cv2.circle(output, (int(demo_dst[i][0][0]+width),demo_dst[i][0][1]),1, (0, 255, 0), 2)
                    
                if difference > shreshood:
                    if func == 1:
                        cv2.circle(output, (demo_src[i][0][0],demo_src[i][0][1]),1, (0, 0, 255), 2)
                        cv2.circle(output, (int(demo_dst[i][0][0]+width),demo_dst[i][0][1]),1, (0, 0, 255), 2)
                    if func == 2:
                        diffLeft.append([demo_src[i][0][0],demo_src[i][0][1]])
                        diffRight.append([demo_dst[i][0][0],demo_dst[i][0][1]])
              
        # 检测结果输出
        if func == 1:
            a = 1
            output = cv2.resize(output,(int(output.shape[1]*a), int(output.shape[0]*a)), interpolation=cv2.INTER_CUBIC)
            cv2.imshow('show', output)
            k = cv2.waitKey(0)
            if k ==27:
                cv2.destroyAllWindows() 
        
        # 聚类后输出
        initialClassNum = 2
        if func == 2:
            outLeft = Kmeans(diffLeft,initialClassNum,img1.shape[0],img1.shape[1])
            numClass = 0
            for i in range(initialClassNum):
                if outLeft[i][2] > 2:
                    numClass = numClass + 1
            outRight = Kmeans(diffRight,numClass,img1.shape[0],img1.shape[1])
            numClass2 = 0
            for i in range(numClass):
                if outRight[i][2] > 2:
                    numClass2 = numClass2 + 1
            if numClass2 < numClass:
                outLeft = Kmeans(diffLeft,numClass2,img1.shape[0],img1.shape[1])
            
            # 将点数大于2的类画出来 点数不足2认为是错误导致的
            for i in range(len(outLeft)):
                if outLeft[i][2] > 2:
                    cv2.circle(output2, (int(outLeft[i][0]),int(outLeft[i][1])),int(np.sqrt(outLeft[i][2]))*size, (0, 255, 255), 2)
            for i in range(len(outRight)):   
                if outRight[i][2] > 2:
                    cv2.circle(output2, (int(outRight[i][0])+width,int(outRight[i][1])),int(np.sqrt(outRight[i][2]))*size, (255, 255, 0), 2)
    
            # 输出结果
            a = 1
            out = cv2.resize(output2,(int(output.shape[1]*a), int(output.shape[0]*a)), interpolation=cv2.INTER_CUBIC)
            cv2.imshow('show', out)    
            k = cv2.waitKey(0)
            if k ==27:
                cv2.destroyAllWindows() 
    
    
    
    
    
    
    
    