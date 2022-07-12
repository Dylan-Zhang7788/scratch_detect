import numpy as np
import cv2

dp = 1
param1 = 200
param2 = 135
min_radius=20

def resize_cv2(w_box,h_box,image):
    w=image.shape[1]
    h=image.shape[0]
    f1=1.0*w_box/w
    f2=1.0*h_box/h
    factor=min([f1,f2])
    width=int(w*factor)
    height=int(h*factor)
    return cv2.resize(image,(width,height),)

def closing_ellipse(img,kernel_size,iterations):
    copy_img=img.copy()
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kernel_size)
    img_close=cv2.morphologyEx(copy_img,cv2.MORPH_CLOSE,kernel,iterations=iterations)
    return img_close

# def draw_circle(img, values):
#     for i in values[0, :]:
#         cv2.circle(img, (int(i[0]), int(i[1])), int(i[2]), (0, 0, 255), 5)
#         cv2.circle(img, (int(i[0]), int(i[1])), 2, (0, 255, 0), 3)

def circle_extraction(img,min_dist,max_radius):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp, min_dist,
                              param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    
    radius=[]
    for i in circles[0,:]:
        radius.append(i[2])
    max_radius=np.argmax(np.array(radius))
    x=int(circles[0,max_radius,0])
    y=int(circles[0,max_radius,1])
    r=int(circles[0,max_radius,2])
    return x,y,r

def ring_extraction(origin_img):
    copy_img=origin_img.copy()
    equalized_img=cv2.equalizeHist(copy_img)
    gauss_img=cv2.GaussianBlur(equalized_img, (5, 5), 5)
    canny_img=cv2.Canny(gauss_img, 100, 200, apertureSize=3)
    closing_img=closing_ellipse(canny_img,(10,10),2)
    x1,y1,r1=circle_extraction(closing_img,30,2000)
    x2,y2,r2=circle_extraction(closing_img,15,int(r1)+5)

    ring_mask=np.zeros(origin_img.shape,np.uint8)
    cv2.circle(ring_mask,(x1,y1),int(r1)-15,(255,255,255),-1)
    cv2.circle(ring_mask,(x2,y2),int(r2)+15,(0,0,0),-1)
    ring=cv2.bitwise_and(ring_mask,equalized_img)
    # cv2.imshow('ring_img',resize_cv2(1000,1000,ring))
    # cv2.waitKey(0)
    return ring

def scratch_detect(img):
    canny_scratch = cv2.Canny(img, 45, 80, apertureSize=3)

    canny_scratch=closing_ellipse(canny_scratch,(10,10),2)
    contours,cnt=cv2.findContours(canny_scratch.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(img, contours, -1, (255, 255, 255), 2, 8)

    area=[]
    area0=[]
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    area0==sorted(area,reverse=True)[4]
    area=np.array(area)
    scratch_id=np.where(area==sorted(area,reverse=True)[4])[0].squeeze()
    rect=cv2.minAreaRect(contours[scratch_id])
    points=cv2.boxPoints(rect)
    points=np.int0(points)
    return rect,points

def main():
    original_image=cv2.imread("./homework_pic.jpg",cv2.IMREAD_GRAYSCALE)
    ring=ring_extraction(original_image)
    rect,points=scratch_detect(ring)
    image=cv2.cvtColor(original_image,cv2.COLOR_GRAY2BGR)
    image=cv2.drawContours(original_image,[points],0,(0,255,0),4)
    length_pixel=max(rect[1])
    print('划痕最小的外接矩形长度：',length_pixel)
    length=length_pixel*0.4
    print('划痕的实际长度： ',length)
    cv2.imshow('rect',resize_cv2(1000,1000,image))
    cv2.waitKey(0)

if __name__=='__main__':
    main()