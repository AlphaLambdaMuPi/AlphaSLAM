import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

from proc_image import rect_image, stereo_match, trans_img

cap0 = cv2.VideoCapture(0)
# cap0.set(3,1280)
# cap0.set(4,960)

cap1 = cv2.VideoCapture(1)
# cap1.set(3,1280)
# cap1.set(4,960)

cnt = 0
while True:
    ret, frame1 = cap1.read()
    ret, frame0 = cap0.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cnt += 1
    if cnt % 100 == 0:
        print(cnt)

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    for i in range(0):
        gray0 = cv2.pyrDown(gray0)
        gray1 = cv2.pyrDown(gray1)

    # stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)
    # stereo = cv2.StereoSGBM_create(numDisparities=128, minDisparity=0, blockSize=10)
    # disparity = stereo.compute(gray0, gray1)
    # cv2.applyColorMap(disparity, cv2.COLORMAP_JET, disparity)

    # disparity = disparity * 10

    # gray0, gray1 = rect_image(gray0, gray1)
    # gray0, _, _ = trans_img(gray0, -14, 0, 0.03)

    st = time.time()
    # bstval = 0
    # bsti = 0
    # disparity = -1
    # for i in range(-20, 24, 4):
        # gr = np.concatenate((gray1[i:,], gray1[:i,]), axis=0)
        # disp = stereo_match(gr, gray0)
        # val = (disp > 10).sum()
        # if val > bstval:
            # bstval, bsti, disparity = val, i, disp
    # print(bsti)

    disparity = stereo_match(gray1, gray0)
    et = time.time() - st
    print("Time:", et)

    mx = np.max(disparity)
    mn = np.min(disparity)
    display = ((disparity-mn)/(mx-mn)*255).astype('uint8')
    display = cv2.applyColorMap(display, cv2.COLORMAP_JET)

    cv2.imshow('frame1', gray1)
    cv2.imshow('frame0', gray0)
    # cv2.imshow('disp', abs(gray0-gray1))
    cv2.imshow('disp', display)

    # plt.imshow(disparity)
    # plt.show()

cap0.release()
cap1.release()
cv2.destroyAllWindows()
