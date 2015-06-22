import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

def trans_img(m1, u, v, th, ix=None, iy=None):
    SX, SY = m1.shape
    ret = np.zeros(m1.shape)
    cs = np.cos(th)
    sn = np.sin(th)
    if ix is None:
        ix = (np.arange(SX) * np.ones((SY, SX))).T.flatten().astype(int)
        iy = (np.arange(SY) * np.ones((SX, SY))).flatten().astype(int)
    iu = np.rint(cs * ix + sn * iy + u).astype(int)
    iv = np.rint(-sn * ix + cs * iy + v).astype(int)

    idx = np.arange(len(iu))
    idx = idx[(iu>=0)*(iu<SX)*(iv>=0)*(iv<SY)]
    ix = ix[idx]
    iy = iy[idx]
    iu = iu[idx]
    iv = iv[idx]
    ret[iu, iv] = m1[ix, iy]
    return ret.astype('uint8'), iu, iv

def pho_err(m1, m2, u, v, th):
    trans, iu, iv = trans_img(m1, u, v, th)
    cnt = len(iu)

    epsi = (trans - m2)[iu,iv]
    diff = epsi**2
    mean_err = diff.sum() / cnt
    return mean_err

def pho_err_grad(m1, m2, u, v, th, sobx, soby):
    pe = pho_err(m1, m2, u, v, th)
    DTH = 0.001
    gu = pho_err(m1, m2, u+1, v, th) - pe
    gv = pho_err(m1, m2, u, v+1, th) - pe
    gth = (pho_err(m1, m2, u, v, th+DTH) - pe) / DTH
    return gu, gv, gth

def rect_image(gray1, gray2):
    st = time.time()
    u, v, th = 0., 0., 0.
    for lay in range(7, -1, -1):
        u *= 2
        v *= 2
        g1 = gray1
        g2 = gray2
        for i in range(lay):
            g1 = cv2.pyrDown(g1)
            g2 = cv2.pyrDown(g2)
        bestu = 0
        bestv = 0
        bestval = 1E10
        for i in range(-2, 3):
            for j in range(-2, 3):
                for t in range(-2, 3):
                    pe = pho_err(g1, g2, i+u, j+v, t*1E-2+th)
                    if pe < bestval:
                        bestval = pe
                        bestu = i+u
                        bestv = j+v
                        bestth = t*1E-2+th
        print(bestu, bestv, bestth, bestval)
        u, v, th = bestu, bestv, bestth

    et = time.time() - st
    print("Time:", et)

    g1, _, _ = trans_img(gray1, u, v, th)
    g2 = gray2
    return g1, g2

def stereo_match(gray1, gray2):
    stereo = cv2.StereoBM_create(numDisparities=128, blockSize=21)
    # stereo = cv2.StereoSGBM_create(numDisparities=128, minDisparity=0, blockSize=21)
    disp = stereo.compute(gray2, gray1)
    return disp

if __name__ == '__main__':
    img1 = cv2.imread('../data/testimg/b001.jpg')
    img2 = cv2.imread('../data/testimg/b005.jpg')

    img1 = cv2.resize(img1, (480, 640), interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(img2, (480, 640), interpolation=cv2.INTER_CUBIC)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    gray1, gray2 = rect_image(gray1, gray2)

    sobx = cv2.Sobel(gray2, cv2.CV_8U, 1, 0)
    soby = cv2.Sobel(gray2, cv2.CV_8U, 0, 1)

    disp = stereo_match(gray1, gray2)

    plt.figure(1)
    # plt.imshow(abs(gray1-gray2), cmap='gray')
    plt.imshow(gray1, cmap='gray')
    # plt.imshow(img1[:,:,::-1], cmap = 'gray', interpolation = 'bicubic')
    # plt.scatter(p1[:, 0, 0], p1[:, 0, 1], s=err*2)

    plt.figure(2)
    plt.imshow(gray2, cmap='gray')
    # plt.imshow(img2[:,:,::-1], cmap = 'gray', interpolation = 'bicubic')
    # plt.scatter(p2[:, 0, 0], p2[:, 0, 1], s=err*2)

    plt.figure(3)
    plt.imshow(disp)

    plt.show()
