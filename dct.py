import numpy as np
import matplotlib.pyplot as plt


def normalize(img):
    vmin = np.min(img)
    vmax = np.max(img)
    return (img - vmin) / (vmax - vmin) * 255

def fft(x):
    n = len(x)
    if n == 2:
        return [x[0] + x[1], x[0] - x[1]]
    
    G = fft(x[::2])
    H = fft(x[1::2])
    W = np.exp(-2j * np.pi * np.arange(n//2) / n)
    WH = W * H
    X = np.concatenate([G + WH, G - WH])
    return X

def dct2(img):
    h, w = img.shape
    if ((h-1) & h) or ((w-1) & w):
        print('Image size not a power of 2')
        return img
    
    img = normalize(img)
    res = np.zeros([h, w], 'complex128')
    for i in range(h):
        res[i, :] = fft(np.concatenate([img[i, :], np.zeros(w)]))[:w]
        res[i, :] = np.real(res[i, :]) * np.sqrt(2 / w)
        res[i, 0] /= np.sqrt(2)
    for j in range(w):
        res[:, j] = fft(np.concatenate([res[:, j], np.zeros(h)]))[:h]
        res[:, j] = np.real(res[:, j]) * np.sqrt(2 / h)
        res[0, j] /= np.sqrt(2)
    return res


img = plt.imread('images/lena.bmp')
# img = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
plt.figure(figsize=(16,16))
plt.subplot(121), plt.imshow(img, 'gray'), plt.axis('off')

dct_img = dct2(img)
log_dct_img = np.log(1 + np.abs(dct_img))
plt.subplot(122), plt.imshow(log_dct_img, 'gray'), plt.axis('off')
plt.show()