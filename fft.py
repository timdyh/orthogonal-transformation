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
  
def fft2(img):
    h, w = img.shape
    if ((h-1) & h) or ((w-1) & w):
        print('Image size not a power of 2')
        return img
    
    img = normalize(img)
    res = np.zeros([h, w], 'complex128')
    for i in range(h):
        res[i, :] = fft(img[i, :])
    for j in range(w):
        res[:, j] = fft(res[:, j])
    return res

def fftshift(img):
    # swap the first and third quadrants, and the second and fourth quadrants
    h, w = img.shape
    h_mid, w_mid = h//2, w//2
    res = np.zeros([h, w], 'complex128')
    res[:h_mid, :w_mid] = img[h_mid:, w_mid:]
    res[:h_mid, w_mid:] = img[h_mid:, :w_mid]
    res[h_mid:, :w_mid] = img[:h_mid, w_mid:]
    res[h_mid:, w_mid:] = img[:h_mid, :w_mid]
    return res


img = plt.imread('images/lena.bmp')
# img = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
plt.figure(figsize=(16,16))
plt.subplot(121), plt.imshow(img, 'gray'), plt.axis('off')

fft_img = fft2(img)
shift_fft_img = np.abs(fftshift(fft_img))
log_shift_fft_img = np.log(1 + shift_fft_img)
plt.subplot(122), plt.imshow(log_shift_fft_img, 'gray'), plt.axis('off')
plt.show()