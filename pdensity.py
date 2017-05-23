import numpy as np
import cv2
import matplotlib.pyplot as plt
#=====================
WHITE_THRESH = 255
#=====================
def count_width(fname):
    img = cv2.imread(fname)  # Load file
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Set to grayscale

    h,w = img.shape
    cnt = 0  # Row counter (blank rows are ignored)
    for i in range(h):  # height
        for j in range(w):
            if img[i,j] < WHITE_THRESH:
                cnt++;
                break
    return cnt

def count_min_width(fnames):
    mini = 999999
    for fname in fnames:
        w = count_width(fname)
        if mini > w:
            mini = w
    return mini

def get_density(fname):
    img = cv2.imread(fname)  # Load file
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Set to grayscale

    h,w = img.shape
    ind = 0  # Row counter (blank rows are ignored)
    d = np.zeros((h))  # Setup initial density to 0.0
    for i in range(h):  # height
        wi = 0  # row length counter
        for j in range(w):  # width
            if img[i,j] < WHITE_THRESH:  # Ignore white background
                d[ind] += img[i,j]
                wi += 1
        if wi != 0:
            d[ind] /= wi
            ind += 1
    return d[0:ind+1]  # <--

def normalize_density(d,m):
    l = len(d)
    if l <= m:
        return d

    r = np.zeros(l)
    ind = 0
    ind2 = 0
    while ind < m and ind2 < l:
        if random.randint() % l < m:
            r[ind] = d[ind2]
            ind += 1
        ind2 += 1
    return r

print(get_density('01f.jpg'))

