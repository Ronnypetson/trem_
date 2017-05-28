import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import random
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
                cnt += 1;
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
        if wi > 0:
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
        if random.randint(0,l-1) < m:
            r[ind] = d[ind2]
            ind += 1
        ind2 += 1
    return r[0:ind]

def get_all_densities_normalized(fnames,m):
    r = []
    for fname in fnames:
        r.append(normalize_density(get_density(fname),m))
    return r

def plot_densities(d,names,ofile):
    n = len(d)
    if n%3 == 0:
        lin = n/3
    else:
        lin = (n/3)+1
    col = 3
    plt.xlabel('row')
    plt.ylabel('pixel density')    
    for i in range(n):
        x = np.arange(0,len(d[i]),1)
        plt.subplot(lin,col,i+1)
        plt.plot(x,d[i])
        plt.title(names[i])
    plt.savefig(ofile)
    plt.show()

sue_letters = '../Sue/trem/l/*.jpg'
letter_files = 'trem/l/*.jpg'
fnames = glob.glob(letter_files)
snames = glob.glob(sue_letters)
#
min_len = count_min_width(fnames)
min_sue = count_min_width(snames)
#
densities = get_all_densities_normalized(fnames,min_len)
sdensities = get_all_densities_normalized(snames,min_sue)
#
ofl = 'graphs.jpg'
sfl = 'sue_graphs.jpg'
plot_densities(densities,fnames,ofl)
plot_densities(sdensities,snames,sfl)

