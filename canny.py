# =========================================================================================
import numpy as np
import cv2
from matplotlib import pyplot as plt
# =========================================================================================
CANNY_LOWER = 255
CANNY_UPPER = 255
# =========================================================================================
img = cv2.imread('trem/h/24.jpg')
#img = cv2.imread('trem/l/01.jpg')
height, width, _ = img.shape
# =========================================================================================
canny = cv2.Canny(img,CANNY_LOWER,CANNY_UPPER)
canny = cv2.dilate(canny, None)
canny = cv2.erode(canny, None)
# =========================================================================================
rst = np.zeros(img.shape,np.uint8)
ulr = np.zeros([height, width],np.uint8)
uld = np.zeros([height, width],np.uint8)
brl = np.zeros([height, width],np.uint8)
bru = np.zeros([height, width],np.uint8)
diag = np.zeros([height, width],np.uint8)
diag2 = np.zeros([height, width],np.uint8)
# =========================================================================================
for i in range(height):				# Upper-left to right
	acc = 0
	prev = 0
	for j in range(width):
		if(canny[i,j] == 0):
			if prev == 255:
				acc += 1	
			ulr[i,j] = acc
		else:
			ulr[i,j] = 0
		prev = canny[i,j]

for i in range(width):				# Upper-left downwards
	acc = 0
	prev = 0
	for j in range(height):
		if(canny[j,i] == 0):
			if prev == 255:
				acc += 1	
			uld[j,i] = acc
		else:
			uld[j,i] = 0
		prev = canny[j,i]

for i in range(height):				# Bottom-right to left
	acc = 0
	prev = 0
	for j in range(width):
		if(canny[height-i-1,width-j-1] == 0):
			if prev == 255:
				acc += 1	
			brl[height-i-1,width-j-1] = acc
		else:
			brl[height-i-1,width-j-1] = 0
		prev = canny[height-i-1,width-j-1]

for i in range(width):				# Bottom-right upwards
	acc = 0
	prev = 0
	for j in range(height):
		if(canny[height-j-1,width-i-1] == 0):
			if prev == 255:
				acc += 1
			bru[height-j-1,width-i-1] = acc
		else:
			bru[height-j-1,width-i-1] = 0
		prev = canny[height-j-1,width-i-1]

for i in range(height-width):		#Upper-left to bottom-right diagonal
	acc = 0
	prev = 0
	ii = i
	for j in range(width):
		if(canny[ii,j] == 0):
			if prev == 255:
				acc += 1
			diag[ii,j] = acc
		else:
			diag[ii,j] = 0
		prev = canny[ii,j]
		ii += 1

for i in range(height-width):		#Upper-right to bottom-left diagonal
	acc = 0
	prev = 0
	ii = i
	for j in range(width):
		ij = width-j-1
		if(canny[ii,ij] == 0):
			if prev == 255:
				acc += 1	
			diag2[ii,ij] = acc
		else:
			diag2[ii,ij] = 0
		prev = canny[ii,ij]
		ii += 1

def probably_inside(i,j):
	return (ulr[i,j]%2+brl[i,j]%2+uld[i,j]%2+bru[i,j]%2+diag[i,j]%2+diag2[i,j]%2) >= 3
	#+diag[i,j]%2+diag2[i,j]%2

# =========================================================================================
for i in range(height):
	for j in range(width):
		if canny[i,j] == 255:	#contorno
			rst[i,j] = img[i,j]
		else:
			if probably_inside(i,j):
				rst[i,j] = img[i,j]
			else:
				rst[i,j] = [255,255,255]
# =========================================================================================
#plt.subplot(121),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(121),plt.imshow(cv2.cvtColor(rst, cv2.COLOR_BGR2RGB))
plt.subplot(122),plt.imshow(canny)
plt.show()

