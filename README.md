# RobotIC
yyt-2
对于第一张图片A的调参处理!!!
import cv2
import numpy as np
image = cv2.imread('/home/yangyuting/下载/A(1).jpg')

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red = np.array([5, 50, 15])
upper_red= np.array([20, 255, 255])

# 定义蓝色的HSV阈值范围
lower_blue = np.array([105, 150, 20])
upper_blue = np.array([160, 255, 255])

mask_red = cv2.inRange(hsv, lower_red, upper_red)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
cv2.imshow('mask_red',mask_blue)
cv2.waitKey(0)
# 形态学操作以去除噪声
kernel = np.ones((5, 5), np.uint8)
mask_red = cv2.erode(mask_red, kernel, iterations=2)
mask_red = cv2.dilate(mask_red, kernel, iterations=2)
mask_blue = cv2.erode(mask_blue, kernel, iterations=2)
mask_blue = cv2.dilate(mask_blue, kernel, iterations=2)

# 寻找轮廓
contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 绘制轮廓和标注
for contour in contours_red:
    if cv2.contourArea(contour) > 600:  # 过滤小轮廓
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(image, 'red', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

for contour in contours_blue:
    if cv2.contourArea(contour) > 200:  # 过滤小轮廓
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, 'Blue', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

red_count = 0
blue_count = 0

# 遍历轮廓
for contour in contours_red:
    # 检查颜色
    if cv2.contourArea(contour) > 600: 
        x, y, w, h = cv2.boundingRect(contour)
        if mask_red[y:y+h, x:x+w].sum() > 0:
            red_count += 1
for contour in contours_blue:
    # 检查颜色
        x, y, w, h = cv2.boundingRect(contour)
        if mask_blue[y:y+h, x:x+w].sum() > 0:
            blue_count += 1
# 显示结果
print(f"Red balls: {red_count}, Blue balls: {blue_count}")
cv2.imshow('Balls', image)
cv2.waitKey(0)
cv2.destroyAllWindows()






对于第二张图片B的调参处理!!!!
import cv2
import numpy as np


image = cv2.imread('/home/yangyuting/下载/B.jpg')
height,width,channels=image.shape
print(height,width,channels)
image = image[0: 640, 0: 500, :]

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red = np.array([3, 50, 15])
upper_red= np.array([20, 255, 255])

# 定义蓝色的HSV阈值范围
lower_blue = np.array([98, 150, 20])
upper_blue = np.array([130, 255, 255])

mask_red = cv2.inRange(hsv, lower_red, upper_red)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
cv2.imshow('mask_red',mask_blue)
cv2.waitKey(0)
# 形态学操作以去除噪声
kernel = np.ones((5, 5), np.uint8)
mask_red = cv2.erode(mask_red, kernel, iterations=2)
mask_red = cv2.dilate(mask_red, kernel, iterations=2)
mask_blue = cv2.erode(mask_blue, kernel, iterations=2)
mask_blue = cv2.dilate(mask_blue, kernel, iterations=2)

# 寻找轮廓
contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 绘制轮廓和标注
for contour in contours_red:
    if cv2.contourArea(contour) > 200:  # 过滤小轮廓
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(image, 'red', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

for contour in contours_blue:
    if cv2.contourArea(contour) > 150:  # 过滤小轮廓
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, 'Blue', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

red_count = 0
blue_count = 0

# 遍历轮廓
for contour in contours_red:
    # 检查颜色
    if cv2.contourArea(contour) > 600: 
        x, y, w, h = cv2.boundingRect(contour)
        if mask_red[y:y+h, x:x+w].sum() > 0:
            red_count += 1
for contour in contours_blue:
    # 检查颜色
    if cv2.contourArea(contour) > 300: 
        x, y, w, h = cv2.boundingRect(contour)
        if mask_blue[y:y+h, x:x+w].sum() > 0:
            blue_count += 1
# 显示结果
print(f"Red balls: {red_count}, Blue balls: {blue_count}")
cv2.imshow('Balls', image)
cv2.waitKey(0)
cv2.destroyAllWindows()




对于第三张图片C的调参处理!!!
import cv2
import numpy as np


image = cv2.imread('/home/yangyuting/下载/C.jpg')
height,width,channels=image.shape
print(height,width,channels)
image = image[100: 640, 20: 500, :]

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red = np.array([2, 15, 20])
upper_red= np.array([20, 255, 255])

# 定义蓝色的HSV阈值范围
lower_blue = np.array([98, 150, 20])
upper_blue = np.array([130, 255, 255])

mask_red = cv2.inRange(hsv, lower_red, upper_red)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
cv2.imshow('mask_red',mask_red)
cv2.waitKey(0)
# 形态学操作以去除噪声
kernel = np.ones((5, 5), np.uint8)
mask_red = cv2.erode(mask_red, kernel, iterations=2)
mask_red = cv2.dilate(mask_red, kernel, iterations=2)
mask_blue = cv2.erode(mask_blue, kernel, iterations=2)
mask_blue = cv2.dilate(mask_blue, kernel, iterations=2)

# 寻找轮廓
contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 绘制轮廓和标注
for contour in contours_red:
    if cv2.contourArea(contour) > 400:  # 过滤小轮廓
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(image, 'red', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

for contour in contours_blue:
    if cv2.contourArea(contour) > 200:  # 过滤小轮廓
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, 'Blue', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

red_count = 0
blue_count = 0

# 遍历轮廓
for contour in contours_red:
    # 检查颜色
    if cv2.contourArea(contour) > 400: 
        x, y, w, h = cv2.boundingRect(contour)
        if mask_red[y:y+h, x:x+w].sum() > 0:
            red_count += 1
for contour in contours_blue:
    # 检查颜色
    if cv2.contourArea(contour) > 300: 
        x, y, w, h = cv2.boundingRect(contour)
        if mask_blue[y:y+h, x:x+w].sum() > 0:
            blue_count += 1
# 显示结果
print(f"Red balls: {red_count}, Blue balls: {blue_count}")
cv2.imshow('Balls', image)
cv2.waitKey(0)
cv2.destroyAllWindows()






对于第四张图片color2的处理!!!!
import cv2
import numpy as np


image = cv2.imread('/home/yangyuting/下载/color2(1).jpg')
height,width,channels=image.shape
print(height,width,channels)
image = image[180:300, 300: 1100, :]

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red = np.array([160, 15, 50])
upper_red= np.array([190, 255, 255])

# 定义蓝色的HSV阈值范围
lower_blue = np.array([70, 150, 50])
upper_blue = np.array([110, 255, 255])
lower_purple = np.array([70, 40, 35])
upper_purple = np.array([160, 255, 255])

mask_red = cv2.inRange(hsv, lower_red, upper_red)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
# 形态学操作以去除噪声
kernel = np.ones((5, 5), np.uint8)
mask_red = cv2.erode(mask_red, kernel, iterations=2)
mask_red = cv2.dilate(mask_red, kernel, iterations=2)
mask_blue = cv2.erode(mask_blue, kernel, iterations=2)
mask_blue = cv2.dilate(mask_blue, kernel, iterations=2)
mask_purple = cv2.erode(mask_purple, kernel, iterations=2)
mask_purple = cv2.dilate(mask_purple, kernel, iterations=2)


# 寻找轮廓
contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_purple , _ = cv2.findContours(mask_purple , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 绘制轮廓和标注
for contour in contours_red:
    if cv2.contourArea(contour) >200:  # 过滤小轮廓
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(image, 'red', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

for contour in contours_blue:
    if cv2.contourArea(contour) > 200:  # 过滤小轮廓
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, 'Blue', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

for contour in contours_purple:
    if cv2.contourArea(contour) > 200:  # 过滤小轮廓
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, 'purple', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)


red_count = 0
blue_count = 0
purple_count=0
# 遍历轮廓
for contour in contours_red:
    # 检查颜色
    if cv2.contourArea(contour) > 150: 
        x, y, w, h = cv2.boundingRect(contour)
        if mask_red[y:y+h, x:x+w].sum() > 0:
            red_count += 1
red_count += 1
for contour in contours_blue:
    # 检查颜色
    if cv2.contourArea(contour) > 300: 
        x, y, w, h = cv2.boundingRect(contour)
        if mask_blue[y:y+h, x:x+w].sum() > 0:
            blue_count += 1
for contour in contours_purple:
    # 检查颜色
    if cv2.contourArea(contour) > 60: 
        x, y, w, h = cv2.boundingRect(contour)
        if mask_purple[y:y+h, x:x+w].sum() > 0:
           purple_count += 1
purple_count += 1
# 显示结果
print(f"Red balls: {red_count}, Blue balls: {blue_count},purple balls:{purple_count}")
cv2.imshow('Balls', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
