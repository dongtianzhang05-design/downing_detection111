import cv2

# 读取彩色图片
img = cv2.imread('VOCdevkit/test/images/000014.jpg')

# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 保存灰度图
cv2.imwrite('000014_gray.jpg', gray)

print("灰度图已保存为 VOCdevkit/test/images/000014_gray.jpg")
