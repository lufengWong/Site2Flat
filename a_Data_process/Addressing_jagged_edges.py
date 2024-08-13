import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union

def plot_image(image, title="Image", cmap='gray'):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

# 加载图像
image = cv2.imread(r'img_5.png')
plot_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "Original Image")

# 转换到灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plot_image(gray, "Gray Image")

# 应用操作来减小矩形之间的间隙
kernel = np.ones((2, 2), np.uint8)
gray = cv2.dilate(gray, kernel, iterations=1)
plot_image(gray, "Dilated Image")

# # 应用操作来减小矩形之间的间隙
# kernel = np.ones((2, 2), np.uint8)
# gray = cv2.erode(gray, kernel, iterations=1)
# plot_image(gray, "Dilated Image")



# 应用阈值处理来分割图像
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
plot_image(thresh, "Threshold Image")

# 寻找轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 拟合直角矩形并将其转换为多边形对象
rectangles = []
for contour in contours:
    if len(contour) > 0:
        polygon = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        rect = cv2.minAreaRect(polygon)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        rectangles.append(Polygon(box))

        # 绘制每个多边形
        temp_image = image.copy()
        cv2.drawContours(temp_image, [polygon], 0, (0, 255, 0), 2)
        plot_image(cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB), "Fitted polygon")


        temp_image = image.copy()
        cv2.drawContours(temp_image, [box], 0, (0, 255, 0), 2)
        plot_image(cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB), "Fitted Rectangle")

# 使用 unary_union 合并所有矩形
# 使用缓冲区分析合并矩形

buffered =[]
for rect in rectangles:
    if rect.area <= 1:
        buffered.append(rect.buffer(2, join_style="mitre", cap_style='square'))
    else:
        buffered.append(rect)


# buffered = [rect.buffer(3,join_style="mitre", cap_style='square') for rect in rectangles]
merged_polygon = unary_union(buffered)

# 绘制合并后的结果
final_image = image.copy()
if merged_polygon:
    for index, poly in enumerate(merged_polygon.geoms):
        print(index)
        x, y = poly.exterior.xy
        plt.fill(x, y, alpha=0.5, fc='r', ec='none')

plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
plt.title('Merged Rectangles into Polygon')
plt.axis('off')
plt.show()
