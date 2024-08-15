import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import cv2 as cv
import my_DEBUG
import Pre_process


def remove_duplicates(x, y):
    unique_y, indices = np.unique(y, return_index=True)
    unique_x = [np.mean(x[y == y_value]) for y_value in unique_y]
    return np.array(unique_x), unique_y


def find_cls(image, H, W, Down_pixel, Detect_Width, main_point, follow_point):
    height, width = image.shape
    cls_points = []
    y = 0
    for i in range(main_point):
        max_sum = -np.inf
        max_point = None

        for j in range(width - W + 1):
            rect_sum = np.sum(image[y:y+H, j:j+W])
            if rect_sum > max_sum:
                max_sum = rect_sum
                max_point = (j + W//2, y)  # Save the top middle point

        cls_points.append(max_point)
            
            # Move down p pixels
        for k in range(follow_point):
            y += Down_pixel

            max_sum = -np.inf
            max_point = None
            for m in range(max(0, cls_points[0][0] - Detect_Width), min(width - W + 1, cls_points[0][0] + Detect_Width)):
                rect_sum = np.sum(image[y:y+H, k:k+W])
                if rect_sum > max_sum:
                    max_sum = rect_sum
                    max_point = (m + W//2, y)  # Save the top middle point

                # Save the second reference point
            cls_points.append(max_point)

        # Move down p pixels again for the next iteration

    # Polynomial fitting
    cls_points = np.array(cls_points)
    x_points, y_points = cls_points[:, 0], cls_points[:, 1]
    x_points, y_points = remove_duplicates(x_points, y_points)
    
    # Use spline fitting for a smooth curve
    spline = make_interp_spline(y_points, x_points)
    y_smooth = np.linspace(min(y_points), max(y_points), 1200)
    x_smooth = spline(y_smooth)

    return x_smooth, y_smooth, cls_points


def make_region_mask(image, x_smooth, y_smooth, range_extend=50):
    # 创建一个与原图像相同大小的黑色掩膜
    h, w = image.shape
    mask = np.zeros(image.shape, dtype=np.uint8)
    
    # 遍历 CLS 曲线上的每一个点
    for i in range(len(x_smooth) - 1):
        x1, y1 = int(x_smooth[i]), int(y_smooth[i])
        x2, y2 = int(x_smooth[i + 1]), int(y_smooth[i + 1])

        # 确保 x 和 y 不超出图像边界
        if 0 <= y1 < h and 0 <= y2 < h:
            # 计算线段之间的斜率
            if x1 != x2:
                slope = (y2 - y1) / (x2 - x1)
                for x in range(min(x1, x2), max(x1, x2) + 1):
                    y = int(y1 + slope * (x - x1))
                    x_min = max(x - range_extend, 0)
                    x_max = min(x + range_extend, w - 1)
                    mask[y, x_min:x_max + 1] = 255
            else:
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    x_min = max(x1 - range_extend, 0)
                    x_max = min(x1 + range_extend, w - 1)
                    mask[y, x_min:x_max + 1] = 255

    return mask


def main()
    image = cv.imread("./dataset/f01/image/0001.png", cv.IMREAD_GRAYSCALE)
    # image = cv.resize(image, (250, 600))
    image = Pre_process.Gaussian(image)
    image = Pre_process.sobel_operator(image)
    image = Pre_process.contrast_stretch(image)
    H, W = 50, 100  # Rectangle size
    Down_pixel = 50  # Vertical step size
    Detect_Width = 50  # Search interval
    main_point = 6  # Number of reference points
    follow_point = 4
    x_smooth, y_smooth, cls_points = find_cls(image, H, W, Down_pixel, Detect_Width, main_point, follow_point)
    mask = make_region_mask(image, x_smooth, y_smooth, 70)
    masked_image = np.where(mask == 255, image, 0)
    # Plotting the results
    plt.imshow(image, cmap='gray')
    plt.plot(x_smooth, y_smooth, color='red', label='CLS')
    plt.scatter(cls_points[:, 0], cls_points[:, 1], color='blue', label='Reference Points')
    plt.legend()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Masked Image")
    plt.imshow(masked_image, cmap='gray')

    plt.show()
    
if __name__ == '__main__':
    main()
