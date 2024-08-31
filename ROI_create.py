import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import cv2 as cv
import Pre_process
import os
import warnings

warnings.filterwarnings('ignore')

def create_line(points):
    points = np.array(points)
    x_points, y_points = points[:, 0], points[:, 1]
    p = np.polyfit(y_points, x_points, 3)
    y_smooth = np.linspace(min(y_points), max(y_points), 1200)
    x_smooth = np.polyval(p, y_smooth)
   
    return x_smooth, y_smooth


def find_cls(image, H=40, W=50, Down_pixel=40, Detect_Width=10, reference_point=15):
    height, width = image.shape
    cls_points = []
    y = 0
    
    max_sum = -1
    max_point = None

    for j in range(width - W + 1):
        rect_sum = np.sum(image[y:y+H, j:j+W])
        if rect_sum > max_sum:
            max_sum = rect_sum
            max_point = (j + W//2, y)  # Save the top middle point

    cls_points.append(max_point)
    k = 0
    while k != reference_point:
        y = Down_pixel + y
        max_sum = -1
        max_point = None
        for m in range(max(0, cls_points[-1][0] - Detect_Width), min(width - W + 1, cls_points[-1][0] + Detect_Width)):
            left_top = m - W//2
            rect_sum = np.sum(image[y:y+H, left_top:left_top+W])
            if rect_sum > max_sum:
                max_sum = rect_sum
                max_point = (left_top + W//2, y)  # Save the top middle point
        if max_point is not None:
            cls_points.append(max_point)
            k = k + 1

    cls_points = np.array(cls_points)
    x_points, y_points = cls_points[:, 0], cls_points[:, 1]
    spline = make_interp_spline(y_points, x_points)
    y_smooth = np.linspace(min(y_points), max(y_points), 1200)
    x_smooth = spline(y_smooth)

    return x_smooth, y_smooth, cls_points


def get_normal_direction(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    normal = np.array([-dy, dx])
    norm = np.linalg.norm(normal)
    if norm == 0:
        normal = np.array([1, 0])
    else:
        normal = normal / norm
    return normal


def find_boundary(image, x_smooth, y_smooth, window_size=(11, 5), r=45, reserve=24.):
    H, W = image.shape
    left_boundary_points = []
    right_boundary_points = []

    half_window_height = window_size[0] // 2
    half_window_width = window_size[1] // 2
    for i in range(0, len(y_smooth)-1, 10):
        x1, y1 = int(x_smooth[i]), int(y_smooth[i])
        x2, y2 = int(x_smooth[i+1]), int(y_smooth[i+1])
        normal = get_normal_direction(x1, y1, x2, y2)
        max_diff = -np.inf
        best_boundary_point = None
        for d in range(-r, r+1):
            window1_center = (min(int(x1 + d * normal[0]), W), min(int(y1 + half_window_height * normal[1]), H))
            window2_center = (min(int(window1_center[0]+window_size[1] * normal[0]), W), min(window1_center[1], H))
            if (0 <= window1_center[0] < W and 0 <= window1_center[1] < H and
                0 <= window2_center[0] < W and 0 <= window2_center[1] < H):
                
                window1_intensity = np.mean(image[
                    max(0, window1_center[1] - half_window_height):min(H, window1_center[1] + half_window_height + 1),
                    max(0, window1_center[0] - half_window_width):min(W, window1_center[0] + half_window_width + 1)
                ])
                
                window2_intensity = np.mean(image[
                    max(0, window2_center[1] - half_window_height):min(H, window2_center[1] + half_window_height + 1),
                    max(0, window2_center[0] - half_window_width):min(W, window2_center[0] + half_window_width + 1)
                ])
                
                intensity_diff = abs(window1_intensity - window2_intensity)
                
                if intensity_diff > max_diff:
                    max_diff = intensity_diff
                    best_boundary_point = (window1_center[0]+half_window_width, window1_center[1])
        if best_boundary_point:
            if best_boundary_point[0] > (x1 + reserve):
                right_boundary_points.append(best_boundary_point)
            elif best_boundary_point[0] < (x1 - reserve):
                left_boundary_points.append(best_boundary_point)
    
    left_x, left_y = create_line(left_boundary_points)
    right_x, right_y = create_line(right_boundary_points)  
    return left_x, left_y, right_x, right_y


def make_region_mask(image, left_x, left_y, right_x, right_y, reserve=20):
    h, w = image.shape
    mask = np.zeros(image.shape, dtype=np.uint8)
    
    for i in range(len(left_x) - 1):
        x1, y1 = int(left_x[i]), int(left_y[i])
        x2, y2 = int(left_x[i + 1]), int(left_y[i + 1])

        if i == 0 and y1 != 0:
            mask[0:y1, 0:x1-reserve] = 255
        
        if 0 <= y1 < h and 0 <= y2 < h:
            if x1 != x2:
                slope = (y2 - y1) / (x2 - x1)
                for x in range(min(x1, x2), max(x1, x2) + 1):
                    y = int(y1 + slope * (x - x1))
                    mask[y, 0:x-reserve] = 255
            else:
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    mask[y, 0:x1-reserve] = 255
    end_px, end_py = int(left_x[-1]), int(left_y[-1])
    if end_py != h:
        mask[end_py:h, 0: end_px-reserve] = 255
                    
    for i in range(len(right_x) - 1):
        x1, y1 = int(right_x[i]), int(right_y[i])
        x2, y2 = int(right_x[i + 1]), int(right_y[i + 1])
        
        if i == 0 and y1 != 0:
            mask[0:y1, x1 + reserve:w] = 255

        if 0 <= y1 < h and 0 <= y2 < h:
            if x1 != x2:
                slope = (y2 - y1) / (x2 - x1)
                for x in range(min(x1, x2), max(x1, x2) + 1):
                    y = int(y1 + slope * (x - x1))
                    mask[y, x + reserve:w] = 255
            else:
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    mask[y, x1 + reserve:w] = 255
    end_px, end_py = int(right_x[-1]), int(right_y[-1])
    if end_py != h:
        mask[end_py:h, end_px + reserve: w] = 255

    return mask


def bin_threshold(image, bin=15):
    H, W = image.shape
    region = H // bin
    next_region = 0
    binary_image = np.zeros_like(image)
    for i in range(region):
        row_mean = image[next_region:next_region+bin-1, 0:W]
        row_mean = row_mean[row_mean > 0]
        row_mean = np.mean(row_mean)
        binary_image[next_region:next_region+bin, 0:W] = (image[next_region:next_region+bin, 0:W] > row_mean).astype(np.uint8) * 255
        next_region = next_region + bin

    return binary_image


def main():
    imgs_path = "./dataset/f03/image/"
    save_path = "./f03_roi/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for img in os.listdir(imgs_path):
        image = cv.imread(os.path.join(imgs_path, img), cv.IMREAD_GRAYSCALE)
        image = Pre_process.Gaussian_filter(image, sigmaX=2, ksize=(7, 7))
        image = cv.resize(image, (250, 600))
        image = Pre_process.cover_image(image)
        x_smooth, y_smooth, cls_points = find_cls(image)
        left_x, left_y, right_x, right_y = find_boundary(image, x_smooth, y_smooth)
        mask = make_region_mask(image, left_x, left_y, right_x, right_y)
        masked_image = np.where(mask == 255, 0, image)
        masked_image = Pre_process.contrast_stretch(masked_image)
        threshold_image = bin_threshold(Pre_process.sobel_operator(image))

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(x_smooth, y_smooth, color='red')
        plt.plot(left_x, left_y, color='green')
        plt.plot(right_x, right_y, color='yellow')
        plt.plot()
        plt.scatter(cls_points[:, 0], cls_points[:, 1], color='blue')
        plt.title("Original Image")
        plt.imshow(image, cmap='gray')
        
        plt.subplot(1, 3, 2)
        plt.title("Masked Image")
        plt.imshow(masked_image, cmap='gray')
        
        plt.subplot(1, 3, 3)
        plt.title("threshold Image")
        plt.imshow(threshold_image, cmap='gray')
        
        plt.show()
    
    
if __name__ == '__main__':
    main()
