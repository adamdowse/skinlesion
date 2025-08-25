# helper functions for py files
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.transform import rotate
from skimage.morphology import binary_dilation

from scipy.ndimage import shift,sobel
from scipy.optimize import curve_fit

import tensorflow as tf
import math
from tqdm import tqdm
import pandas as pd
import os

from numpy.lib.stride_tricks import as_strided

def view_as_blocks(arr_in, block_shape):
    """Block view of the input n-dimensional array (using re-striding).

    Blocks are non-overlapping views of the input array.

    Parameters
    ----------
    arr_in : ndarray, shape (M[, ...])
        Input array.
    block_shape : tuple
        The shape of the block. Each dimension must divide evenly into the
        corresponding dimensions of `arr_in`.

    Returns
    -------
    arr_out : ndarray
        Block view of the input array.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.util.shape import view_as_blocks
    >>> A = np.arange(4*4).reshape(4,4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> B = view_as_blocks(A, block_shape=(2, 2))
    >>> B[0, 0]
    array([[0, 1],
           [4, 5]])
    >>> B[0, 1]
    array([[2, 3],
           [6, 7]])
    >>> B[1, 0, 1, 1]
    13

    >>> A = np.arange(4*4*6).reshape(4,4,6)
    >>> A  # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0,  1,  2,  3,  4,  5],
            [ 6,  7,  8,  9, 10, 11],
            [12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23]],
           [[24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35],
            [36, 37, 38, 39, 40, 41],
            [42, 43, 44, 45, 46, 47]],
           [[48, 49, 50, 51, 52, 53],
            [54, 55, 56, 57, 58, 59],
            [60, 61, 62, 63, 64, 65],
            [66, 67, 68, 69, 70, 71]],
           [[72, 73, 74, 75, 76, 77],
            [78, 79, 80, 81, 82, 83],
            [84, 85, 86, 87, 88, 89],
            [90, 91, 92, 93, 94, 95]]])
    >>> B = view_as_blocks(A, block_shape=(1, 2, 2))
    >>> B.shape
    (4, 2, 3, 1, 2, 2)
    >>> B[2:, 0, 2]  # doctest: +NORMALIZE_WHITESPACE
    array([[[[52, 53],
             [58, 59]]],
           [[[76, 77],
             [82, 83]]]])
    """
    if not isinstance(block_shape, tuple):
        raise TypeError('block needs to be a tuple')

    block_shape = np.array(block_shape)
    if (block_shape <= 0).any():
        raise ValueError("'block_shape' elements must be strictly positive")

    if block_shape.size != arr_in.ndim:
        raise ValueError("'block_shape' must have the same length " "as 'arr_in.shape'")

    arr_shape = np.array(arr_in.shape)
    if (arr_shape % block_shape).sum() != 0:
        raise ValueError("'block_shape' is not compatible with 'arr_in'")

    # -- restride the array to build the block view
    new_shape = tuple(arr_shape // block_shape) + tuple(block_shape)
    new_strides = tuple(arr_in.strides * block_shape) + arr_in.strides

    arr_out = as_strided(arr_in, shape=new_shape, strides=new_strides)

    return arr_out

def asymmetryIoU(mask, n_angles=8, average=True,draw=False):
    #label each connected elements in the mask
    labeled_mask = label(mask)
    #get the properties of each labeled region
    props = regionprops(labeled_mask)
    #average the centers of the regions
    center = np.mean([prop.centroid for prop in props], axis=0)
    #transform the image so the center of the image is at the center of the mask
    image_center = np.array(mask.shape) / 2
    translation = image_center - center
    translated_mask = shift(mask, translation, order=0, mode='constant', cval=0)
    translated_mask = translated_mask > 0.5

    #for n angles between 0 and 180 degrees calculate the IoU for a flipped mask over the center and angle
    ious = []
    for angle in np.linspace(0, 180, n_angles, endpoint=False):
        rot_mask = rotate(translated_mask, angle, order=0, mode='constant', cval=0)
        rot_mask = rot_mask > 0.5
        flipped_mask = np.flipud(rot_mask)

        unison = np.logical_or(flipped_mask, rot_mask)
        overlap = np.logical_and(flipped_mask, rot_mask)
        ious.append(np.sum(overlap) / np.sum(unison) if np.sum(unison) > 0 else 0)
    #return the average IoU if average is True, otherwise return the IoUs
    if draw:
        fig, ax = plt.subplots(1,3, figsize=(15, 5))
        ax[0].imshow(mask, cmap='gray')
        ax[0].set_title('Original Mask')
        ax[0].scatter(center[1], center[0], color='red', label='Center', s=100)
        ax[0].legend()
        ax[1].imshow(translated_mask, cmap='gray')
        ax[1].set_title('Translated Mask')
        for angle_deg in np.linspace(0, 180, n_angles, endpoint=False):
            angle_rad = np.deg2rad(angle_deg)
            length = 500
            new_center = mask.shape[0] / 2, mask.shape[1] / 2

            dx = np.cos(angle_rad) * length / 2
            dy = np.sin(angle_rad) * length / 2

            x0, x1 = new_center[0] - dx, new_center[0] + dx
            y0, y1 = new_center[1] - dy, new_center[1] + dy
            ax[1].plot([y0, y1], [x0, x1], color='blue', linewidth=1, label=f'Angle {angle_deg:.1f}°')
        #ax[1].legend()
        ax[2].bar(np.linspace(0, 180, n_angles, endpoint=False), ious, width=10, color='orange', alpha=0.7)
        ax[2].set_title('IoU for each angle')
        ax[2].set_xlabel('Angle (degrees)')
        ax[2].set_ylabel('IoU')
        fig.savefig('/com.docker.devenvironments.code/SkinLesion/Graphs/asymmetry_iou.png', bbox_inches='tight')
    return np.mean(ious) if average else ious

def fastasymmetryIoU(mask, n_angles=8, average=True, draw=False):
    labeled_mask = label(mask)
    props = regionprops(labeled_mask)
    center = np.mean([prop.centroid for prop in props], axis=0)
    image_center = np.array(mask.shape) / 2
    translation = image_center - center
    translated_mask = shift(mask, translation, order=0, mode='constant', cval=0) > 0.5

    angles = np.linspace(0, 180, n_angles, endpoint=False)
    ious = []
    for angle in angles:
        rot_mask = rotate(translated_mask, angle, order=0, mode='constant', cval=0) > 0.5
        flipped_mask = np.flipud(rot_mask)
        unison = np.logical_or(flipped_mask, rot_mask)
        overlap = np.logical_and(flipped_mask, rot_mask)
        union_sum = np.sum(unison)
        ious.append(np.sum(overlap) / union_sum if union_sum > 0 else 0)

    return np.mean(ious) if average else ious

def asymmetryColour(mask, image, n_angles=8, average=True,draw=False):
    #label each connected elements in the mask
    labeled_mask = label(mask)
    #get the properties of each labeled region
    props = regionprops(labeled_mask)
    #average the centers of the regions
    center = np.mean([prop.centroid for prop in props], axis=0)
    #transform the image so the center of the image is at the center of the mask
    image_center = np.array(mask.shape) / 2
    translation = image_center - np.array(center)
    translated_mask = shift(mask, translation, order=0, mode='constant', cval=0)
    translated_mask = translated_mask > 0.5
    #translate each channel of the image
    translated_image = np.zeros_like(image)
    for i in range(image.shape[-1]):
        translated_image[..., i] = shift(image[..., i], translation, order=0, mode='constant', cval=0)

    #mask the image with the translated mask
    masked_image = translated_image * translated_mask[..., np.newaxis]

    #for n angles between 0 and 180 degrees calculate the IoU for a flipped mask over the center and angle
    col_averages = []
    for angle in np.linspace(0, 180, n_angles, endpoint=False):
        rot_image = rotate(masked_image, angle, order=0, mode='constant', cval=0)
        flipped_image = np.flipud(rot_image)
        #difference the flipped and rotated images
        diff_image = abs(flipped_image - rot_image)
        #calculate the mean of the difference image
        col_averages.append(np.mean(diff_image))
    
    if draw:
        fig,ax = plt.subplots(1,3, figsize=(15, 5))
        ax[0].imshow(mask, cmap='gray')
        ax[0].set_title('Original Mask')
        ax[0].scatter(center[1], center[0], color='red', label='Center', s=100)
        ax[0].legend()
        ax[1].imshow(image)
        ax[1].set_title('Image')
        ax[2].imshow(diff_image, cmap='gray')
        ax[2].set_title('Colour Difference Image for angle 0')
        #draw horizontal line representing flip
        ax[2].axhline(y=mask.shape[0] / 2, color='blue', linestyle='--', label='Flip Line')
        ax[2].legend()
        fig.savefig('/com.docker.devenvironments.code/SkinLesion/Graphs/asymmetry_col.png', bbox_inches='tight')
    #return the average colour difference if average is True, otherwise return the colour differences
    return np.mean(col_averages) if average else col_averages

def fastasymmetryColour(mask, image, n_angles=8, average=True, draw=False):
    labeled_mask = label(mask)
    props = regionprops(labeled_mask)
    center = np.mean([prop.centroid for prop in props], axis=0)
    image_center = np.array(mask.shape) / 2
    translation = image_center - center
    translated_mask = shift(mask, translation, order=0, mode='constant', cval=0) > 0.5
    translated_image = shift(image, (*translation, 0), order=0, mode='constant', cval=0)

    masked_image = translated_image * translated_mask[..., np.newaxis]
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    col_averages = []
    for angle in angles:
        rot_image = rotate(masked_image, angle, order=0, mode='constant', cval=0)
        flipped_image = np.flipud(rot_image)
        diff_image = np.abs(flipped_image - rot_image)
        col_averages.append(np.mean(diff_image))

    return np.mean(col_averages) if average else col_averages

def compactIndex(mask,draw=False):
    #label each connected elements in the mask
    labeled_mask = label(mask)
    #get the properties of each labeled region
    props = regionprops(labeled_mask)
    #calculate the area and perimeter of each region
    areas = np.sum([prop.area for prop in props])
    perimeters = np.sum([prop.perimeter for prop in props])
    #calculate the compactness index for each region
    compactness = (perimeters ** 2) / (4 * np.pi * areas)
    #return the mean compactness index
    if draw:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(mask, cmap='gray')
        ax.set_title('Compactness Index Mask')
        for prop in props:
            # Draw the perimeter of each region
            y, x = prop.coords[:, 0], prop.coords[:, 1]
            ax.plot(x, y, color='red', linewidth=1)
            # Draw the centroid of each region
            ax.scatter(prop.centroid[1], prop.centroid[0], color='blue', s=50, label='Centroid')
        ax.legend()
        fig.savefig('/com.docker.devenvironments.code/SkinLesion/Graphs/compactness_index.png', bbox_inches='tight')
    return compactness

def fastcompactIndex(mask, draw=False):
    labeled_mask = label(mask)
    props = regionprops(labeled_mask)
    areas = np.array([prop.area for prop in props])
    perimeters = np.array([prop.perimeter for prop in props])
    compactness = (perimeters ** 2) / (4 * np.pi * areas)
    return np.mean(compactness)

def fracalDimension(mask,draw=False):
    p = min(mask.shape)
    sizes = 2**np.arange(int(np.floor(np.log2(p))), 1, -1)

    counts = []
    for size in sizes:
        # Resize the image to make it evenly divisible by the box size
        new_shape = (mask.shape[0] // size * size,
                     mask.shape[1] // size * size)
        resized = mask[:new_shape[0], :new_shape[1]]

        # Split image into blocks
        blocks = view_as_blocks(resized, block_shape=(size, size))
        
        # Count non-empty boxes
        box_count = np.sum(np.any(np.any(blocks, axis=-1), axis=-1))
        counts.append(box_count)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    #print(coeffs)
    if draw:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(np.log(sizes), np.log(counts), label='Log-Log Plot')
        ax.plot(np.log(sizes), coeffs[0] * np.log(sizes) + coeffs[1], 'r--', label=f'Fit: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}')
        ax.set_xlabel('log(Box Size)')
        ax.set_ylabel('log(Count of Non-Empty Boxes)')
        ax.set_title('Fractal Dimension Calculation')
        ax.legend()
        fig.savefig('/com.docker.devenvironments.code/SkinLesion/Graphs/fractal_dimension.png', bbox_inches='tight')
    return -coeffs[0]

def fastfracalDimension(mask, draw=False):
    p = min(mask.shape)
    sizes = 2 ** np.arange(int(np.floor(np.log2(p))), 1, -1)
    counts = []
    for size in sizes:
        blocks = view_as_blocks(mask[:mask.shape[0] // size * size, :mask.shape[1] // size * size], block_shape=(size, size))
        counts.append(np.sum(np.any(blocks, axis=(2, 3))))
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def borderGradient(mask, image,draw=False):
    #using the mask find the border of the mask
    labeled_mask = label(mask)
    props = regionprops(labeled_mask)
    border_mask = np.zeros_like(mask, dtype=bool)
    for prop in props:
        # Get the coordinates of the border pixels
        coords = prop.coords
        # Create a border mask
        border_mask[coords[:, 0], coords[:, 1]] = True

        dilated_mask = binary_dilation(border_mask)
        border_mask = dilated_mask & ~mask  # Keep only the border pixels

    
    #make image greyscale
    if image.ndim == 3:
        grey_image = np.mean(image, axis=-1)
    else:
        grey_image = image
    
    #at each boarder pixel apply a sobel filter to the grey image and calulate the gradient magnitude
    sobel_x = sobel(grey_image, axis=0)
    sobel_y = sobel(grey_image, axis=1)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_direction = np.arctan2(sobel_y, sobel_x)
    #apply the border mask to the gradient magnitude
    border_gradient = gradient_magnitude[border_mask]
    border_angle = gradient_direction[border_mask]
    #return the mean gradient magnitude at the border pixels
    mean_border_gradient = np.mean(border_gradient) if border_gradient.size > 0 else 0
    mean_border_angle = np.mean(border_angle) if border_angle.size > 0 else 0
    if draw:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(mask, cmap='gray')
        ax[0].set_title('Mask with Border')
        ax[0].scatter(np.where(border_mask)[1], np.where(border_mask)[0], color='red', s=1, label='Border Pixels')
        ax[0].legend()
        
        ax[1].imshow(grey_image, cmap='gray')
        ax[1].set_title('Gradient Magnitude at Border')
        ax[1].scatter(np.where(border_mask)[1], np.where(border_mask)[0], c=border_gradient, cmap='hot', s=1)
        fig.colorbar(ax[1].collections[0], ax=ax[1], label='Gradient Magnitude')

        ax[2].imshow(grey_image, cmap='gray')
        ax[2].set_title('Gradient Direction at Border')
        ax[2].scatter(np.where(border_mask)[1], np.where(border_mask)[0], c=border_angle, cmap='hsv', s=1)
        fig.colorbar(ax[2].collections[0], ax=ax[2], label='Gradient Direction (radians)')
        
        fig.savefig('/com.docker.devenvironments.code/SkinLesion/Graphs/border_gradient.png', bbox_inches='tight')
    return mean_border_gradient, mean_border_angle

def fastborderGradient(mask, image, draw=False):
    border_mask = binary_dilation(mask) & ~mask
    grey_image = np.mean(image, axis=-1) if image.ndim == 3 else image
    sobel_x = sobel(grey_image, axis=0)
    sobel_y = sobel(grey_image, axis=1)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    border_gradient = gradient_magnitude[border_mask]
    return np.mean(border_gradient) if border_gradient.size > 0 else 0

def fitted_gaussian(mask, image, draw=False):
    def gaussian_2d(x, y, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
        """
        2D Gaussian function.
        """
        a = (np.cos(theta)**2) / (2*sigma_x**2) + (np.sin(theta)**2) / (2*sigma_y**2)
        b = -(np.sin(2*theta)) / (4*sigma_x**2) + (np.sin(2*theta)) / (4*sigma_y**2)
        c = (np.sin(theta)**2) / (2*sigma_x**2) + (np.cos(theta)**2) / (2*sigma_y**2)
        return offset + amplitude * np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))

    grey_image = np.mean(image, axis=-1) if image.ndim == 3 else image
    masked_image = grey_image[mask]
    #fit a gaussian to the masked image
    y,x = np.where(mask)
    z = masked_image
    amp_guess = np.max(z)
    x0_guess = np.mean(x)
    y0_guess = np.mean(y)
    sigma_x_guess = np.std(x)
    sigma_y_guess = np.std(y)
    theta_guess = 0.0  # No rotation for simplicity
    offset_guess = np.mean(z)
    initial_guess = [amp_guess, x0_guess, y0_guess, sigma_x_guess, sigma_y_guess, theta_guess, offset_guess]

    # Fit the Gaussian using curve_fit
    try:
        popt, _ = curve_fit(
            lambda coords, amplitude, x0, y0, sigma_x, sigma_y, theta, offset: gaussian_2d(
                coords[:, 0], coords[:, 1], amplitude, x0, y0, sigma_x, sigma_y, theta, offset
            ),
            np.column_stack((x, y)),
            z,
            p0=initial_guess
        )
    except RuntimeError:
        print("Gaussian fitting failed.")
        return None

    # Extract fitted parameters
    amplitude, x0, y0, sigma_x, sigma_y, theta, offset = popt

    if draw:
        # Generate the fitted Gaussian for visualization
        yy, xx = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
        fitted_gaussian = gaussian_2d(xx, yy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset)

        # Plot the original image, mask, and fitted Gaussian
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(grey_image, cmap='gray')
        ax[0].set_title('Original Image')
        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title('Mask')
        ax[2].imshow(fitted_gaussian, cmap='hot')
        ax[2].set_title('Fitted Gaussian')
        fig.savefig('/com.docker.devenvironments.code/SkinLesion/Graphs/fitted_gaussian.png', bbox_inches='tight')
    return popt


@tf.function
def process_image_tf(image, mask, n_angles=8):
    """
    Process an image and mask using TensorFlow operations.
    """
    # Ensure mask is binary
    mask = tf.cast(mask > 0.5, tf.float32)

    # Calculate the center of the mask
    mask_indices = tf.where(mask > 0)
    center = tf.reduce_mean(tf.cast(mask_indices, tf.float32), axis=0)

    # Translate the mask to center it
    image_shape = tf.shape(mask)
    image_center = tf.cast(image_shape[:2] / 2, tf.float32)
    translation = image_center - center
    translated_mask = tf.roll(mask, shifts=tf.cast(translation, tf.int32), axis=[0, 1])

    # Asymmetry IoU
    angles = tf.linspace(0.0, 180.0, n_angles)
    ious = []
    for angle in angles:
        radians = angle * (tf.constant(np.pi) / 180.0)
        rotated_mask = rotate_and_crop(rotated_mask, image_shape[0], image_shape[1], angle, do_crop=True)
        flipped_mask = tf.image.flip_up_down(rotated_mask)
        union = tf.logical_or(flipped_mask, rotated_mask)
        intersection = tf.logical_and(flipped_mask, rotated_mask)
        iou = tf.reduce_sum(tf.cast(intersection, tf.float32)) / tf.reduce_sum(tf.cast(union, tf.float32))
        ious.append(iou)
    asymmetry_iou = tf.reduce_mean(ious)

    # Compactness Index
    area = tf.reduce_sum(mask)
    perimeter = tf.reduce_sum(tf.image.sobel_edges(mask))
    compactness = (perimeter ** 2) / (4 * np.pi * area)

    # Fractal Dimension
    sizes = tf.cast(2 ** tf.range(tf.math.log(tf.cast(tf.reduce_min(image_shape[:2]), tf.float32)) / tf.math.log(2.0), 1, -1), tf.int32)
    counts = []
    for size in sizes:
        resized_mask = tf.image.resize(mask, [size, size], method='nearest')
        counts.append(tf.reduce_sum(tf.cast(resized_mask > 0, tf.float32)))
    log_sizes = tf.math.log(tf.cast(sizes, tf.float32))
    log_counts = tf.math.log(tf.cast(counts, tf.float32))
    fractal_coeffs = tf.linalg.lstsq(tf.expand_dims(log_sizes, axis=-1), tf.expand_dims(log_counts, axis=-1), fast=True)
    fractal_dimension = -fractal_coeffs[0, 0]

    # Border Gradient
    sobel_x = tf.image.sobel_edges(image)[..., 0]
    sobel_y = tf.image.sobel_edges(image)[..., 1]
    gradient_magnitude = tf.sqrt(sobel_x ** 2 + sobel_y ** 2)
    border_mask = tf.cast(tf.image.sobel_edges(mask) > 0, tf.float32)
    border_gradient = tf.reduce_mean(gradient_magnitude * border_mask)

    # Return all features
    return {
        "asymmetry_iou": asymmetry_iou,
        "compactness_index": compactness,
        "fractal_dimension": fractal_dimension,
        "border_gradient": border_gradient,
    }


def rotate_and_crop(image, output_height, output_width, rotation_degree, do_crop):
    """Rotate the given image with the given rotation degree and crop for the black edges if necessary
    Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        rotation_degree: The degree of rotation on the image.
        do_crop: Do cropping if it is True.
    Returns:
        A rotated image.
    """

    # Rotate the given image with the given rotation degree
    if rotation_degree != 0:
        image = tf.contrib.image.rotate(image, math.radians(rotation_degree), interpolation='BILINEAR')

        # Center crop to ommit black noise on the edges
        if do_crop == True:
            lrr_width, lrr_height = _largest_rotated_rect(output_height, output_width, math.radians(rotation_degree))
            resized_image = tf.image.central_crop(image, float(lrr_height)/output_height)    
            image = tf.image.resize_images(resized_image, [output_height, output_width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)

    return image

def _largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.
    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
    Converted to Python by Aaron Snoswell
    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )



def save_predictions_to_dataframe(model, seg_model, dataset, output_csv_path, threshold=0.5):
    """
    Save predictions from the model to a CSV file with image names.
    
    Args:
        model: The trained model for making predictions.
        dataset: The dataset containing images and their names.
        output_csv_path: Path to save the CSV file with predictions.
    """
    predictions = []
    for images, masks, image_names in tqdm(dataset, desc="Processing dataset"):

        if seg_model is not None:
            #pass the image through the segmentation model to get the mask
            masks = seg_model(images)
            #may have to threshold
            if threshold is not None:
                masks = tf.cast(masks > threshold, tf.float32)
        
        preds = model((images, masks))
        # Convert predictions to a flat list of dictionaries
        asymmetry_iou = preds['asymmetryIoU'].numpy()
        asymmetry_colour = preds['asymmetryColour'].numpy()
        compact_index = preds['compactIndex'].numpy()
        fractal_dimension = preds['fractalDimension'].numpy()
        border_gradient = preds['borderGradient'].numpy()
        for i in range(len(image_names)):
            predictions.append({
                "image_name": image_names[i].numpy().decode('utf-8'),
                "asymmetry_iou": asymmetry_iou[i],
                "asymmetry_colour": asymmetry_colour[i],
                "compact_index": compact_index[i],
                "fractal_dimension": fractal_dimension[i],
                "border_gradient": border_gradient[i]
            })
            
    
    df = pd.DataFrame(predictions)
    df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

def create_segmentation_dataset(data_dir, image_size=(576, 576), batch_size=2, train=True):
    if train:
        images_dir = os.path.join(data_dir, 'train')
    else:
        images_dir = os.path.join(data_dir, 'test')
    masks_dir = os.path.join(data_dir, 'SegmentationMaps')
    classes = tf.io.gfile.listdir(images_dir)
    image_paths = []
    mask_paths = []
    for c in classes:
        imgs = tf.io.gfile.glob(f"{images_dir}/{c}/*.jpg")
        for img in imgs:
            img_name = img.split('/')[-1].replace('.jpg', '')

            mask_path = os.path.join(masks_dir, f"{img_name}_segmentation.png")
            if tf.io.gfile.exists(mask_path):
                image_paths.append(img)
                mask_paths.append(mask_path)

    def _load_image_mask(img_path, mask_path):
        img = tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3)
        mask = tf.image.decode_png(tf.io.read_file(mask_path), channels=1)
        img = tf.image.resize(img, image_size)
        mask = tf.image.resize(mask, image_size)
        img = img / 255.0
        mask = tf.cast(mask > 0, tf.float32)
        #get the image name from the path
        img_name = tf.strings.split(img_path, '/')[-1]
        return img, mask, img_name

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(_load_image_mask, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

class FeatureExtractionModel(tf.keras.Model):
    def __init__(self, n_angles=8):
        super(FeatureExtractionModel, self).__init__()
        self.n_angles = n_angles

    def translate_images(self, images, translations):
        """Translates a batch of images using ImageProjectiveTransformV3."""
        batch_size = tf.shape(images)[0]
        dx = translations[:, 1]
        dy = translations[:, 0]
        transforms = tf.stack([
            tf.ones(batch_size, dtype=tf.float32), tf.zeros(batch_size, dtype=tf.float32), dx,
            tf.zeros(batch_size, dtype=tf.float32), tf.ones(batch_size, dtype=tf.float32), dy,
            tf.zeros(batch_size, dtype=tf.float32), tf.zeros(batch_size, dtype=tf.float32),
        ], axis=1)
        image_shape = tf.shape(images)
        height, width = image_shape[1], image_shape[2]
        return tf.raw_ops.ImageProjectiveTransformV3(
            images=images, transforms=transforms, output_shape=[height, width],
            interpolation="BILINEAR", fill_value=0.0)

    def rotate_image(self, image, angle):
        """Rotates a batch of images by a given angle (in degrees)."""
        angle_rad = angle * (np.pi / 180.0)
        cos_val = tf.cos(angle_rad)
        sin_val = tf.sin(angle_rad)
        transform = tf.convert_to_tensor([
            [cos_val, -sin_val, 0], [sin_val, cos_val, 0], [0, 0, 1]])
        transform = tf.reshape(transform, [-1])[:8]
        image_shape = tf.shape(image)
        height, width = image_shape[-3], image_shape[-2]
        return tf.raw_ops.ImageProjectiveTransformV3(
            images=image,
            transforms=tf.repeat(tf.expand_dims(transform, axis=0), tf.shape(image)[0], axis=0),
            output_shape=[height, width], interpolation="BILINEAR", fill_value=0.0)

    def asymmetryIoU(self, mask):
        mask = tf.cast(mask > 0.5, tf.float32)
        mask_indices = tf.where(mask > 0)
        row_ids = tf.cast(mask_indices[:, 0], tf.int32)
        values = tf.cast(mask_indices[:, 1:], tf.float32)
        ragged_indices = tf.RaggedTensor.from_value_rowids(values, row_ids)
        centers = tf.reduce_mean(ragged_indices.to_tensor(), axis=1)
        image_shape = tf.cast(tf.shape(mask)[1:3], tf.float32)
        image_center = image_shape / 2.0
        translation = image_center - centers
        translated_mask = self.translate_images(tf.expand_dims(mask, -1), translation)[:, :, :, 0]
        angles = tf.linspace(0.0, 180.0, self.n_angles)
        def compute_iou(angle):
            rotated_mask = self.rotate_image(tf.expand_dims(translated_mask, -1), angle)[:, :, :, 0]
            flipped_mask = tf.image.flip_up_down(rotated_mask)
            intersection = tf.reduce_sum(tf.cast(rotated_mask * flipped_mask > 0.5, tf.float32), axis=[1, 2])
            union = tf.reduce_sum(tf.cast((rotated_mask + flipped_mask) > 0.5, tf.float32), axis=[1, 2])
            return intersection / (union + 1e-6)
        ious = tf.map_fn(compute_iou, angles)
        return tf.reduce_mean(ious, axis=0)

    def asymmetryColour(self, mask, image):
        mask_expanded = tf.expand_dims(tf.cast(mask > 0.5, tf.float32), axis=-1)
        mask_indices = tf.where(mask > 0.5)
        row_ids = tf.cast(mask_indices[:, 0], tf.int32)
        values = tf.cast(mask_indices[:, 1:], tf.float32)
        ragged_indices = tf.RaggedTensor.from_value_rowids(values, row_ids)
        centers = tf.reduce_mean(ragged_indices.to_tensor(), axis=1)
        image_shape = tf.cast(tf.shape(mask)[1:3], tf.float32)
        image_center = image_shape / 2.0
        translation = image_center - centers
        translated_image = self.translate_images(image, translation)
        translated_mask = self.translate_images(mask_expanded, translation)
        masked_image = translated_image * translated_mask
        angles = tf.linspace(0.0, 180.0, self.n_angles)
        def compute_color_diff(angle):
            rotated_image = self.rotate_image(masked_image, angle)
            flipped_image = tf.image.flip_up_down(rotated_image)
            diff_image = tf.abs(flipped_image - rotated_image)
            return tf.reduce_mean(diff_image, axis=[1, 2, 3])
        col_averages = tf.map_fn(compute_color_diff, angles)
        return tf.reduce_mean(col_averages, axis=0)

    def compactIndex(self, mask):
        mask = tf.expand_dims(tf.cast(mask > 0.5, tf.float32), -1)
        area = tf.reduce_sum(mask, axis=[1, 2, 3])
        sobel_edges = tf.image.sobel_edges(mask)
        perimeter = tf.reduce_sum(tf.sqrt(sobel_edges[..., 0]**2 + sobel_edges[..., 1]**2), axis=[1, 2, 3])
        return (perimeter ** 2) / (4 * np.pi * area + 1e-6)

    def fracalDimension(self, mask):
        mask = tf.expand_dims(tf.cast(mask > 0.5, tf.float32), -1)
        p = tf.reduce_min(tf.shape(mask)[1:3])
        log_p = tf.math.log(tf.cast(p, tf.float32)) / tf.math.log(2.0)
        end_range = tf.maximum(0.0, log_p - 1)
        sizes = 2 ** tf.range(end_range, 0, -1)
        sizes = tf.cast(sizes, tf.int32)
        def count_boxes(size):
            resized_mask = tf.image.resize(mask, [size, size], method='nearest')
            return tf.reduce_sum(tf.cast(resized_mask > 0.5, tf.float32), axis=[1, 2, 3])
        counts = tf.map_fn(count_boxes, sizes, dtype=tf.float32)
        log_sizes = tf.math.log(tf.cast(sizes, tf.float32))
        log_counts = tf.math.log(tf.transpose(counts) + 1e-6)

        def linear_regression(x):
            y = tf.expand_dims(x, axis=1)
            A = tf.stack([log_sizes, tf.ones_like(log_sizes)], axis=1)
            solution = tf.linalg.lstsq(A, y)
            return solution[0, 0]

        return tf.vectorized_map(linear_regression, log_counts)

    def borderGradient(self, mask, image):
        mask = tf.expand_dims(tf.cast(mask > 0.5, tf.float32), -1)
        sobel_image = tf.image.sobel_edges(image)
        gradient_magnitude = tf.reduce_mean(tf.sqrt(sobel_image[..., 0]**2 + sobel_image[..., 1]**2), axis=-1, keepdims=True)
        sobel_mask = tf.image.sobel_edges(mask)
        border_mask = tf.sqrt(sobel_mask[..., 0]**2 + sobel_mask[..., 1]**2)
        border_gradient = tf.reduce_sum(gradient_magnitude * border_mask, axis=[1, 2, 3])
        border_length = tf.reduce_sum(border_mask, axis=[1, 2, 3])
        return border_gradient / (border_length + 1e-6)

    @tf.function
    def call(self, inputs):
        image, mask = inputs
        # FIX: Ensure mask is 3D by removing the channel dimension if it exists
        if len(mask.shape) == 4 and mask.shape[-1] == 1:
            mask = tf.squeeze(mask, axis=-1)

        asymmetry_iou = self.asymmetryIoU(mask)
        asymmetry_colour = self.asymmetryColour(mask, image)
        compact_index = self.compactIndex(mask)
        fractal_dimension = self.fracalDimension(mask)
        border_gradient = self.borderGradient(mask, image)
        return {
            "asymmetryIoU": asymmetry_iou,
            "asymmetryColour": asymmetry_colour,
            "compactIndex": compact_index,
            "fractalDimension": fractal_dimension,
            "borderGradient": border_gradient
        }

class SegmentationDataLoader(tf.keras.utils.Sequence):
    def __init__(self, root_dir, train=True, batch_size=32, image_size=(256, 256)):
        super().__init__()
        self.root_dir = root_dir
        if train:
            self.images_dir = root_dir+ '/train' #/com.docker.devenvironments.code/SkinLesion/Data/HAM10000/train
        else:
            self.images_dir = root_dir + '/test'
        self.masks_dir = root_dir + '/SegmentationMaps' #/com.docker.devenvironments.code/SkinLesion/Data/HAM10000/SegmentationMaps
        self.batch_size = batch_size
        self.image_size = image_size
        self.classes = tf.io.gfile.listdir(self.images_dir)
        self.image_filenames = []
        for c in self.classes:
            # Collect all image filenames for each class
            self.image_filenames.extend(tf.io.gfile.glob(f"{self.images_dir}/{c}/*.jpg"))
        print(f"Found {len(self.image_filenames)} images in {self.images_dir}")
        #ensure a mask exists for each image
        c = 0
        for img in self.image_filenames:
            img_name = img.split('/')[-1].replace('.jpg', '_segmentation.png')
            mask_path = f"{self.masks_dir}/{img_name}"
            if tf.io.gfile.exists(mask_path):
                c += 1
            else:
                print(f"Mask not found for image: {img_name}")

        assert len(self.image_filenames) == c, "Number of images and masks must match"

        # shuffle the image filenames
        self.image_filenames = tf.random.shuffle(self.image_filenames)

    def __len__(self):
        return len(self.image_filenames) // self.batch_size

    def __getitem__(self, idx):
        batch_images = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_masks = tf.strings.regex_replace(batch_images, r'/HAM10000/.*/', '/HAM10000/SegmentationMaps/')
        batch_masks = tf.strings.regex_replace(batch_masks, r'\.jpg$', '_segmentation.png')

        images = [tf.image.decode_jpeg(tf.io.read_file(img), channels=3) for img in batch_images]
        masks = [tf.image.decode_png(tf.io.read_file(mask), channels=1) for mask in batch_masks]

        images = [tf.image.resize(img, self.image_size) for img in images]
        masks = [tf.image.resize(mask, self.image_size) for mask in masks]

        # Normalize images to [0, 1] range
        images = [img / 255.0 for img in images]
        # Convert masks to binary (0 or 1)
        masks = [tf.cast(mask > 0, tf.float32) for mask in masks]

        images = tf.stack(images)
        masks = tf.stack(masks)

        return images, masks
    
    def on_epoch_end(self):
        # Shuffle the data at the end of each epoch
        self.image_filenames = tf.random.shuffle(self.image_filenames)


if __name__ == "__main__":
    # Example usage
    # mask = np.array([[0, 0, 1, 1, 0],
    #                  [0, 0, 1, 1, 0],
    #                  [1, 0, 1, 0, 0],
    #                  [0, 1, 0, 0, 0],
    #                  [0, 0, 0, 0, 0]])  # Example binary
    
    # image = np.random.rand(5, 5, 3)  # Random color image

    image_id = "ISIC_0024324.jpg"
    #image_id = "ISIC_0034320.jpg"  # Example image ID
    image = plt.imread(f"/com.docker.devenvironments.code/SkinLesion/Data/HAM10000/Images/{image_id}")
    mask = plt.imread(f"/com.docker.devenvironments.code/SkinLesion/Data/HAM10000/SegmentationMaps/{image_id.split('.')[0]}_segmentation.png")
    if len(mask.shape) != 2:
        # If the mask has an alpha channel or is RGB, convert it to grayscale
        if mask.shape[-1] == 4:
            mask = mask[:, :, 0]
        elif mask.shape[-1] == 3:
            mask = np.mean(mask, axis=-1)
    mask = mask > 0.5  # Convert to binary mask
    print("image type:", type(image))
    print("mask type:", type(mask))
    print("image shape:", image.shape)
    print("mask shape:", mask.shape)

    draw = True
    print("Asymmetry IoU:", asymmetryIoU(mask,draw=draw))
    print("Asymmetry Colour:", asymmetryColour(mask, image,draw=draw))
    print("Compact Index:", compactIndex(mask,draw=draw))
    print("Fractal Dimension:", fracalDimension(mask,draw=draw))
    print("Border Gradient:", borderGradient(mask, image,draw=draw))
    print("Fitted Gaussian Parameters:", fitted_gaussian(mask, image, draw=draw))