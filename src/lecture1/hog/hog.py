"""
Created on July 22 00:28, 2020

@author: fassial
"""
import math
import numpy as np
# local dep
import utils

"""
_hog_channel_gradient:
    compute unnormalized gradient image along `row` and `col` axes
    @params:
        channel(np.array)   : grayscale image or one of image channel
    @rets:
        g_row(np.array)     : channel gradient along `row` axes
        g_col(np.array)     : channel gradient along `col` axes
"""
def _hog_channel_gradient(channel):
    g_row = np.empty(channel.shape, dtype=np.double)
    g_row[0, :] = 0
    g_row[-1, :] = 0
    g_row[1:-1, :] = channel[2:, :] - channel[:-2, :]
    g_col = np.empty(channel.shape, dtype=np.double)
    g_col[:, 0] = 0
    g_col[:, -1] = 0
    g_col[:, 1:-1] = channel[:, 2:] - channel[:, :-2]

    return g_row, g_col

"""
_hog_histograms:
    get cell non-normalized gradient histogram
    @params:
        g_row(np.array)     : channel gradient along `row` axes
        g_col(np.array)     : channel gradient along `col` axes
        c_row(int)          : number of pixels in a cell along row-axis
        c_col(int)          : number of pixels in a cell along col-axis
        orientations(int)   : number of histogram horizontal axis items
    @rets:
        orientation_histogram(np.array) : cell non-normalized gradient histogram
"""
def _hog_histograms(g_row, g_col, c_row, c_col, orientations):
    assert g_col.shape == g_row.shape
    # set local vars
    s_row, s_col = g_col.shape[:2]
    n_cells_row = int(s_row // c_row)  # number of cells along row-axis
    n_cells_col = int(s_col // c_col)  # number of cells along col-axis
    angle_unit = math.pi / orientations
    # init orientation_histogram
    orientation_histogram = np.zeros((n_cells_row, n_cells_col, orientations))

    # compute orientations integral images
    # row: y, col: x
    for i in range(n_cells_row):
        for j in range(n_cells_col):
            for ii in range(c_row):
                for jj in range(c_col):
                    # get index
                    idx_i, idx_j = (i * c_row + ii), (j * c_col + jj)
                    # get arctan value & grad value
                    angle_tan = (g_row[idx_i, idx_j] / g_col[idx_i, idx_j]) if g_col[idx_i, idx_j] != 0 else math.inf
                    angle = math.atan(angle_tan)
                    grad = math.sqrt(pow(g_row[idx_i, idx_j], 2) + pow(g_col[idx_i, idx_j], 2))
                    # set orientation_histogram
                    for k in range(orientations):
                        if (angle >= k * angle_unit) and (angle < (k + 1) * angle_unit):
                            # find corresponding orientation
                            # 1. the hog image is hard to see
                            # orientation_histogram[i, j, k] += grad / (c_col * c_row)
                            # 2. the gradient is not spread out in the adjacent directions
                            # orientation_histogram[i, j, k] += grad
                            # 3. the gradient is spread out in the adjacent directions
                            # orientation_histogram[i, j, k] += grad * (1 - (angle - k * angle_unit) / angle_unit)
                            # orientation_histogram[i, j, (k+1)%orientations] += grad * ((angle - k * angle_unit) / angle_unit)
                            # 4. consider the distance from the center
                            G = utils.bilinear_func(
                                x = (jj + 0.5) - (c_col / 2),
                                y = (ii + 0.5) - (c_row / 2),
                                W = c_col,
                                H = c_row
                            )
                            orientation_histogram[i, j, k] += G * grad * (1 - (angle - k * angle_unit) / angle_unit)
                            orientation_histogram[i, j, (k+1)%orientations] += G * grad * ((angle - k * angle_unit) / angle_unit)

    # return res
    return orientation_histogram

"""
_hog_normalize_block:
    get block normalized gradient histogram
    #block normalization method:
        ``L1``
            Normalization using L1-norm.
        ``L1-sqrt``
            Normalization using L1-norm, followed by square root.
        ``L2``
            Normalization using L2-norm.
        ``L2-Hys``
            Normalization using L2-norm, followed by limiting the
            maximum values to 0.2 (`Hys` stands for `hysteresis`) and
            renormalization using L2-norm. (default)
            For details, see [3]_, [4]_.
    @params:
        block(np.array) : block non-normalized gradient histogram
        method(string)  : normalization method
        eps(int)        : epsilon(avoid dividing by 0)
    @rets:
        out(np.array)   : block normalized gradient histogram
"""
def _hog_normalize_block(block, method, eps=1e-5):
    if method == 'L1':
        out = block / (np.sum(np.abs(block)) + eps)
    elif method == 'L1-sqrt':
        out = np.sqrt(block / (np.sum(np.abs(block)) + eps))
    elif method == 'L2':
        out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
    elif method == 'L2-Hys':
        out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
        out = np.minimum(out, 0.2)
        out = out / np.sqrt(np.sum(out ** 2) + eps ** 2)
    else:
        raise ValueError('Selected block normalization method is invalid.')

    return out

"""
hog:
    get hog features of image
    @params:
        image(np.array)         : original image
        orientations(int)       : number of histogram horizontal axis items
        pixels_per_cell(tuple)  : pixels per cell along (row, col)-axis
        cells_per_block(tuple)  : cells per block along (row, col)-axis
        block_norm(string)      : normalization method of block non-normalized gradient histogram
        visualize(bool)         : whether to visualize the hog feature
        transform_sqrt(bool)    : whether apply power law compression to normalize the image before processing
        feature_vector(bool)    : whether return the data as a feature vector
        multichannel(bool)      : if True, the last `image` dimension is considered as a color channel, otherwise as spatial.
    @rets:
        normalized_blocks(np.array) : hog descriptor for the image
        hog_image(np.array)         : a visualisation of the HOG image
"""
def hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
        block_norm='L2-Hys', visualize=False, transform_sqrt=False,
        feature_vector=True, multichannel=None):
    """
    Notes
    -----
    The presented code implements the HOG extraction method from [2]_ with
    the following changes: (I) blocks of (3, 3) cells are used ((2, 2) in the
    paper); (II) no smoothing within cells (Gaussian spatial window with sigma=8pix
    in the paper); (III) L1 block normalization is used (L2-Hys in the paper).

    Power law compression, also known as Gamma correction, is used to reduce
    the effects of shadowing and illumination variations. The compression makes
    the dark regions lighter. When the kwarg `transform_sqrt` is set to
    ``True``, the function computes the square root of each color channel
    and then applies the hog algorithm to the image.
    """
    image = np.atleast_2d(image)

    if multichannel is None:
        multichannel = (image.ndim == 3)

    ndim_spatial = image.ndim - 1 if multichannel else image.ndim
    if ndim_spatial != 2:
        raise ValueError('Only images with 2 spatial dimensions are '
                         'supported. If using with color/multichannel '
                         'images, specify `multichannel=True`.')

    """
    The first stage applies an optional global image normalization
    equalisation that is designed to reduce the influence of illumination
    effects. In practice we use gamma (power law) compression, either
    computing the square root or the log of each color channel.
    Image texture strength is typically proportional to the local surface
    illumination so this compression helps to reduce the effects of local
    shadowing and illumination variations.
    """

    if transform_sqrt:
        image = np.sqrt(image)

    """
    The second stage computes first order image gradients. These capture
    contour, silhouette and some texture information, while providing
    further resistance to illumination variations. The locally dominant
    color channel is used, which provides color invariance to a large
    extent. Variant methods may also include second order image derivatives,
    which act as primitive bar detectors - a useful feature for capturing,
    e.g. bar like structures in bicycles and limbs in humans.
    """

    if image.dtype.kind == 'u':
        # convert uint image to float
        # to avoid problems with subtracting unsigned numbers
        image = image.astype('float')

    if multichannel:
        g_row_by_ch = np.empty_like(image, dtype=np.double)
        g_col_by_ch = np.empty_like(image, dtype=np.double)
        g_magn = np.empty_like(image, dtype=np.double)

        for idx_ch in range(image.shape[2]):
            g_row_by_ch[:, :, idx_ch], g_col_by_ch[:, :, idx_ch] = \
                _hog_channel_gradient(image[:, :, idx_ch])
            g_magn[:, :, idx_ch] = np.hypot(g_row_by_ch[:, :, idx_ch],
                                            g_col_by_ch[:, :, idx_ch])

        # For each pixel select the channel with the highest gradient magnitude
        idcs_max = g_magn.argmax(axis=2)
        rr, cc = np.meshgrid(np.arange(image.shape[0]),
                             np.arange(image.shape[1]),
                             indexing='ij',
                             sparse=True)
        g_row = g_row_by_ch[rr, cc, idcs_max]
        g_col = g_col_by_ch[rr, cc, idcs_max]
    else:
        g_row, g_col = _hog_channel_gradient(image)

    """
    The third stage aims to produce an encoding that is sensitive to
    local image content while remaining resistant to small changes in
    pose or appearance. The adopted method pools gradient orientation
    information locally in the same way as the SIFT [Lowe 2004]
    feature. The image window is divided into small spatial regions,
    called "cells". For each cell we accumulate a local 1-D histogram
    of gradient or edge orientations over all the pixels in the
    cell. This combined cell-level 1-D histogram forms the basic
    "orientation histogram" representation. Each orientation histogram
    divides the gradient angle range into a fixed number of
    predetermined bins. The gradient magnitudes of the pixels in the
    cell are used to vote into the orientation histogram.
    """

    s_row, s_col = image.shape[:2]
    c_row, c_col = pixels_per_cell
    b_row, b_col = cells_per_block

    n_cells_row = int(s_row // c_row)  # number of cells along row-axis
    n_cells_col = int(s_col // c_col)  # number of cells along col-axis

    # compute orientations integral images
    orientation_histogram = _hog_histograms(
        g_row,
        g_col,
        c_row,
        c_col,
        orientations
    )

    # now compute the histogram for each cell
    hog_image = None

    if visualize:
        radius = min(c_row, c_col) // 2 - 1
        orientations_arr = np.arange(orientations)
        # set dr_arr, dc_arr to correspond to midpoints of orientation bins
        orientation_bin_midpoints = (
            np.pi * (orientations_arr + .5) / orientations)
        dr_arr = radius * np.sin(orientation_bin_midpoints)
        dc_arr = radius * np.cos(orientation_bin_midpoints)
        hog_image = np.zeros((s_row, s_col), dtype=float)
        for r in range(n_cells_row):
            for c in range(n_cells_col):
                for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
                    centre = tuple([r * c_row + c_row // 2,
                                    c * c_col + c_col // 2])
                    rr, cc = utils.line(int(centre[0] - dc),
                                       int(centre[1] + dr),
                                       int(centre[0] + dc),
                                       int(centre[1] - dr))
                    hog_image[rr, cc] += orientation_histogram[r, c, o]

    """
    The fourth stage computes normalization, which takes local groups of
    cells and contrast normalizes their overall responses before passing
    to next stage. Normalization introduces better invariance to illumination,
    shadowing, and edge contrast. It is performed by accumulating a measure
    of local histogram "energy" over local groups of cells that we call
    "blocks". The result is used to normalize each cell in the block.
    Typically each individual cell is shared between several blocks, but
    its normalizations are block dependent and thus different. The cell
    thus appears several times in the final output vector with different
    normalizations. This may seem redundant but it improves the performance.
    We refer to the normalized block descriptors as Histogram of Oriented
    Gradient (HOG) descriptors.
    """

    n_blocks_row = (n_cells_row - b_row) + 1
    n_blocks_col = (n_cells_col - b_col) + 1
    normalized_blocks = np.zeros((n_blocks_row, n_blocks_col,
                                  b_row, b_col, orientations))

    for r in range(n_blocks_row):
        for c in range(n_blocks_col):
            block = orientation_histogram[r:r + b_row, c:c + b_col, :]
            normalized_blocks[r, c, :] = \
                _hog_normalize_block(block, method=block_norm)

    """
    The final step collects the HOG descriptors from all blocks of a dense
    overlapping grid of blocks covering the detection window into a combined
    feature vector for use in the window classifier.
    """

    if feature_vector:
        normalized_blocks = normalized_blocks.ravel()

    if visualize:
        return normalized_blocks, hog_image
    else:
        return normalized_blocks
