import numpy
from scipy.signal import convolve2d as conv2d


def anomaly_filter(img, source, interval=3):
    L = img[:, :, 0]
    pixels = []
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            if source[i][j] == 1:
                pixels.append(L[i][j])
    avg = numpy.mean(pixels)
    dev = numpy.std(pixels)
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            if abs(L[i][j] - avg) > interval * dev:
                source[i][j] = 0
    return source


def get_patch(size, coord, patch_size):
    rad = (patch_size - 1) / 2
    rows = int(min(coord[0] + rad, size[0] - 1) - max(coord[0] - rad, 0) + 1)
    cols = int(min(coord[1] + rad, size[1] - 1) - max(coord[1] - rad, 0) + 1)
    patch = numpy.zeros((rows, cols, 2))
    for i in range(rows):
        for j in range(cols):
            patch[i][j] = [i + max(coord[0] - rad, 0), j + max(coord[1] - rad, 0)]
    return patch, (rows, cols)


def best_match(img, source, patch, threshold):
    search_x = img.shape[0] - patch.shape[0]
    search_y = img.shape[1] - patch.shape[1]
    min_dist = numpy.inf
    best_loc = None
    # total_num = below_thres = 0
    for i in range(search_x):
        for j in range(search_y):
            if not source[i: i + patch.shape[0], j: j + patch.shape[1]].all():
                continue
            dist = 0
            for m in range(patch.shape[0]):
                for n in range(patch.shape[1]):
                    patch_x, patch_y = patch[m][n].astype(int)
                    if source[patch_x][patch_y] == 0:
                        continue
                    dist += (float(img[i + m][j + n][0]) - float(img[patch_x][patch_y][0])) ** 2
                    dist += (float(img[i + m][j + n][1]) - float(img[patch_x][patch_y][1])) ** 2
                    dist += (float(img[i + m][j + n][2]) - float(img[patch_x][patch_y][2])) ** 2
            # if dist < threshold:
            #     below_thres += 1
            if dist < min_dist:
                min_dist = dist
                best_loc = (i, j)
            if min_dist <= threshold:
                # print(min_dist)
                return best_loc
            # total_num += 1
    # print(f'{below_thres} below threshold with minimum {min_dist} out of {total_num} patches')
    return best_loc


def criminisi(img_masked, patch_size, alpha, beta):
    img_shape = img_masked.shape
    conv_kernel = numpy.array([[1, 1, 1],
                               [1, -8, 1],
                               [1, 1, 1]])

    fill_region = numpy.zeros(img_shape[0: 2])
    fill_region_size = 0
    src_region = numpy.zeros(img_shape[0: 2])
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if numpy.array_equal(img_masked[i][j], [0, 0, 0]):
                fill_region[i][j] = 1
                fill_region_size += 1
            else:
                src_region[i][j] = 1

    # src_region = anomaly_filter(img_masked, src_region, interval=alpha)

    ix = numpy.zeros(img_shape)
    iy = numpy.zeros(img_shape)
    ix[:, :, 0], iy[:, :, 0] = numpy.gradient(img_masked[:, :, 0])
    ix[:, :, 1], iy[:, :, 1] = numpy.gradient(img_masked[:, :, 1])
    ix[:, :, 2], iy[:, :, 2] = numpy.gradient(img_masked[:, :, 2])
    ix = numpy.sum(ix, axis=2) / (3 * 255)
    iy = numpy.sum(iy, axis=2) / (3 * 255)
    Ix = -iy
    Iy = ix

    filled_patch_num = filled_pixel_num = 0
    while fill_region.any():
        # print(f'Filling patch {filled_patch_num + 1}:')

        # print('Finding pixels to be filled...')
        dR = numpy.where(conv2d(fill_region, conv_kernel, mode='same') > 0)
        dR_list = list(zip(dR[0], dR[1]))
        Nx, Ny = numpy.gradient(src_region)
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                if tuple((i, j)) not in dR_list:
                    Nx[i][j] = Ny[i][j] = 0
        
        # print('Calculating priorities...')
        priorities = {}
        for loc in dR_list:
            pat_coords, _ = get_patch(img_shape[0: 2], loc, patch_size)
            conf = 0
            for i in range(pat_coords.shape[0]):
                for j in range(pat_coords.shape[1]):
                    coord_x, coord_y = pat_coords[i][j].astype(int)
                    if fill_region[coord_x][coord_y] == 0:
                        conf += 1
            conf = conf / (pat_coords.shape[0] * pat_coords.shape[1])
            data = abs(Ix[loc[0]][loc[1]] * Nx[loc[0]][loc[1]] + Iy[loc[0]][loc[1]] * Ny[loc[0]][loc[1]]) / 255
            priorities[loc] = conf * data

        fill_pixel = max(priorities, key=priorities.get)
        fill_patch, _ = get_patch(img_shape[0: 2], fill_pixel, patch_size)

        # print('Finding best source patch...')
        rep_loc = best_match(img_masked, src_region, fill_patch, threshold=beta)
        for i in range(fill_patch.shape[0]):
            for j in range(fill_patch.shape[1]):
                pixel_x, pixel_y = fill_patch[i][j].astype(int)
                if fill_region[pixel_x][pixel_y] == 1:
                    rep_x, rep_y = rep_loc
                    img_masked[pixel_x][pixel_y] = img_masked[rep_x + i][rep_y + j]
                    fill_region[pixel_x][pixel_y] = 0
                    src_region[pixel_x][pixel_y] = 1
                    Ix[pixel_x][pixel_y] = Ix[rep_x + i][rep_y + j]
                    Iy[pixel_x][pixel_y] = Iy[rep_x + i][rep_y + j]
                    filled_pixel_num += 1
        filled_patch_num += 1
        if filled_patch_num < 6 or filled_patch_num % 10 == 0:
            print(f'Filled {filled_patch_num} patches, {filled_pixel_num} of {fill_region_size} pixels.')

    return img_masked
