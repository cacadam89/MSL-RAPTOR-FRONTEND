import math
import numpy as np
import numpy.linalg as la

def bb_corners_to_angled_bb(points, output_coord_type='xy'):
    """
    BB_CORNERS_TO_ANGLE Function that takes takes coordinates of a bounding
    box corners and returns it as center, size and angle.
    points = [x1,y1;x2,y2;x3,y3;x4,y4] (N x 2 matrix)
    """
    sortind_x = np.argsort(points[:, 0], axis=0)  # sort points by x coordinate

    # of the points furthest to the left, which is lower and which is higher? (bottom left / top left)
    if points[sortind_x[0], 1] > points[sortind_x[1], 1]:
        bl_x = points[sortind_x[0], 0]
        bl_y = points[sortind_x[0], 1]
        tl_x = points[sortind_x[1], 0]
        tl_y = points[sortind_x[1], 1]
    else:
        bl_x = points[sortind_x[1], 0]
        bl_y = points[sortind_x[1], 1]
        tl_x = points[sortind_x[0], 0]
        tl_y = points[sortind_x[0], 1]

    # of the points furthest to the right, which is lower and which is higher? (bottom right / top right)
    if points[sortind_x[2], 1] > points[sortind_x[3], 1]:
        br_x = points[sortind_x[2], 0]
        br_y = points[sortind_x[2], 1]
        tr_x = points[sortind_x[3], 0]
        tr_y = points[sortind_x[3], 1]
    else:
        br_x = points[sortind_x[3], 0]
        br_y = points[sortind_x[3], 1]
        tr_x = points[sortind_x[2], 0]
        tr_y = points[sortind_x[2], 1]

    # print(bl_x)
    # print(bl_y)
    # print(br_x)
    # print(br_y)
    if np.abs(br_x - bl_x) < 0.00001:
        print("error")
    angle = -np.arctan((bl_y - br_y) / (bl_x - br_x))
    x_center = np.mean([br_x, bl_x, tl_x, tr_x])
    y_center = np.mean([br_y, bl_y, tl_y, tr_y])
    width = la.norm([br_x - bl_x, br_y - bl_y])
    height = la.norm([br_x - tr_x, br_y - tr_y])
    output = np.array([x_center, y_center, width, height, angle])
    if output_coord_type.lower() == 'rc':
        # r is y, col is x
        output = np.array([y_center, x_center, width, height, angle])

    return output


def calc_row_idx(k, n):
    return int(math.ceil((1/2.) * (- (-8*k + 4 *n**2 -4*n - 7)**0.5 + 2*n -1) - 1))

def elem_in_i_rows(i, n):
    return i * (n - 1 - i) + (i*(i + 1))/2

def calc_col_idx(k, i, n):
    return int(n - elem_in_i_rows(i + 1, n) + k)

def condensed_to_square(k, n):
    i = calc_row_idx(k, n)
    j = calc_col_idx(k, i, n)
    return i, j