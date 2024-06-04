import numpy as np


def get_if_header_already_post_processed(header, postprocess):
    comments = list()
    for comment in header["COMMENT"]:
        comments.append(comment.strip("   "))
    comments = set(comments)
    return postprocess in comments


def get_scale(hwcs):
    cd11 = hwcs['CD1_1']
    cd12 = hwcs['CD1_2']
    cd21 = hwcs['CD2_1']
    cd22 = hwcs['CD2_2']
    return np.sqrt(cd11 ** 2 + cd12 ** 2) * 3600


def get_ccw(hwcs):
    cd11 = hwcs['CD1_1']
    cd12 = hwcs['CD1_2']
    cd21 = hwcs['CD2_1']
    cd22 = hwcs['CD2_2']
    det = cd11 * cd22 - cd12 * cd21
    if det >= 0:
        parity = 1.
    else:
        parity = -1.
    T = parity * cd11 + cd22
    A = parity * cd21 - cd12
    return -np.degrees(np.arctan2(A, T))
