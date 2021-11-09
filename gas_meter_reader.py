#!/usr/bin/env python3
"""Read gas meter using machine vision"""
import sys
import glob
import os
import os.path
import math
import numpy as np
import cv2

DIALS = [
    #offset, clockwise, factor
    [0, False, 1],
    [0, True, 1],
    [0, False, 1],
    [0, True, 1],
    [0, False, 0],
    [0, False, 0],
    [0, False, 0],
    [0, False, 0],
    [0, False, 0],
    [0, False, 0],
    [0, False, 0],
    [0, False, 0],
    [0, False, 0],
    [0, False, 0],
    [0, False, 0]
]

def clear_debug():
    """Clear debug directory"""
    filelist = glob.glob("output/*.jpg")
    for file in filelist:
        os.remove(file)

def write_debug(img, name, sample):
    """Write image to debug directory"""
    cv2.imwrite(f"output/{sample}-{name}.jpg", img)

def find_least_covered_angle(edges, idx, sample):
    """Find angle with the fewest pixels covered from center"""
    height, width = edges.shape[:2]

    center = [width / 2, height / 2]
    radius = int(width / 2)

    least_count = height * width
    least_angle = None
    streak = 0
    step = 1
    parts = 4.0
    deb = -1

    # Remove the border, often noisy
    #mask = np.zeros((height, width, 1), np.uint8)
    #cv2.circle(mask, (int(center[0]), int(center[1])), radius,
    #           (255, 255, 255), 20)
    #mask = cv2.bitwise_not(mask)
    #trimmed = cv2.bitwise_and(edges, edges, mask=mask)
    trimmed = edges.copy()
    #write_debug(trimmed, f"trimmed-{idx}", sample)
    for partangle in range(0, 1440, step):
        angle = partangle / parts
        angle_r = angle * (np.pi / 180)

        origin_point = [center[0], 0]
        angle_point = [
            math.cos(angle_r) * (origin_point[0] - center[0]) - \
            math.sin(angle_r) * (origin_point[1] - center[1]) + center[0],
            math.sin(angle_r) * (origin_point[0] - center[0]) + \
            math.cos(angle_r) * (origin_point[1] - center[1]) + center[1]
        ]

        mask = np.zeros((height, width, 1), np.uint8)
        cv2.line(mask, (int(center[0]), int(center[1])),
                 (int(angle_point[0]), int(angle_point[1])),
                 (255, 255, 255), 2)
        masked = cv2.bitwise_and(trimmed, trimmed, mask=mask)
        count = cv2.countNonZero(masked)
        if deb == idx:
            write_debug(masked, f"masked-{idx}-{angle}-{count}", sample)
        if count < least_count:
            least_count = count
            least_angle = angle
            streak = 1
            if deb == idx:
                print("first least_count %d, least_angle %s" %
                      (least_count, str(least_angle)))
        elif count == least_count:
            streak = streak + 1
            least_angle = (2 * angle - ((streak-1) * step)/parts) / 2
            if deb == idx:
                print("streak=%s least_count %d, angle=%s, least_angle %s" %
                      (streak, least_count, str(angle), str(least_angle)))
        else:
            streak = 1

    return least_angle

def black_white_points(in_img):
    """Normalize & crush image to darkest/lightest pixels"""
    crush = 15
    blackest, whitest, min_loc, max_loc = cv2.minMaxLoc(in_img)
    _ = min_loc
    _ = max_loc
    out_img = in_img.copy()
    blackest += crush
    whitest -= crush
    offset = blackest
    scale = 255.0 / (whitest-blackest)
    #print("Blackest: %d, whitest: %d, scale %f" % (blackest, whitest, scale))

    # dst = src1*alpha + src2*beta + gamma;
    # dst = cv.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])
    out_img = cv2.addWeighted(in_img, scale, in_img, 0, -offset)

    return out_img

def read_dial(config, idx, img, sample, prev_value):
    """Read one dial"""
    offset, clockwise, factor = config
    if prev_value:
        # Bias dial from next less-significant digit
        prev_factor = 3.6 * (prev_value - 5)
        if clockwise:
            prev_factor = 0 - prev_factor
        offset += prev_factor
    offset_r = offset * (np.pi / 180)

    height, width = img.shape[:2]
    center = [width / 2, height / 2]

    offset_img = img.copy()
    origin_point = [center[0], 0]
    offset_point = [
        math.cos(offset_r) * (origin_point[0] - center[0]) -
        math.sin(offset_r) * (origin_point[1] - center[1]) + center[0],
        math.sin(offset_r) * (origin_point[0] - center[0]) +
        math.cos(offset_r) * (origin_point[1] - center[1]) + center[1]
    ]
    cv2.line(offset_img, (int(center[0]), int(center[1])),
             (int(offset_point[0]), int(offset_point[1])), (0, 255, 0), 2)
    write_debug(offset_img, f"dial-{idx}", sample)

    if len(img.shape) == 3: # color
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #write_debug(blurred, f"blurred-{idx}", sample)

    #edges = cv2.Canny(blurred, 50, 200)
    ret, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    write_debug(thresh, f"thresh-{idx}", sample)

    angle = find_least_covered_angle(thresh, idx, sample) + offset
    angle_r = angle * (np.pi / 180)
    angle_point = [
        math.cos(angle_r) * (origin_point[0] - center[0]) - \
        math.sin(angle_r) * (origin_point[1] - center[1]) + center[0],
        math.sin(angle_r) * (origin_point[0] - center[0]) + \
        math.cos(angle_r) * (origin_point[1] - center[1]) + center[1]]
    if len(img.shape) == 2: # greyscale
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        color_img = img.copy()
    cv2.line(color_img, (int(center[0]), int(center[1])),
             (int(angle_point[0]), int(angle_point[1])), (0, 0, 255), 2)
    write_debug(color_img, f"angle-{idx}", sample)

    if angle < 0:
        angle = 360 + angle
    angle_p = angle/360
    if not clockwise:
        angle_p = 1 - angle_p

    return (angle, int(10*angle_p*factor))

def get_circles(original, sample):
    """Find circles in captured image"""
    write_debug(original, "frame", sample)

    #area = [400, 700,350, 96+1020+200]
    area = [400, 400, 1300, 710] # x1, y1, x2, y2
    #area = [418, 20, 418+1018, 20+391] # x1, y1, x2, y2

    crop = original[area[1]:area[3], area[0]:area[2]].copy()

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    norm = black_white_points(gray)
    write_debug(norm, "norm", sample)

    blurred = cv2.GaussianBlur(norm, (5, 5), 0)
    write_debug(blurred, "blurred", sample)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 40,
                               np.array([]), 100, 100, 20, 300)
    if circles is not None:
        circles = circles.tolist()[0]
    return norm, circles

def process(crop, circles, sample):
    """Process a captured image"""
    prev_value = None
    dials = np.uint16(np.around(circles))

    sorted_dials = sorted(dials, key=lambda dial: dial[0])
    result = []
    firstangle = None

    if len(crop.shape) == 2: # greyscale
        color_crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    else:
        color_crop = crop.copy()
    for idx, dial in reversed(list(enumerate(sorted_dials))):
        x_pos, y_pos, radius = dial
        radius = radius + 5
        dial_img = crop[y_pos-radius:y_pos+radius,
                        x_pos-radius:x_pos+radius].copy()
        angle, value = read_dial(DIALS[idx], idx, dial_img, sample, prev_value)
        if DIALS[idx][2]:
            result.append(value)
            if not firstangle:
                firstangle = angle
                firstvalue = value
            prev_value = value
        # draw the outer circle
        cv2.circle(color_crop, (x_pos, y_pos), radius, (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(color_crop, (x_pos, y_pos), 2, (0, 0, 255), 3)

    write_debug(color_crop, "circles", sample)

    remainder = (firstangle/360.0)*10 - firstvalue
    result.reverse()
    result.append(remainder)
    return result

def assemble_reading(result):
    """Assemble dial readings into a numeric result"""
    output = 0.0
    power = len(result) - 2
    for res in result:
        output += res * 10**power
        power = max(0, power-1)
    return output

def main(argv):
    """Command line entry point"""
    clear_debug()

    filename = argv[0] if len(sys.argv) > 0 else ""

    if not os.path.exists(filename):
        print("Usage: python3 power-meter-reader.py <image>")
        sys.exit(1)

    original = cv2.imread(filename)

    crop, circles = get_circles(original, 0)
    result = process(crop, circles, 0)
    output = assemble_reading(result)
    print ("HERE")
    print("%f" % output)

if __name__ == '__main__':
    main(sys.argv[1:])