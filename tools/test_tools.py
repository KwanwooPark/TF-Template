import numpy as np
from config_kp import cfg
import glob
import xml
import cv2
from xml.etree.ElementTree import Element, SubElement, ElementTree
from xml.dom import minidom
import struct
from scipy.optimize import linear_sum_assignment

color_list = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (200, 96, 0), (0, 200, 96), (96, 0, 200),
                                                     (200, 0, 96), (96, 200, 0), (0, 96, 200)]


class LANE():
    def __init__(self):
        self.Th_int = 0
        self.Tw_int = 0
        self.socre = 0

        self.Th = 0
        self.Tw = 0
        self.h = np.zeros((cfg.point_num,), np.float)
        self.w = np.zeros((cfg.point_num,), np.float)
        self.p = np.zeros((4,), np.float)

        self.h_min = 0
        self.h_max = 0

        self.miss = 0
        self.catch = 1
        self.appear = False

def make_point_image_for_prediction(img, hm, kp, threshold=0.3): # w * h * 8 * 2
    color_list = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 128, 0), (0, 255, 128), (128, 0, 255)]

    inds = np.where(hm >= threshold)
    Hs = inds[0]
    Ws = inds[1]

    count = 0
    for ind in range(Hs.size):
        count += 1
        TH_int = Hs[ind]
        TW_int = Ws[ind]

        hs = kp[TH_int, TW_int, 0:cfg.point_num] + TH_int + 0.5
        ws = kp[TH_int, TW_int, cfg.point_num:cfg.point_num * 2] + TW_int + 0.5
        for p in range(6):
            h = int(hs[p] * cfg.rate + 0.5)
            w = int(ws[p] * cfg.rate + 0.5)
            cv2.circle(img, (w, h), 3, color=color_list[count % 6], thickness=-1)

    return img

def extract_list(hm, kp, threshold):
    inds = np.where(hm > threshold)
    x = inds[0]
    y = inds[1]
    lane_list = []
    for ind in range(x.size):
        now_lane = LANE()
        now_lane.Th_int = x[ind]
        now_lane.Tw_int = y[ind]
        now_lane.score = hm[now_lane.Th_int, now_lane.Tw_int]
        now_lane.h = (kp[now_lane.Th_int, now_lane.Tw_int, 0:cfg.point_num] + now_lane.Th_int + 0.5) * cfg.rate
        now_lane.w = (kp[now_lane.Th_int, now_lane.Tw_int, cfg.point_num:cfg.point_num * 2] + now_lane.Tw_int + 0.5) * cfg.rate
        now_lane.h_min = now_lane.h[0]
        now_lane.h_max = now_lane.h[-1]
        now_lane.Th = now_lane.Th_int * cfg.rate
        now_lane.Tw = now_lane.Tw_int * cfg.rate
        now_lane.p = np.polyfit(now_lane.h, now_lane.w, 3, rcond=0)
        lane_list.append(now_lane)
    lane_list = sorted(lane_list, key=lambda pair: pair.score, reverse=True)
    return lane_list


def lane_NMS(lane_list, threshold):
    out_lane = []
    while lane_list:
        now_lane = lane_list[0]
        out_lane.append(now_lane)

        new_list = []
        for idx in range(1, len(lane_list)):
            target_lane = lane_list[idx]
            diff = diff_nurb(now_lane, target_lane, threshold)
            if diff > threshold:
                new_list.append(lane_list[idx])
                continue
        lane_list = new_list
    return out_lane


def diff_nurb(lane1, lane2, threshold):
    h_max1 = lane1.h_max
    h_max2 = lane2.h_max
    h_min1 = lane1.h_min
    h_min2 = lane2.h_min

    min_length = min(h_max1 - h_min1, h_max2 - h_min2)

    h_max1, h_max2 = sorted([h_max1, h_max2])
    h_min1, h_min2 = sorted([h_min1, h_min2])
    if min_length * 0.5 > (h_max1 - h_min2):
        return threshold * 2

    h1, h2 = h_max1, h_min2
    poly1, poly2 = lane1.p.copy(), lane2.p.copy()

    poly1[0] /= 4
    poly1[1] /= 3
    poly1[2] /= 2
    poly2[0] /= 4
    poly2[1] /= 3
    poly2[2] /= 2
    poly = poly1 - poly2
    w2 = poly[0] * h2 * h2 * h2 * h2 + poly[1] * h2 * h2 * h2 + poly[2] * h2 * h2 + poly[3] * h2
    w1 = poly[0] * h1 * h1 * h1 * h1 + poly[1] * h1 * h1 * h1 + poly[2] * h1 * h1 + poly[3] * h1
    diff = abs(w1 - w2) / (h1 - h2 + 1e-6)
    return min(diff, threshold * 2)

def Draw_lane(img, lane_list):
    for idx, lane in enumerate(lane_list):
        color1 = (255, 0, 0)
        color2 = (255, 0, 255)
        for p in range(cfg.point_num):
                h = int(lane.h[p] + 0.5)
                w = int(lane.w[p] + 0.5)
                cv2.circle(img, (w, h), 3, color=color1, thickness=-1)

        h_ = np.arange(lane.h_min, lane.h_max)
        w_ = np.polyval(lane.p, h_)
        for i in range(h_.shape[0]):
            cv2.circle(img, (int(w_[i] + 0.5), int(h_[i] + 0.5)), 1, color=color1, thickness=-1)

        text = '%d, %0.2f, %d, %d' % (idx+1, lane.score, lane.catch, lane.miss)
        cv2.putText(img, text, (int(lane.Tw + 0.5), int(lane.Th - 10.5)), cv2.FONT_HERSHEY_DUPLEX, 0.5, color2)

    return img


def Draw_lane_to_origin(img, lane_list, crop_h, ratio):
    draw_lane = []
    for idx, lane in enumerate(lane_list):
        if lane.score > 0.5:
            draw_lane.append(lane)

    draw_lane = sorted(draw_lane, key=lambda pair: pair.Tw, reverse=True)
    for idx, lane in enumerate(draw_lane):
        if lane.score < 0.5:
            continue

        color1 = color_list[idx % 9]

        for p in range(cfg.point_num):
                h = int(lane.h[p] * ratio + crop_h + 0.5)
                w = int(lane.w[p] * ratio + 0.5)
                cv2.circle(img, (w, h), int(3 * ratio + 0.5), color=color1, thickness=-1)

        h_ = np.arange(lane.h_min, lane.h_max)
        w_ = np.polyval(lane.p, h_)
        for i in range(h_.shape[0]):
            h = int(h_[i] * ratio + crop_h + 0.5)
            w = int(w_[i] * ratio + 0.5)
            cv2.circle(img, (w, h), int(1 * ratio + 0.5), color=color1, thickness=-1)
    return img


def Draw_lane_to_origin_only_points(img, lane_list, crop_h, ratio):
    draw_lane = []
    for idx, lane in enumerate(lane_list):
        if lane.score > 0.5:
            draw_lane.append(lane)

    draw_lane = sorted(draw_lane, key=lambda pair: pair.Tw, reverse=True)
    for idx, lane in enumerate(draw_lane):
        if lane.score < 0.5:
            continue

        color1 = color_list[idx % 9]

        for p in range(cfg.point_num):
                h = int(lane.h[p] * ratio + crop_h + 0.5)
                w = int(lane.w[p] * ratio + 0.5)
                cv2.circle(img, (w, h), int(3 * ratio + 0.5), color=color1, thickness=-1)
    return img



def Draw_lane_to_origin_byscore(img, lane_list, crop_h, ratio):
    for idx, lane in enumerate(lane_list):
        xx = (lane.score - 0.3) / 0.7
        if xx < 0:
            xx = 0
        elif xx > 1:
            xx = 1

        color_r = 255 * (1 - xx)
        color_g = 0
        color_b = 255 * xx
        color1 = (color_b, color_g, color_r)

        for p in range(cfg.point_num):
                h = int(lane.h[p] * ratio + crop_h + 0.5)
                w = int(lane.w[p] * ratio + 0.5)
                cv2.circle(img, (w, h), int(3 * ratio + 0.5), color=color1, thickness=-1)

        h_ = np.arange(lane.h_min, lane.h_max)
        w_ = np.polyval(lane.p, h_)
        for i in range(h_.shape[0]):
            h = int(h_[i] * ratio + crop_h + 0.5)
            w = int(w_[i] * ratio + 0.5)
            cv2.circle(img, (w, h), int(1 * ratio + 0.5), color=color1, thickness=-1)
    return img


def save_lane(folder_name, img_name, full_path, save_folder, lane_list):
    root = Element('annotation')
    SubElement(root, 'folder').text = folder_name
    SubElement(root, 'filename').text = img_name
    SubElement(root, 'path').text = full_path

    size = SubElement(root, 'size')
    SubElement(size, 'width').text = '1920'
    SubElement(size, 'height').text = '1080'
    SubElement(size, 'depth').text = '3'
    for lane in lane_list:
        obj = SubElement(root, 'lane')
        points = SubElement(obj, 'points')
        for i in range(cfg.point_num):
            SubElement(points, 'h%d' % i).text = str(lane.h[i])
            SubElement(points, 'w%d' % i).text = str(lane.w[i])

        center = SubElement(obj, 'center')
        SubElement(center, 'h').text = str(lane.ch)
        SubElement(center, 'w').text = str(lane.cw)

        poly = SubElement(obj, 'poly')
        SubElement(poly, 'p3th').text = str(lane.p[0])
        SubElement(poly, 'p2th').text = str(lane.p[1])
        SubElement(poly, 'p1th').text = str(lane.p[2])
        SubElement(poly, 'p0th').text = str(lane.p[3])

        SubElement(obj, 'score').text = str(lane.score)
        SubElement(obj, 'appear').text = str(lane.appear)

    tree = ElementTree(root)
    tree.write(save_folder + img_name.replace('.png', '.xml').replace('.jpg', '.xml'))


def lane_revision(lane_list, cut, rate):
    out_list = []
    for lane in lane_list:
        New_lane = LANE()

        New_lane.h = lane.h * rate + cut[0]
        New_lane.w = lane.w * rate + cut[1]
        New_lane.ch = lane.ch * rate + cut[0]
        New_lane.cw = lane.cw * rate + cut[1]
        New_lane.p = np.polyfit(lane.h, lane.w, 3, rcond=0)
        New_lane.score = lane.score
        New_lane.appear = lane.appear
        out_list.append(New_lane)

    return out_list
