import numpy as np
from config import cfg
import glob
from tools.tools import *
import xml
from xml.etree.ElementTree import Element, SubElement, ElementTree
from xml.dom import minidom
import struct
from scipy.optimize import linear_sum_assignment

color_list = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 128, 0), (0, 255, 128), (128, 0, 255)]


class LANE():
    def __init__(self):
        self.ch = 0
        self.cw = 0
        self.ch_int = 0
        self.cw_int = 0

        self.h = np.zeros((16,), np.float)
        self.w = np.zeros((16,), np.float)
        self.p = np.zeros((4,), np.float)
        self.socre = 0
        self.miss = 0
        self.catch = 1
        self.appear = False


def extract_list(hm, os, kp, threshold=0.5):
    hm_max = pool2d(hm, kernel_size=3, stride=1, padding=1, pool_mode='max')
    keep = (hm_max == hm).astype(np.float)
    hm = hm * keep

    inds = np.where(hm > threshold)
    x = inds[0]
    y = inds[1]
    lane_list = []
    for ind in range(x.size):
        now_lane = LANE()

        cx_int = x[ind]
        cy_int = y[ind]

        now_lane.ch_int = cx_int
        now_lane.cw_int = cy_int

        now_lane.ch = cx_int * cfg.rate + os[cx_int, cy_int, 0]
        now_lane.cw = cy_int * cfg.rate + os[cx_int, cy_int, 1]

        point = kp[cx_int, cy_int, :, :].copy()  # 8 * 2
        now_lane.h = point[:, 0] + now_lane.ch
        now_lane.w = point[:, 1] + now_lane.cw

        now_lane.p = np.polyfit(now_lane.h, now_lane.w, 3, rcond=0)
        now_lane.score = hm[cx_int, cy_int]

        lane_list.append(now_lane)

    lane_list = sorted(lane_list, key=lambda pair: pair.score, reverse=True)

    return lane_list


def lane_NMS(lane_list, pred_hm, threshold):
    out_lane = []
    while lane_list:
        now_lane = lane_list[0]
        out_lane.append(now_lane)
        pred_hm[now_lane.ch_int, now_lane.cw_int] = 0

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
    h_max1 = lane1.h[15].copy()
    h_max2 = lane2.h[15].copy()
    h_min1 = lane1.h[0].copy()
    h_min2 = lane2.h[0].copy()

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


def save_to_bin(mydata, file_name):
    mydata = mydata.astype(np.float32)
    mydata = mydata.reshape((-1,))
    f = open(file_name, "wb")
    myfmt = 'f' * len(mydata)
    bin = struct.pack(myfmt, *mydata)
    f.write(bin)
    f.close()


def tracking(prev_list, cur_list, pred_hm, pred_os, pred_kp, threshold=50, threshold_hm=0.2, num_appear=4, num_remove=4):
    prev_num = len(prev_list)
    cur_num = len(cur_list)
    max_num = max(prev_num, cur_num)

    cost = np.zeros((max_num, max_num), np.float32)
    for last_idx, last_lane in enumerate(prev_list):
        for now_idx, now_lane in enumerate(cur_list):
            cost[last_idx, now_idx] = diff_nurb(last_lane, now_lane, threshold)

    _inds, prev2cur = linear_sum_assignment(cost)
    cur2prev = np.zeros_like(prev2cur)
    cur2prev[prev2cur[_inds]] = _inds


    no_detected_list = []
    detected_list = []

    for cur_ind, cur_lane in enumerate(cur_list):
        prev_ind = cur2prev[cur_ind]
        if (prev_ind >= prev_num) or (cost[prev_ind, cur_ind] > threshold):
            detected_list.append(cur_lane)
        else:
            cur_lane.catch = prev_list[prev_ind].catch + 1
            if cur_lane.catch >= num_appear:
                cur_lane.appear = True
            detected_list.append(cur_lane)

    for prev_ind, prev_lane in enumerate(prev_list):
        cur_ind = prev2cur[prev_ind]
        if (cur_ind >= cur_num) or (cost[prev_ind, cur_ind] > threshold):
            if prev_lane.catch >= num_appear:
                no_detected_list.append(prev_lane)

    for lane in no_detected_list:
        ch, cw = lane.ch_int, lane.cw_int

        h_start, h_end = max(ch - 1, 0), min(ch + 2, int(cfg.height / cfg.rate))
        w_start, w_end = max(cw - 1, 0), min(cw + 2, int(cfg.width / cfg.rate))

        max_diff = threshold
        add_lane = LANE()

        for ch in range(h_start, h_end):
            for cw in range(w_start, w_end):
                if pred_hm[ch, cw] > threshold_hm:
                    new_lane = LANE()
                    new_lane.ch_int = ch
                    new_lane.cw_int = cw

                    new_lane.ch = ch * cfg.rate + pred_os[ch, cw, 0]
                    new_lane.cw = cw * cfg.rate + pred_os[ch, cw, 1]

                    point = pred_kp[ch, cw, :, :].copy()  # 8 * 2
                    new_lane.h = point[:, 0] + lane.ch
                    new_lane.w = point[:, 1] + lane.cw

                    new_lane.p = np.polyfit(new_lane.h, new_lane.w, 3, rcond=0)
                    new_lane.score = pred_hm[ch, cw]
                    diff_ = diff_nurb(new_lane, lane, threshold)
                    if diff_ < max_diff:
                        max_diff = diff_
                        add_lane = new_lane

        if max_diff >= threshold:
            lane.appear = False
            lane.miss += 1
            if lane.miss < num_remove:
                detected_list.append(lane)
            continue

        pred_hm[add_lane.ch_int, add_lane.cw_int] = 0

        add_lane.catch = lane.catch + 1
        if add_lane.catch >= num_appear:
            add_lane.appear = True
        detected_list.append(add_lane)

    return detected_list


def tracking_v2(prev_list, cur_list, threshold, threshold_hm, num_appear=4, num_remove=3):
    prev_num = len(prev_list)
    cur_num = len(cur_list)
    max_num = max(prev_num, cur_num)

    cost = np.zeros((max_num, max_num), np.float32)
    for last_idx, last_lane in enumerate(prev_list):
        for now_idx, now_lane in enumerate(cur_list):
            cost[last_idx, now_idx] = diff_nurb(last_lane, now_lane, threshold)

    _inds, prev2cur = linear_sum_assignment(cost)
    cur2prev = np.zeros_like(prev2cur)
    cur2prev[prev2cur[_inds]] = _inds

    tracker_list = []
    cur_matched = [False] * cur_num
    for prev_ind, prev_lane in enumerate(prev_list):
        cur_ind = prev2cur[prev_ind]
        if (cur_ind >= cur_num) or (cost[prev_ind, cur_ind] > threshold):
            if prev_lane.catch < num_appear:
                continue
            prev_lane.miss += 1
            if prev_lane.miss >= num_remove:
                continue
            prev_lane.appear = False
            tracker_list.append(prev_lane)
        else:
            cur_matched[cur_ind] = True
            new_tracker = cur_list[cur_ind]
            new_tracker.catch = prev_lane.catch + 1
            if new_tracker.catch >= num_appear:
                new_tracker.appear = True
            tracker_list.append(new_tracker)

    for cur_ind in range(cur_num):
        if cur_matched[cur_ind]:
            continue
        if cur_list[cur_ind].score < threshold_hm:
            continue
        tracker_list.append(cur_list[cur_ind])

    return tracker_list


def Draw_lane(img, lane_list):
    count = 0
    for idx, lane in enumerate(lane_list):
        if lane.appear:
            color1 = (0, 255, 0)
            color2 = (0, 255, 255)
        else:
            color1 = (255, 0, 0)
            color2 = (255, 0, 255)

        # cv2.circle(img, (int(lane.cw + 0.5), int(lane.ch + 0.5)), 6, color=color, thickness=2)

        for p in range(16):
                h = int(lane.h[p] + 0.5)
                w = int(lane.w[p] + 0.5)
                cv2.circle(img, (w, h), 3, color=color1, thickness=-1)

        h_ = np.arange(lane.h[0], lane.h[15])
        w_ = np.polyval(lane.p, h_)
        for i in range(h_.shape[0]):
            cv2.circle(img, (int(w_[i] + 0.5), int(h_[i] + 0.5)), 1, color=color1, thickness=-1)

        text = '%d, %0.2f, %d, %d' % (idx+1, lane.score, lane.catch, lane.miss)
        cv2.putText(img, text, (int(lane.cw + 0.5), int(lane.ch - 10.5)), cv2.FONT_HERSHEY_DUPLEX, 0.5, color2)

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
        for i in range(16):
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

    # rough_string = xml.etree.ElementTree.tostring(root, 'utf-8')
    # reparsed = minidom.parseString(rough_string)
    # tree = reparsed.toprettyxml(indent="  ")
    #
    # f = open(save_folder + img_name.replace('.png', '.xml').replace('.jpg', '.xml'), 'w')
    # f.write(tree)
    # f.close()

    tree = ElementTree(root)
    tree.write(save_folder + img_name.replace('.png', '.xml').replace('.jpg', '.xml'))

# 
# def save_lane(folder_name, img_name, full_path, save_folder, lane_list, center_list):
#     root = Element('annotation')
#     SubElement(root, 'folder').text = folder_name
#     SubElement(root, 'filename').text = img_name
#     SubElement(root, 'path').text = full_path
# 
#     size = SubElement(root, 'size')
#     SubElement(size, 'width').text = '1920'
#     SubElement(size, 'height').text = '1080'
#     SubElement(size, 'depth').text = '3'
#     for lane in lane_list:
#         tp = lane[0]
#         ct = lane[1]
#         pl = lane[2]
#         ln = lane[3]
# 
#         obj = SubElement(root, 'lane')
#         points = SubElement(obj, 'points')
#         for i in range(16):
#             SubElement(points, 'x%d' % i).text = str(tp[i, 0])
#             SubElement(points, 'y%d' % i).text = str(tp[i, 1])
# 
#         center = SubElement(obj, 'center')
#         SubElement(center, 'x').text = str(ct[0])
#         SubElement(center, 'y').text = str(ct[1])
# 
#         length = SubElement(obj, 'length')
#         SubElement(length, 'start').text = str(ln[0])
#         SubElement(length, 'end').text = str(ln[1])
# 
#         poly = SubElement(obj, 'poly')
#         SubElement(poly, 'p3').text = str(pl[0])
#         SubElement(poly, 'p2').text = str(pl[1])
#         SubElement(poly, 'p1').text = str(pl[2])
#         SubElement(poly, 'p0').text = str(pl[3])
# 
#     for lane in center_list:
#         pl = lane[2]
#         ln = lane[3]
# 
#         obj = SubElement(root, 'center_lane')
# 
#         length = SubElement(obj, 'length')
#         SubElement(length, 'start').text = str(ln[0])
#         SubElement(length, 'end').text = str(ln[1])
# 
#         poly = SubElement(obj, 'poly')
#         SubElement(poly, 'p3').text = str(pl[0])
#         SubElement(poly, 'p2').text = str(pl[1])
#         SubElement(poly, 'p1').text = str(pl[2])
#         SubElement(poly, 'p0').text = str(pl[3])
# 
#     rough_string = xml.etree.ElementTree.tostring(root, 'utf-8')
#     reparsed = minidom.parseString(rough_string)
#     tree = reparsed.toprettyxml(indent="  ")
# 
#     f = open(save_folder + img_name.replace('.png', '.xml').replace('.jpg', '.xml'), 'w')
#     f.write(tree)
#     f.close()
# 
#     # tree = ElementTree.(root)
#     # tree.write(save_folder + img_name.replace('.png', '.xml').replace('.jpg', '.xml'))


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
