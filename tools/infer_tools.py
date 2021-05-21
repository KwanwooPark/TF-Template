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

# lane = [point(16*2), center(2), poly(4), length(2), heat(1), track_duple(1), track_remove(1), numbering(1)]
def save_to_bin(mydata, file_name):
    mydata = mydata.astype(np.float32)
    mydata = mydata.reshape((-1,))
    f = open(file_name, "wb")
    myfmt = 'f' * len(mydata)
    bin = struct.pack(myfmt, *mydata)
    f.write(bin)
    f.close()

def get_point_diff(points, poly):
    pred = np.polyval(poly, points[:, 0])
    return abs(np.mean(pred - points[:, 1]))

def evaluate(gt_points, pred_points, thr_acc):
    gt_lanes = np.zeros((len(gt_points), 480, 800, 3), np.uint8)
    pred_lanes = np.zeros((len(pred_points), 480, 800, 3), np.uint8)
    for idx1, pred_point in enumerate(pred_points):
        pred_point = pred_point[0]
        pred_point = (pred_point + 0.5).astype(np.int)
        for i in range(15):
            pred_lanes[idx1, :, :, :] = cv2.line(pred_lanes[idx1, :, :, :], (pred_point[i, 1], pred_point[i, 0]), (pred_point[i+1, 1], pred_point[i+1, 0]), (255, 255, 255), 30)
    pred_lanes[pred_lanes > 0] = 1
    pred_lanes = pred_lanes[:, :, :, 0]

    for idx1, gt_point in enumerate(gt_points):
        gt_point = gt_point[0]
        gt_point = (gt_point + 0.5).astype(np.int)
        for i in range(15):
            gt_lanes[idx1, :, :, :] = cv2.line(gt_lanes[idx1, :, :, :], (gt_point[i, 1], gt_point[i, 0]),
                                                 (gt_point[i + 1, 1], gt_point[i + 1, 0]), (255, 255, 255), 30)
    gt_lanes[gt_lanes > 0] = 1
    gt_lanes = gt_lanes[:, :, :, 0]
    detected = np.zeros((len(gt_points), len(pred_points)), np.float)
    for i in range(len(gt_points)):
        for j in range(len(pred_points)):
            gt_lane = gt_lanes[i]
            pred_lane = pred_lanes[j]
            mg = gt_lane + pred_lane
            mg[mg > 0] = 1
            mb = gt_lane * pred_lane
            detected[i, j] = np.sum(mb) / np.sum(mg)

    TP = 0
    FP = 0
    FN = 0

    for i in range(len(gt_points)):
        dv = detected[i, :]
        if dv.shape[0] == 0:
            FN += 1
            continue

        dv_min = np.max(dv)
        if dv_min <0.5:
            FN += 1
        else:
            TP += 1

    for i in range(len(pred_points)):
        dv = detected[:, i]
        if dv.shape[0] == 0:
            FP += 1
            continue
        dv_min = np.max(dv)
        if dv_min < 0.5:
            FP += 1

    return TP, FP, FN

def extract_list(hm, os, kp, threshold=0.5, need_nms=False): # w * h * 8 * 2
    if need_nms:
        hm_max = pool2d(hm, kernel_size=5, stride=1, padding=2, pool_mode='max')
        keep = (hm_max == hm).astype(np.float)
        hm = hm * keep

    inds = np.where(hm > threshold)
    x = inds[0]
    y = inds[1]
    lane_list = []
    for ind in range(x.size):
        cx_int = x[ind]
        cy_int = y[ind]

        cx = cx_int * cfg.rate + os[cx_int, cy_int, 0]
        cy = cy_int * cfg.rate + os[cx_int, cy_int, 1]

        point = kp[cx_int, cy_int, :, :].copy()  # 8 * 2
        point[:, 0] += cx
        point[:, 1] += cy
        poly = np.polyfit(point[:, 0], point[:, 1], 3, rcond=0)

        lane_list.append([point, (cx, cy), poly, np.array([point[0, 0], point[-1, 0]]), hm[cx_int, cy_int]])
    return lane_list


def calc_height(lane1, lane2):
    if lane1[3][0] < lane2[3][0]:
        h_low = lane2[3][0]
    else:
        h_low = lane1[3][0]
    if lane1[3][1] < lane2[3][1]:
        h_high = lane1[3][1]
    else:
        h_high = lane2[3][1]
    return h_low, h_high


def diff_nurb(poly1, poly2, h1, h2):
    poly1[0] /= 4
    poly1[1] /= 3
    poly1[2] /= 2
    poly2[0] /= 4
    poly2[1] /= 3
    poly2[2] /= 2
    poly = poly1 - poly2
    w2 = poly[0] * h2 * h2 * h2 * h2 + poly[1] * h2 * h2 * h2 + poly[2] * h2 * h2 + poly[3] * h2
    w1 = poly[0] * h1 * h1 * h1 * h1 + poly[1] * h1 * h1 * h1 + poly[2] * h1 * h1 + poly[3] * h1
    return abs(w2 - w1) / (h2 - h1 + 1e-6)


# lane = [point(16*2), center(2), poly(4), heat(1)]
def tracking(last_list, now_list, thr_diff=100):
    last_num = len(last_list)
    now_num = len(now_list)
    max_num = max(last_num, now_num)
    cost = np.zeros((max_num, max_num), np.float32)
    for last_idx, last_lane in enumerate(last_list):
        for now_idx, now_lane in enumerate(now_list):
            h_low, h_high = calc_height(now_lane, last_lane)
            if h_low > h_high:
                cost[last_idx, now_idx] = 9999
                continue
            diff = diff_nurb(now_lane[2].copy(), last_lane[2].copy(), h_low, h_high)
            if diff > thr_diff:
                cost[last_idx, now_idx] = 9999
            else:
                cost[last_idx, now_idx] = diff
    row_ind, col_ind = linear_sum_assignment(cost)
    for lane in now_list:
        lane.append(1)
        lane.append(0)


    rest_lane = []
    for idx, last_lane in enumerate(last_list):
        matched_lane_ind = col_ind[idx]
        if matched_lane_ind >= now_num:
            rest_lane.append(last_lane)
            continue
        if cost[idx, matched_lane_ind] > thr_diff:
            rest_lane.append(last_lane)
            continue
        now_list[matched_lane_ind][5] = last_lane[5] + 1

    return now_list, rest_lane

def find_lane(rest_lane, pred_hm, pred_os, pred_kp, thr_hm=0.2, thr_diff=100, kernel_half_size=2, thr_remove=3):
    out_lane = []
    for lane in rest_lane:
        # lane = [point(16*2), center(2), poly(4), heat(1)]
        Center_h, Center_w = lane[1]
        Center_h_int, Center_w_int = int(Center_h / cfg.rate), int(Center_w / cfg.rate)
        h_start, h_end = max(Center_h_int - kernel_half_size, 0), min(Center_h_int + kernel_half_size + 1, int(cfg.height / cfg.rate))
        w_start, w_end = max(Center_w_int - kernel_half_size, 0), min(Center_w_int + kernel_half_size + 1, int(cfg.width / cfg.rate))

        hm_cand = pred_hm[h_start:h_end, w_start:w_end]
        max_ind = np.argmax(hm_cand)
        max_h, max_w = int(max_ind / hm_cand.shape[1]), max_ind % hm_cand.shape[1]
        max_h, max_w = max_h + h_start, max_w + w_start
        heat = pred_hm[max_h, max_w]
        if heat < thr_hm:
            lane[6] += 1
            if lane[6] < thr_remove:
                out_lane.append(lane)
            continue

        ch = max_h * cfg.rate + pred_os[max_h, max_w, 0]
        cw = max_w * cfg.rate + pred_os[max_h, max_w, 1]

        point = pred_kp[max_h, max_w, :, :].copy()  # 8 * 2
        point[:, 0] += ch
        point[:, 1] += cw
        poly = np.polyfit(point[:, 0], point[:, 1], 3, rcond=0)
        new_lane = [point, (ch, cw), poly, (point[0, 0], point[-1, 0]), heat, lane[5], 0]

        h_low, h_high = calc_height(new_lane, lane)
        if h_low > h_high:
            lane[6] += 1
            if lane[6] < thr_remove:
                out_lane.append(lane)
            continue
        diff = diff_nurb(new_lane[2].copy(), lane[2].copy(), h_low, h_high)
        if diff > thr_diff:
            lane[6] += 1
            if lane[6] < thr_remove:
                out_lane.append(lane)
            continue

        new_lane[5] += 1
        out_lane.append(new_lane)
    return out_lane



def lane_sorting(lane_list):
    lane_height = []
    for lane in lane_list:
        lane_height.append(lane[4])
    return [x for _, x in sorted(zip(lane_height, lane_list), key=lambda pair: pair[0], reverse=True)]


def lane_NMS(lane_list, thr_diff=50):
    out_lane = []
    while lane_list:
        now_lane = lane_list[0]
        out_lane.append(now_lane)
        new_list = []
        for idx in range(1, len(lane_list)):
            h_low, h_high = calc_height(now_lane, lane_list[idx])
            if h_low > h_high:
                new_list.append(lane_list[idx])
                continue
            diff = diff_nurb(now_lane[2].copy(), lane_list[idx][2].copy(), h_low, h_high)
            if diff > thr_diff:
                new_list.append(lane_list[idx])
                continue
        lane_list = new_list
    return out_lane

def Get_real(lane_list, thr_numd=3):
    out_list = []
    for lane in lane_list:
        if (lane[5] < thr_numd) or (lane[6] > 0):
            continue
        out_list.append(lane)
    return out_list



def lane_labeling(lane_list):
    #312 400
    if len(lane_list) == 0:
        return []
    lane_width_list = []
    for lane in lane_list:
        p = lane[2]
        lane_width_list.append(np.polyval(p, 312))

    lane_width_list = np.array(lane_width_list)
    inds = lane_width_list.argsort()
    lane_sorted = np.array(lane_list)[inds].tolist()

    # tmp_x = zip(lane_width_list, lane_list)
    # tmp_y = sorted(lane_list, key=lambda lane_width_list: lane_width_list)
    # lane_sorted = [x for _, x in tmp_y]
    ind = -1
    for lane in lane_sorted:
        if lane[0][-1][1] < 400:
            ind += 1
    left_num = ind + 1
    right_num = len(lane_sorted) - left_num
    for i in range(left_num):
        lane_sorted[i].append(-i - 1)
    for i in range(right_num):
        lane_sorted[left_num+i].append(i + 1)
    return lane_sorted

def Get_center(lane_list):
    center_list = []
    for i in range(len(lane_list)-1):
        poly_1 = lane_list[i][2]
        poly_2 = lane_list[i + 1][2]
        poly = (poly_1 + poly_2) / 2

        length1 = lane_list[i][3]
        length2 = lane_list[i + 1][3]
        length = np.array([max(length1[0], length2[0]), min(length1[1], length2[1])])

        center_list.append([np.zeros((16, 2)), np.zeros((2,)), poly, length])
    return center_list


def Draw_lane(img, lane_list):
    count = 0
    for lane in lane_list:
        count += 1
        points = lane[0]
        centers = lane[1]
        polys = lane[2]
        length = lane[3]

        cv2.circle(img, (int(centers[1] + 0.5), int(centers[0] + 0.5)), 6, color=color_list[count % 6], thickness=2)
        cv2.putText(img, str(count), (int(centers[1] + 0.5), int(centers[0] - 10.5)), cv2.FONT_HERSHEY_DUPLEX, 0.5, color_list[count % 6])
        for p in range(16):
                h = int(points[p, 0] + 0.5)
                w = int(points[p, 1] + 0.5)
                cv2.circle(img, (w, h), 3, color=color_list[count % 6], thickness=-1)

        h_ = np.arange(length[0], length[1])
        w_ = np.polyval(polys, h_)
        for i in range(h_.shape[0]):
            cv2.circle(img, (int(w_[i] + 0.5), int(h_[i] + 0.5)), 1, color=color_list[count % 6], thickness=-1)

    return img


def save_lane(folder_name, img_name, full_path, save_folder, lane_list, center_list):
    root = Element('annotation')
    SubElement(root, 'folder').text = folder_name
    SubElement(root, 'filename').text = img_name
    SubElement(root, 'path').text = full_path

    size = SubElement(root, 'size')
    SubElement(size, 'width').text = '1920'
    SubElement(size, 'height').text = '1080'
    SubElement(size, 'depth').text = '3'
    for lane in lane_list:
        tp = lane[0]
        ct = lane[1]
        pl = lane[2]
        ln = lane[3]

        obj = SubElement(root, 'lane')
        points = SubElement(obj, 'points')
        for i in range(16):
            SubElement(points, 'x%d' % i).text = str(tp[i, 0])
            SubElement(points, 'y%d' % i).text = str(tp[i, 1])

        center = SubElement(obj, 'center')
        SubElement(center, 'x').text = str(ct[0])
        SubElement(center, 'y').text = str(ct[1])

        length = SubElement(obj, 'length')
        SubElement(length, 'start').text = str(ln[0])
        SubElement(length, 'end').text = str(ln[1])

        poly = SubElement(obj, 'poly')
        SubElement(poly, 'p3').text = str(pl[0])
        SubElement(poly, 'p2').text = str(pl[1])
        SubElement(poly, 'p1').text = str(pl[2])
        SubElement(poly, 'p0').text = str(pl[3])

    for lane in center_list:
        pl = lane[2]
        ln = lane[3]

        obj = SubElement(root, 'center_lane')

        length = SubElement(obj, 'length')
        SubElement(length, 'start').text = str(ln[0])
        SubElement(length, 'end').text = str(ln[1])

        poly = SubElement(obj, 'poly')
        SubElement(poly, 'p3').text = str(pl[0])
        SubElement(poly, 'p2').text = str(pl[1])
        SubElement(poly, 'p1').text = str(pl[2])
        SubElement(poly, 'p0').text = str(pl[3])

    rough_string = xml.etree.ElementTree.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    tree = reparsed.toprettyxml(indent="  ")

    f = open(save_folder + img_name.replace('.png', '.xml').replace('.jpg', '.xml'), 'w')
    f.write(tree)
    f.close()

    # tree = ElementTree.(root)
    # tree.write(save_folder + img_name.replace('.png', '.xml').replace('.jpg', '.xml'))

def lane_revision(lane_list, cut, rate):
    # lane = [point(16*2), center(2), poly(4), length(2), heat(1)]
    out_list = []
    for lane in lane_list:
        point = lane[0]
        center = lane[1]
        point *= rate
        point[:, 0] += cut[0]
        point[:, 1] += cut[1]

        cx = center[0] * 2.25 + cut[0]
        cy = center[1] * 2.25 + cut[1]
        poly = np.polyfit(point[:, 0], point[:, 1], 3, rcond=0)
        out_list.append([point, (cx, cy), poly, np.array([point[0, 0], point[-1, 0]]), lane[4], lane[5], lane[6]])
    return out_list
