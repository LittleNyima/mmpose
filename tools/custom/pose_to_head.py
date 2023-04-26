import argparse
import os
import json
import cv2
from tqdm import tqdm
from mmcv import Config


def kp2head(result):
    for r in result["results"]:
        kps = r["keypoints"]
        valid = [i for i in (2, 5, 8, 11, 14) if kps[i]]
        if not valid:
            r["head"] = []
            continue
        # head center
        xs = [kps[v-2] for v in valid]
        ys = [kps[v-1] for v in valid]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        xcenter = (xmin + xmax) // 2
        ycenter = (ymin + ymax) // 2
        # padding
        valid = [i for i in (17, 20, 35, 38) if kps[i]]
        if valid:
            xs = [kps[v-2] for v in valid]
            ys = [kps[v-1] for v in valid]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            pad = max(xmax - xmin, ymax - ymin) / 1.5
            pad = int(pad)
            pad = max(20, pad)
        else:
            pad = 20
        edge = pad
        # edge = max(xmax - xmin, ymax - ymin) + pad
        half = edge // 2
        r["head"] = [xcenter - half, ycenter - half, xcenter + half, ycenter + half]
        r["head"] = list(map(int, r["head"]))
    return result


def vis(result, img_root, vis_root):
    img = cv2.imread(os.path.join(img_root, result["image"]["file_name"]))
    for r in result["results"]:
        if r["head"]:
            img = cv2.rectangle(img, tuple(r["head"][:2]), tuple(r["head"][2:]), (0, 0, 255))
    os.makedirs(os.path.dirname(os.path.join(vis_root, result["image"]["file_name"])), exist_ok=True)
    cv2.imwrite(os.path.join(vis_root, result["image"]["file_name"]), img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    config = Config.fromfile(args.config)
    
    ann = config.data.val.ann_file
    img_root = config.data.val.img_prefix
    vis_root = os.path.join(img_root, "../" "vis2017")

    with open(ann, "r") as f:
        ann = json.load(f)
    os.makedirs(vis_root, exist_ok=True)
    id2image = dict()
    id2results = dict()
    for item in ann["images"]:
        id2image[item['id']] = item
    for item in ann["annotations"]:
        id2results.setdefault(item["image_id"], [])
        id2results[item["image_id"]].append(item)
    for k, v in tqdm(id2results.items()):
        results = kp2head({
            "image": id2image[k],
            "results": v,
        })
        vis(results, img_root, vis_root)
    with open(os.path.join(img_root, "..", "annotations_with_head.json"), "w") as fout:
        json.dump(ann, fout, indent=2)
