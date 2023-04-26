import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=str, required=True)
    parser.add_argument("--ann", type=str, required=True)
    args = parser.parse_args()

    with open(args.ann, "r") as fin:
        ann = json.load(fin)
    with open(args.res, "r") as fin:
        res = json.load(fin)
    ann["annotations"] = res
    with open(args.ann, "w") as fout:
        json.dump(ann, fout, indent=2)
