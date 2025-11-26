import argparse
import os
import subprocess
import shutil
import sys


def run(cmd):
    print("运行:" + " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def find_json(root, name):
    for dirpath, _, filenames in os.walk(root):
        if name in filenames:
            return os.path.join(dirpath, name)
    return None


def move_dir(src, dst):
    if not os.path.exists(src):
        return False
    ensure_dir(os.path.dirname(dst))
    if os.path.exists(dst):
        print(f"目标已存在: {dst}")
    else:
        shutil.move(src, dst)
    return True


def copy_dir(src, dst):
    if not os.path.exists(src):
        return False
    ensure_dir(os.path.dirname(dst))
    if os.path.exists(dst):
        print(f"目标已存在: {dst}")
    else:
        shutil.copytree(src, dst)
    return True


def unpack_all(root):
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            lower = f.lower()
            if lower.endswith((".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz")):
                out = os.path.join(dirpath, f + "_extracted")
                ensure_dir(out)
                try:
                    shutil.unpack_archive(fp, out)
                except Exception as e:
                    print(f"解压失败: {fp}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    args = parser.parse_args()
    base = args.base_dir
    if shutil.which("odl") is None:
        print("未检测到 OpenDataLab CLI: 请先执行 pip install opendatalab，并运行 odl login")
        sys.exit(1)
    ensure_dir(base)
    tmp = os.path.join(base, "_odl_tmp")
    ensure_dir(tmp)
    run(["odl", "get", "OpenDataLab/Objects365", "--dest", tmp])
    unpack_all(tmp)
    extracted = tmp
    train_images_v1 = None
    train_images_v2 = None
    val_images_v1 = None
    val_images_v2 = None
    for dirpath, dirnames, _ in os.walk(extracted):
        if os.path.basename(dirpath) == "v1" and os.path.basename(os.path.dirname(dirpath)) == "images":
            parent = os.path.basename(os.path.dirname(os.path.dirname(dirpath)))
            if parent == "train":
                train_images_v1 = dirpath
            elif parent == "val":
                val_images_v1 = dirpath
        if os.path.basename(dirpath) == "v2" and os.path.basename(os.path.dirname(dirpath)) == "images":
            parent = os.path.basename(os.path.dirname(os.path.dirname(dirpath)))
            if parent == "train":
                train_images_v2 = dirpath
            elif parent == "val":
                val_images_v2 = dirpath
    train_json = find_json(extracted, "zhiyuan_objv2_train.json")
    val_json = find_json(extracted, "zhiyuan_objv2_val.json")
    ensure_dir(os.path.join(base, "train", "images", "v1"))
    ensure_dir(os.path.join(base, "train", "images", "v2"))
    ensure_dir(os.path.join(base, "val", "images", "v1"))
    ensure_dir(os.path.join(base, "val", "images", "v2"))
    if train_images_v1:
        move_dir(train_images_v1, os.path.join(base, "train", "images", "v1"))
    if train_images_v2:
        move_dir(train_images_v2, os.path.join(base, "train", "images", "v2"))
    if val_images_v1:
        move_dir(val_images_v1, os.path.join(base, "val", "images", "v1"))
    if val_images_v2:
        move_dir(val_images_v2, os.path.join(base, "val", "images", "v2"))
    if train_json:
        shutil.move(train_json, os.path.join(base, "train", "zhiyuan_objv2_train.json"))
    if val_json:
        shutil.move(val_json, os.path.join(base, "val", "zhiyuan_objv2_val.json"))
    ensure_dir(os.path.join(base, "train", "images_from_val"))
    copy_dir(os.path.join(base, "val", "images", "v1"), os.path.join(base, "train", "images_from_val", "v1"))
    copy_dir(os.path.join(base, "val", "images", "v2"), os.path.join(base, "train", "images_from_val", "v2"))
    remap = os.path.join(os.path.dirname(__file__), "remap_obj365.py")
    resize = os.path.join(os.path.dirname(__file__), "resize_obj365.py")
    run(["python", remap, "--base_dir", base])
    run(["python", resize, "--base_dir", base])
    print("完成")


if __name__ == "__main__":
    main()