# _*_ coding:utf-8 _*_
"""
@File     : split_dataset.py
@Project  : develop
@Time     : 2025/2/28/028 9:56
@Author   : Yan Qinglin
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2025/2/28/028 9:56        1.0             None
"""
import os
import random
import shutil
import math

from typing import List, Optional


def get_label_path(img_path, label_folder='labels', suffix='txt'):
    """
    生成标签文件路径

    :param img_path: 原始图像路径（如：E:/Data/.../images/001.jpg）
    :param label_folder: 标签目录名称（默认：labels）
    :param suffix: 标签文件后缀（默认：txt）
    :return: 对应的标签文件路径
    """
    # 分解原始路径
    img_dir = os.path.dirname(img_path)
    parent_dir = os.path.dirname(img_dir)

    # 构建标签目录路径
    label_dir = os.path.join(parent_dir, label_folder)

    # 处理文件名
    filename = os.path.basename(img_path)
    name_without_ext = os.path.splitext(filename)[0]

    # 组合最终路径
    return os.path.join(label_dir, f"{name_without_ext}.{suffix}")


def split_dataset(
        dataset_dirs: List[str],
        save_dir: Optional[str] = None,
        split_names: Optional[List[str]] = None,
        split_ratios: Optional[List[float]] = None,
        image_folder: str = "images",
        label_folders: Optional[List[str]] = None,
        operation: str = "copy",
        shuffle: bool = True,
        unlabeled_ratio: float = 1.0,
        label_suffix: str = "txt"  # 新增参数
    ):
    """
    分割数据集
    :param dataset_dirs: 数据集路径集合，会遍历每一个文件夹
    :param save_dir: 数据集路径,未提供时将使用dataset_dirs中第一个文件夹作为输出路径
    :param split_names: 分割后数据集的名称，默认为 ["train", "eval", "test"]
    :param split_ratios: 分割比例，默认为 [0.9, 0.1, 0]
    :param image_folder: 图像文件夹名称，默认为 "images"，只能提供一个图像文件夹名称
    :param label_folders: 标签文件夹名称列表，默认为 ["labels"]，提供多个标签文件夹名称将会分别对每个都进行分割
    :param operation: 操作方式，可选值为 "copy", "move", "txt"，默认为 "copy"即复制分割后的结果，包括图像和标签文件
    :param shuffle: 是否打乱后进行分割
    :param unlabeled_ratio: 无标签文件采样比例，默认1.0即对无标签文件也全部采样
    :param label_suffix: 标签文件后缀格式
    """
    if split_ratios is None:
        split_ratios = [0.9, 0.1, 0]
    if label_folders is None:
        label_folders = ["labels"]
    if split_names is None:
        split_names = ["train", "eval", "test"]
    if save_dir is None:
        save_dir = dataset_dirs[0]
    assert sum(split_ratios) == 1, "分割比例之和必须为 1"
    if not all(os.path.isdir(d) for d in dataset_dirs):
        raise ValueError("无效的dataset_dirs参数")
    total_files = []
    remaining_unlabeled = []
    for dataset_path in dataset_dirs:
        image_dir = os.path.join(dataset_path, image_folder)
        if not os.path.exists(image_dir):
            print(f"图像文件夹 {image_dir} 不存在")
            return

        image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

        # 分离有标签和无标签的图片
        labeled_images = []
        unlabeled_images = []

        for f in image_files:
            base_name = os.path.splitext(f)[0]
            # has_label = any(
            #     os.path.exists(label_path := os.path.join(dataset_path, lf, base_name + label_suffix))
            #     and os.path.getsize(label_path) > 0  # 新增非空校验
            #     for lf in label_folders
            # )
            has_label = any(
                (os.path.exists(label_path := os.path.join(dataset_path, lf, base_name + '.' + label_suffix))
                 and os.path.getsize(label_path) > 0)
                for lf in label_folders
            )
            (labeled_images if has_label else unlabeled_images).append(f)
        print(labeled_images)

        # 对无标签图片进行采样
        sampled_unlabeled = []

        if unlabeled_ratio > 0 and len(unlabeled_images) > 0:
            keep_num = int(len(unlabeled_images) * unlabeled_ratio)
            sampled_unlabeled = random.sample(unlabeled_images, keep_num)
            remaining_unlabeled += [img for img in unlabeled_images if img not in sampled_unlabeled]
        else:
            remaining_unlabeled += unlabeled_images.copy()

        effective_images = labeled_images + sampled_unlabeled
        effective_images = [os.path.join(image_dir, image_name) for image_name in effective_images]
        total_files += effective_images

    if shuffle:
        random.shuffle(total_files)

    total = len(total_files)
    split_indices = [0]
    accumulated = 0
    for ratio in split_ratios[:-1]:
        accumulated += ratio
        split_indices.append(math.ceil(total * accumulated))
    split_indices.append(total)
    print(split_indices)

    for i, (start, end) in enumerate(zip(split_indices[:-1], split_indices[1:])):
        split_name = split_names[i]
        split_images = total_files[start:end]

        if split_name == 'test':
            split_images += remaining_unlabeled

        if operation == "txt":
            txt_path = os.path.join(save_dir, f"{split_name}.txt")
            with open(txt_path, "w") as f:
                for image in split_images:
                    f.write(image + "\n")
        else:
            split_image_dir = os.path.join(save_dir, split_name, image_folder)
            os.makedirs(split_image_dir, exist_ok=True)

            for label_folder in label_folders:
                split_label_dir = os.path.join(save_dir, split_name, label_folder)
                os.makedirs(split_label_dir, exist_ok=True)

                for image_file in split_images:
                    label_file = get_label_path(image_file, label_folder, suffix=label_suffix)

                    if operation == "copy":
                        shutil.copy2(image_file, split_image_dir)
                        if os.path.exists(label_file):
                            shutil.copy2(label_file, split_label_dir)
                    elif operation == "move":
                        shutil.move(image_file, split_image_dir)
                        if os.path.exists(label_file):
                            shutil.move(label_file, split_label_dir)


if __name__ == "__main__":
    datasets = [
        r"/media/ai/Data_SSD1/yql/datasets/kc/customized/nmsr/old",
        # '/media/ai/Data_SSD2/yql/kc/hxjs/face',
    ]
    txt_names = ['001', '002', '003', '004', '005']
    # [0.2, 0.2, 0.2, 0.2, 0.2]
    split_dataset(datasets, shuffle=True, operation="txt", image_folder='images', split_names=None, label_folders=['labels'],
                  split_ratios=[0.9, 0.1, 0.0], unlabeled_ratio=1.0, label_suffix='txt')


