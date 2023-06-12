# -*- coding: utf-8 -*-
'''
Author: gaoyong gaoyong06@qq.com
Date: 2023-06-08 10:51:43
LastEditors: gaoyong gaoyong06@qq.com
LastEditTime: 2023-06-12 17:41:36
FilePath: \Tag2Text\image_caption.py
Description: 自动生成图片标签和内容描述
'''
import argparse
import json
import os
import time

import imghdr
import torch
import torchvision.transforms as transforms
from PIL import Image
from image_caption_dir import initialize_model, generate, connect_to_database, insert_into_database, check_file_processed
from loguru import logger


def parse_args():
    """
    This function parses command line arguments for a Tag2Text inference model.
    :return: The function `parse_args()` is returning the parsed arguments from the command line using
    the `argparse` module.
    """
    parser = argparse.ArgumentParser(
        description='Tag2Text inference for tagging and captioning')
    parser.add_argument('--images',
                        metavar='IMAGE-LIST',
                        nargs='+',
                        help='list of space-separated image filenames',
                        default=[])
    parser.add_argument('--pretrained',
                        metavar='DIR',
                        help='path to pretrained model',
                        default='D:/work/Tag2Text/pretrained/tag2text_swin_14m.pth')
    parser.add_argument('--image-size',
                        default=384,
                        type=int,
                        metavar='N',
                        help='input image size (default: 448)')
    parser.add_argument('--thre',
                        default=0.68,
                        type=float,
                        metavar='N',
                        help='threshold value')
    parser.add_argument('--specified-tags',
                        default='None',
                        help='User input specified tags')
    parser.add_argument('--cache-path',
                        default='None',
                        help='cache model file path')
    parser.add_argument('--db-host',
                        default='localhost',
                        help='MySQL host')
    parser.add_argument('--db-port',
                        default=3306,
                        type=int,
                        help='MySQL port')
    parser.add_argument('--db-user',
                        default='root',
                        help='MySQL username')
    parser.add_argument('--db-pass',
                        default='root',
                        help='MySQL password')
    parser.add_argument('--db-name',
                        default='content_ner',
                        help='MySQL database name')

    return parser.parse_args()


def inference(db_conn, image_list, model, image_size, input_tags=None):
    """
    This function takes a list of images or a directory containing images, a model, generates captions
    for the images, and optionally takes a list of input tags to generate captions with those tags.
    :param image_list: A list of input images the model will use to generate captions and potentially
    predict tags for.
    :param model: The neural network model used for generating captions and predicting tags for an input
    image.
    :param input_tags: The input tags are lists of strings that represent tags or sets of tags that are
    used as hints for the model to generate captions for the given images. It is an optional parameter and
    can be set to None or left empty if no tag hint is required, defaults to None.
    :return: A list of dictionaries, each containing predicted tags, input tags (if provided), and
    generated captions for a given input image.
    """

    results = []
    insert_image_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), normalize
    ])

    if image_list and isinstance(image_list, list):
        for img_path in image_list:
            filepath = os.path.abspath(img_path)
            if not os.path.isfile(filepath) or not imghdr.what(filepath):
                logger.warning(f"Skipping invalid image file: {filepath}")
                continue
            is_processed, tags, caption = check_file_processed(
                db_conn, filepath)

            if not is_processed:
                img = Image.open(filepath).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                res = generate(model, img_tensor, input_tags)

                # 写入数据库
                tags, input_tags, caption = res
                insert_image_list.append((filepath, tags, caption))

            results.append({
                "filepath": filepath,
                "model_identified_tags": tags,
                "user_specified_tags": input_tags,
                "image_caption": caption
            })

        # 如果 insert_image_list 中待插入的图片记录数量超过阈值，则进行批量插入操作
        if len(insert_image_list) > 0:
            success, error = insert_into_database(
                db_conn, insert_image_list)
            if success:
                logger.info(
                    f"Inserted {len(insert_image_list)} records into database.")
                insert_image_list.clear()
            else:
                logger.error(error)
        logger.info(f"Processed {len(results)} files successfully.")

    return results


def main():
    """
    This function loads a pre-trained image captioning model, processes input images in a directory,
    and generates captions for each image based on specified and identified tags.
    """
    start_time = time.time()
    args = parse_args()

    # set up the database connection pool
    db_config = {
        'db_host': args.db_host,
        'db_port': args.db_port,
        'db_user': args.db_user,
        'db_pass': args.db_pass,
        'db_name': args.db_name
    }

    # check if a list of images is provided
    images = args.images if args.images else None
    # initialize the model
    model = initialize_model(
        args.cache_path, args.pretrained, args.image_size, args.thre)

    # write db
    try:
        # 获取数据库连接对象并使用
        db_conn = connect_to_database(
            db_config["db_host"], db_config["db_port"], db_config["db_user"], db_config["db_pass"], db_config["db_name"])
        logger.info("Successfully connected to database.")

        # perform inference on images
        data = inference(db_conn, images, model,
                         args.image_size, input_tags=None)

    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
    finally:
        # 释放数据库连接
        db_conn.close()
        logger.info("Successfully disconnected from database.")

    # output the results
    results = {
        "status": 0,
        "message": 'ok',
        "data": data
    }

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(
        f"Processed {len(results['data'])} images in {elapsed_time:.2f} seconds.")

    print(json.dumps(results, ensure_ascii=False, indent=2))

# 使用示例：
# python image_caption.py --cache-path C:/Users/gaoyo/.cache/Tag2Text --images C:/Users/gaoyo/Desktop/test1/20170912_234158_1_14_70wf.jpeg C:/Users/gaoyo/Desktop/test1/20170918_192435_1_57_9yto.jpeg


if __name__ == '__main__':
    main()
