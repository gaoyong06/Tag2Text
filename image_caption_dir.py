'''
Author: gaoyong gaoyong06@qq.com
Date: 2023-06-08 11:44:38
LastEditors: gaoyong gaoyong06@qq.com
LastEditTime: 2023-06-08 16:59:32
FilePath: \Tag2Text\test.py
Description: 自动生成图片目录下的图片标签和内容描述
'''
import argparse
import imghdr
import json
import os
import time

import pymysql.cursors
import torch
import torchvision.transforms as transforms
from PIL import Image
from image_caption import initialize_model, generate, inference
import datetime

def parse_args():
    """
    This function parses command line arguments for a Tag2Text inference model.
    :return: The function `parse_args()` is returning the parsed arguments from the command line using
    the `argparse` module.
    """
    parser = argparse.ArgumentParser(
        description='Tag2Text inference for tagging and captioning')
    parser.add_argument('--image-dir',
                        metavar='DIR',
                        help='path to directory containing input images',
                        default='images/')
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

def insert_image(db_conn, image_path, tags, caption):
    """
    This function inserts an image’s information into the database.
    :param db_conn: A database connection object.
    :param image_path: The image’s local path on disk.
    :param tags: The image’s predicted tags.
    :param caption: The image’s generated caption.
    :return: True if the image was inserted successfully, False otherwise.
    """
    try:
        with db_conn.cursor() as cursor:
            sql = "SELECT image_id FROM tbl_image_caption WHERE local_path=%s AND tags<>'' AND caption<>''"
            cursor.execute(sql, (image_path,))
            if cursor.rowcount > 0:
                print(f"{image_path} already processed. Skipping.")
                return False
            
            sql = "INSERT INTO tbl_image_caption (local_path, tags, caption) VALUES (%s, %s, %s)"
            cursor.execute(sql, (image_path, tags, caption))
            print(f"{cursor.rowcount} rows inserted.")
            
        db_conn.commit()  # commit changes to the database
        return True

    except Exception as ex:
        db_conn.rollback()  # undo changes on error
        print(f"Failed to insert {image_path} into database: {ex}")
        return False

def process_directory(db_conn, image_dir, model, image_size, input_tags=None):
    """
    This function processes a directory or list of images, generates captions and tags, and writes the
    results to the database.
    :param db_conn: A database connection object.
    :param image_dir: A directory path containing a set of images to process. All images in the directory and its subdirectories will be processed.
    :param model: The trained Tag2Text image captioning model.
    :param image_size: The desired input image size for the model.
    :param input_tags: Optional user-specified tags to use for caption generation.
    """
    print(f"Processing directory: {image_dir}")
    if input_tags is not None:
        print(f"Using input tags: {input_tags}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), normalize
    ])

    total_processed = 0
    skipped_files = 0
    total_files = sum([len(files) for r, d, files in os.walk(image_dir)])

    for root, dirs, files in os.walk(image_dir):
        for filename in files:
            filepath = os.path.abspath(os.path.join(root, filename))
            if not imghdr.what(filepath):
                continue

            # check if the image has been processed before
            filepath = filepath.replace("\\", "/")
            with db_conn.cursor() as cursor:
                sql = "SELECT `tags`, `caption` FROM `tbl_image_caption` WHERE `local_path`=%s"
                cursor.execute(sql, (filepath,))
                result = cursor.fetchone()
            if result is not None and result["tags"] and result["caption"]:
                print(f"{filepath} already processed. Skipping.")
                skipped_files += 1
                continue

            # process the image and insert the result into the database
            print(f"Processing file: {filepath}")
            start_time = datetime.datetime.now()

            img = Image.open(filepath).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            res = generate(model, img_tensor, input_tags)
            tags, input_tags, caption = res
            insert_image(db_conn, filepath, tags, caption)

            end_time = datetime.datetime.now()
            process_time = end_time - start_time
            print(f"Processed in {process_time.total_seconds()} seconds.")

            total_processed += 1
            progress = (total_processed + skipped_files) / total_files * 100
            print(f"Progress: {progress:.2f}% ({total_processed + skipped_files}/{total_files})")

    print(f"Total processed: {total_processed}, skipped files: {skipped_files}")


def main():
    """
    This is the main function that loads the model, sets up the database connection, and processes the
    specified image directory or list of images.
    """
    # parse command line arguments
    args = parse_args()

    # set up the database connection
    db_conn = pymysql.connect(
        host=args.db_host,
        port=args.db_port,
        user=args.db_user,
        password=args.db_pass,
        db=args.db_name,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

     # initialize the model
    model = initialize_model(
        args.cache_path, args.pretrained, args.image_size, args.thre)

    # process the input images
    process_directory(db_conn, args.image_dir, model,
                    args.image_size, args.specified_tags)

    # close the database connection
    db_conn.close()

    print("Done.")

# >python image_caption_dir.py --cache-path C:/Users/gaoyo/.cache/Tag2Text --image-dir C:/Users/gaoyo/Desktop/test1 --db-host 127.0.0.1 --db-port 3306 --db-user root --db-pass root --db-name content_ner
# python image_caption_dir.py --cache-path C:/Users/gaoyo/.cache/Tag2Text --image-dir D:/work/wechat_download_data/images/ --db-host 127.0.0.1 --db-port 3306 --db-user root --db-pass root --db-name content_ner
if __name__ == '__main__':
    main()

