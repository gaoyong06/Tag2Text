'''
Author: gaoyong gaoyong06@qq.com
Date: 2023-06-08 11:44:38
LastEditors: gaoyong gaoyong06@qq.com
LastEditTime: 2023-06-13 21:45:55
FilePath: \Tag2Text\image_caption_dir.py
Description: 自动生成图片目录下的图片标签和内容描述
'''
import argparse
import imghdr
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from loguru import logger
import csv
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
from models.tag2text import tag2text_caption
import torch
from torchvision import transforms
import pymysql.cursors


EXPIRATION_DAYS = 14
FILE_LIST_SUFFIX = '_file_list.csv'


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


def initialize_model(cache_path, pretrained, image_size, thre):
    """
    This function initializes a Tag2Text model based on specified and identified tags.
    :param cache_path: Cache model file path.
    :param pretrained: Path to the pre-trained model.
    :param image_size: Input image size.
    :param thre: Threshold value for tagging.
    :return: A pre-trained Tag2Text model.
    """

    # delete some tags that may disturb captioning
    # 127: "quarter"; 2961: "back", 3351: "two"; 3265: "three"; 3338: "four"; 3355: "five"; 3359: "one"
    delete_tag_index = [127, 2961, 3351, 3265, 3338, 3355, 3359]

    if os.path.exists(cache_path):
        model = torch.load(cache_path)
    else:
        model = tag2text_caption(
            pretrained=pretrained,
            image_size=image_size,
            vit='swin_b',
            delete_tag_index=delete_tag_index
        )
        model.threshold = thre  # threshold for tagging
        model.eval()
        torch.save(model, cache_path)

    return model


def generate(model, image, input_tags=None):
    """
    This function generates tags and captions for an input image.
    :param model: The neural network model used for generating captions and predicting tags for an input
    image.
    :param image: The input image to generate tags and captions for.
    :param input_tags: The input tags used as hints for the model to generate captions for the input image.
    It is an optional parameter and can be set to None or left empty if no tag hint is required.
    :return: A tuple of predicted tags, input tags, and generated captions.
    """

    if input_tags in ('', 'none', 'None'):
        input_tags = None

    with torch.no_grad():
        caption, tag_predict = model.generate(image,
                                              tag_input=None,
                                              max_length=50,
                                              return_tag_predict=True)

    if input_tags is None:
        return tag_predict[0], None, caption[0]

    input_tag_list = [input_tags.replace(',', ' | ')]
    with torch.no_grad():
        caption, input_tags = model.generate(image,
                                             tag_input=input_tag_list,
                                             max_length=50,
                                             return_tag_predict=True)
    return tag_predict[0], input_tags[0], caption[0]


def get_file_list_csv(image_dir):
    """
    Get the file list CSV filename associated with the specified image directory.
    """
    logger.info("start")
    dir_name = os.path.basename(os.path.normpath(image_dir))
    csv_file = os.path.join(image_dir, dir_name + FILE_LIST_SUFFIX)
    return csv_file


def create_file_list(image_dir):
    """
    Create a list of image file paths in the specified directory and its subdirectories.
    """
    logger.info("start")
    # 用于存放图像文件的路径
    file_list = []

    # 递归遍历指定目录及其子目录，并将每个图像文件的路径添加到 file_list 列表中
    for root, dirs, files in os.walk(image_dir):
        for filename in files:
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png') or filename.endswith('.webp'):
                filepath = os.path.join(root, filename)
                filepath = filepath.replace("\\", "/")
                file_list.append(filepath)

    return file_list


def save_file_list(file_list, csv_file):
    """
    Save the list of file paths to a CSV file.
    """
    logger.info(f"start. {csv_file}")
    with open(csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['filepath'])
        for filepath in file_list:
            csvwriter.writerow([filepath])


def load_file_list_csv(image_dir):
    """
    Load the list of file paths from a CSV file if it is not expired.
    """
    logger.info("start")
    csv_file = get_file_list_csv(image_dir)

    if os.path.isfile(csv_file):
        mod_time = os.path.getmtime(csv_file)
        exp_time = datetime.fromtimestamp(mod_time + EXPIRATION_DAYS*24*60*60)
        if exp_time >= datetime.now():
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                file_list = [row[0] for row in reader]
            return file_list

    # 如果 CSV 文件不存在或已过期，则重新生成新的文件列表
    file_list = create_file_list(image_dir)
    save_file_list(file_list, csv_file)
    return file_list


def check_file_processed(db_conn, filepath):
    """ Check whether a file has been processed and inserted into the database before or not. """
    sql = "SELECT `tags`, `caption` FROM `tbl_image_caption` WHERE `local_path`=%s"
    try:
        with db_conn.cursor() as cursor:

            # 检查该图像是否已被处理。
            filepath = filepath.replace("\\", "/")
            cursor.execute(sql, (filepath,))
            result = cursor.fetchone()
            if result is not None and result["tags"] and result["caption"]:
                logger.info(f"{filepath} has been processed. Skipping.")
                return True, result["tags"], result["caption"]
    except Exception as e:
        logger.error(
            f"Error check file processed. {str(e)}")

    return False, None, None


def insert_into_database(db_conn, insert_image_list):
    """ Batch insert data into the database """
    if len(insert_image_list) > 0:
        try:
            with db_conn.cursor() as cursor:
                sql = "INSERT INTO `tbl_image_caption` (`local_path`, `tags`, `caption`) VALUES (%s, %s, %s)"
                for insert_image in insert_image_list:
                    if len(insert_image) != 3 or not insert_image[0] or not insert_image[1] or not insert_image[2]:
                        logger.warning(f"Invalid input data: {insert_image}")
                        continue
                    cursor.execute(sql, insert_image)
                db_conn.commit()
                return True, ''
        except Exception as e:
            logger.error(
                f"Error Inserted {len(insert_image_list)} records into database. {str(e)}")
            return False, str(e)
    return True, ''


def connect_to_database(db_host, db_port, db_user, db_pass, db_name):
    return pymysql.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_pass,
        db=db_name,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )


def worker_thread(batch_files: List[str], device, model, image_size, input_tags, db_config):
    """ Process a batch of image files, generate their tags and captions, and insert them into the database. """
    # 初始化已处理文件路径列表和待插入数据库的图像列表
    processed_files = []
    insert_image_list = []

    # 定义标准化及变换过程
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])

    try:
        # 获取数据库连接对象并使用
        with connect_to_database(db_config["db_host"], db_config["db_port"], db_config["db_user"], db_config["db_pass"], db_config["db_name"]) as db_conn:
            logger.info("Successfully connected to database.")
            for filepath in batch_files:
                logger.info(f"Processing file: {filepath}")
                
                filepath = filepath.replace("\\", "/")
                # 检查该文件是否为图像文件。
                if imghdr.what(filepath) is None:
                    logger.warning(
                        f"{filepath} is not an image file. Skipping.")
                    continue

                # 检查该图像是否已被处理。
                is_processed, tags, caption = check_file_processed(
                    db_conn, filepath)
                if is_processed:
                    logger.info(
                        f"{filepath} has been processed. Skipping.")
                    continue
                
                # 处理图像。
                try:
                    with open(filepath, 'rb') as f:
                        img = Image.open(f).convert("RGB")
                        img_tensor = transform(img).unsqueeze(0).to(device)
                        # 生成标签和说明
                        tags, input_tags, caption = generate(
                            model, img_tensor, input_tags)
                        logger.debug(f"Tags: {tags}. Caption: {caption}")
                        # 将需要插入数据库的记录加入列表 insert_image_list
                        insert_image_list.append((filepath, tags, caption))
                        # 如果 insert_image_list 中待插入的图片记录数量超过阈值，则进行批量插入操作
                        if len(insert_image_list) >= 100:
                            success, error = insert_into_database(
                                db_conn, insert_image_list)
                            if success:
                                logger.info(
                                    f"Inserted {len(insert_image_list)} records into database.")
                                insert_image_list.clear()
                            else:
                                logger.error(error)
                        logger.info(f"{filepath} processed successfully. ")
                        # 如果执行到这里，则表示该文件已处理
                        processed_files.append(filepath)
                except Exception as e:
                    logger.error(
                        f"Failed to process {filepath}. Error message: {str(e)}")
                    continue

            # 最后将 insert_image_list 中的内容插入到数据库中
            if len(insert_image_list) > 0:
                success, error = insert_into_database(
                    db_conn, insert_image_list)
                if success:
                    logger.info(
                        f"Inserted {len(insert_image_list)} records into database.")
                    insert_image_list.clear()
                else:
                    logger.error(error)

            logger.info(f"Batch of {len(batch_files)} images processed.")

    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
    finally:
        logger.info("Successfully disconnected from database.")

    # 返回已处理的文件路径列表
    return processed_files


def process_directory_mp(db_config, image_dir, model, image_size, input_tags=None, batch_size=1000, max_workers=6):
    """
     Iterate over all image files in the specified directory and generate tags and caption for each image using the specified model.
     multiprocessing is used in parallel.
     """
    logger.info(
        f"Processing directory (parallel, batching {batch_size}): {image_dir}")
    file_list = load_file_list_csv(image_dir)
    total_files = len(file_list)
    total_batches = (total_files + batch_size - 1) // batch_size
    logger.info(f"Total number of files: {total_files}")
    logger.info(f"Total number of batches: {total_batches}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, total_files)
            batch_files = file_list[start_idx:end_idx]
            future = executor.submit(
                worker_thread, batch_files, device, model, image_size, input_tags, db_config)
            futures.append(future)
            logger.info(
                f"Batch {i+1}/{total_batches} of {len(batch_files)} images submitted.")

        processed_files = []
        for future in as_completed(futures):
            try:
                res = future.result()
                if res is not None:
                    processed_files.extend(res)
                    logger.info(
                        f"Batch processed. {len(res)} images processed.")
            except Exception as exc:
                logger.error(f"Exception occurred: {exc}")
            else:
                logger.info("Batch successfully completed.")

    logger.info("Workers finished.")
    return processed_files


def main():
    """
    This is the main function that loads the model, sets up the database connection,
    and processes the specified image directory or list of images using multiprocessing.
    """
    logger.info("main start")
    # parse command line arguments
    args = parse_args()
    logger.info("Command line arguments parsed.")

    # set up the database connection pool
    db_config = {
        'db_host': args.db_host,
        'db_port': args.db_port,
        'db_user': args.db_user,
        'db_pass': args.db_pass,
        'db_name': args.db_name
    }
    # initialize the model
    logger.info("Preparing to initialize the model...")
    model = initialize_model(
        args.cache_path, args.pretrained, args.image_size, args.thre)
    logger.info("Model initialization completed.")

    # process the input images
    logger.info(f"Processing images in directory: {args.image_dir}")
    processed_files = process_directory_mp(
        db_config, args.image_dir, model, args.image_size, args.specified_tags)
    logger.info(f"{len(processed_files)} images processed.")
    logger.info("Main function completed.")


# 使用示例：python image_caption_dir.py --cache-path C:/Users/gaoyo/.cache/Tag2Text --image-dir D:/work/wechat_download_data/images/ --db-host 127.0.0.1 --db-port 3306 --db-user root --db-pass root --db-name content_ner
if __name__ == "__main__":
    main()
