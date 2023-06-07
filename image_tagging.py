# -*- coding: utf-8 -*-
'''
Author: gaoyong gaoyong06@qq.com
Date: 2023-05-25 08:51:43
LastEditors: gaoyong gaoyong06@qq.com
LastEditTime: 2023-06-06 15:18:23
FilePath: \Tag2Text\image_tagging.py
Description: 自动生成图片标签和内容描述 (程序逻辑本身和inference.py是相同的,只是做了一些封装和输出格式的调整)
'''
import argparse
import json
import os

import torch
import torchvision.transforms as transforms

from PIL import Image
from models.tag2text import tag2text_caption
# import translators as ts
import imghdr

def parse_args():
    """
    This function parses command line arguments for a Tag2Text inference model.
    :return: The function `parse_args()` is returning the parsed arguments from the command line using
    the `argparse` module.
    """
    parser = argparse.ArgumentParser(
        description='Tag2Text inference for tagging and captioning')
    parser.add_argument('--image',
                        metavar='DIR',
                        help='path to dataset',
                        default='images/1641173_2291260800.jpg')
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

    return parser.parse_args()



def inference(image, model, input_tag="None"):
    """
    This function takes an image and a model, generates a caption for the image, and optionally takes an
    input tag to generate a caption with that tag.
    
    :param image: an input image that the model will use to generate a caption and potentially predict
    tags for
    :param model: The neural network model used for generating captions and predicting tags for an input
    image
    :param input_tag: The input tag is a string that represents a tag or a set of tags that are used as
    a hint for the model to generate a caption for the given image. It is an optional parameter and can
    be set to None or left empty if no tag hint is required, defaults to None (optional)
    :return: a tuple of three values: the predicted tag for the input image, the predicted tag for the
    input image with the input tag as a constraint, and the generated caption for the input image.
    """

    with torch.no_grad():
        caption, tag_predict = model.generate(image,
                                              tag_input=None,
                                              max_length=50,
                                              return_tag_predict=True)

    if input_tag in ['', 'none', 'None']:
        return tag_predict[0], None, caption[0]

    input_tag_list = [input_tag.replace(',', ' | ')]
    with torch.no_grad():
        caption, input_tag = model.generate(image,
                                            tag_input=input_tag_list,
                                            max_length=50,
                                            return_tag_predict=True)

    return tag_predict[0], input_tag[0], caption[0]

# 使用示例：
# python image_tagging.py --cache-path C:/Users/gaoyo/.cache/Tag2Text --image D:/work/images/detect_gender/24.jpeg
# 返回值：
# {
#     "status": 0,
#     "message": "ok",
#     "data": {
#         "model_identified_tags": "woman | bed | girl | people | person | man | couple | head | lay | lie | young",
#         "model_identified_tags_zh": "女人 |床 |女孩 |人物 |人 |男人 |情侣 |头 |莱 |谎言 |年轻",
#         "user_specified_tags": null,
#         "image_caption": "a young man and woman laying in bed with their heads together",
#         "image_caption_zh": "一对年轻男女躺在床上，头并拢"
#     }
# }

def main():
    """
    This function loads a pre-trained image captioning model, processes an input image, and generates a
    caption for the image based on specified and identified tags.
    """
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(), normalize
    ])
    
    # delete some tags that may disturb captioning
    # 127: "quarter"; 2961: "back", 3351: "two"; 3265: "three"; 3338: "four"; 3355: "five"; 3359: "one"
    delete_tag_index = [127,2961, 3351, 3265, 3338, 3355, 3359]

    #######load model
    if os.path.exists(args.cache_path):
        model = torch.load(args.cache_path)
    else:
        model = tag2text_caption(
            pretrained=args.pretrained,
            image_size=args.image_size,
            vit='swin_b',
            delete_tag_index=delete_tag_index
        )
        model.threshold = args.thre  # threshold for tagging
        model.eval()
        torch.save(model, args.cache_path)

    # 移动模型到设备上
    model = model.to(device)

    # 加载图片并进行数据预处理
    raw_image = Image.open(args.image).convert("RGB").resize(
        (args.image_size, args.image_size))
    image = transform(raw_image).unsqueeze(0).to(device)
    res = inference(image, model, args.specified_tags)
    
    # 打开翻译程序执行效率太低，暂时关闭
    # tags_zh = ts.translate_text(res[0], to_language="zh")
    # caption_zh = ts.translate_text(res[2], to_language="zh")
    tags_zh = ""
    caption_zh = ""

    data = {
        "model_identified_tags":  res[0],
        "model_identified_tags_zh": tags_zh,
        "user_specified_tags": res[1],
        "image_caption": res[2],
        "image_caption_zh": caption_zh
    }
    results = {
        "status": 0,
        "message": 'ok',
        "data": data
    }

    print(json.dumps(results, ensure_ascii=False, indent=4))


if __name__ == '__main__':
    main()