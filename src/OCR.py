import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gc  # 引入垃圾回收模块
import detectron2
from pdf2image import convert_from_path
import layoutparser as lp
from PIL import ImageFilter, ImageOps
import shutil


def user_input_interactive():
    """
    与用户交互，获取PDF文件路径、起始页码和结束页码，并将PDF文件上传到 ~/营销故事生成/data/ 目录下。
    返回：
        pdf_output_path (str): 上传后的PDF文件路径
        start_page (int): 用户输入的起始页码
        end_page (int): 用户输入的结束页码
    """
    # 让用户输入本地文件路径
    pdf_input_path = input("请输入要处理的PDF文件的本地路径：")

    # 验证文件是否存在
    if not os.path.exists(pdf_input_path):
        print("文件不存在，请检查路径！")
        return None, None, None

    # 创建 ~/营销故事生成/data/ 目录
    target_dir = os.path.expanduser("~/营销故事生成/data/")
    os.makedirs(target_dir, exist_ok=True)

    # 将文件移动到 ~/营销故事生成/data/ 下
    pdf_file_name = os.path.basename(pdf_input_path)
    pdf_output_path = os.path.join(target_dir, pdf_file_name)
    shutil.copy(pdf_input_path, pdf_output_path)

    print(f"文件已上传并保存至 {pdf_output_path}")

    # 让用户输入起始和结束页码
    start_page = int(input("请输入处理的起始页码："))
    end_page = int(input("请输入处理的结束页码："))

    return pdf_output_path, start_page, end_page


def process_image(img):
    """
    对图像进行预处理，如去除噪声、增强对比度等。
    参数:
        img (PIL.Image): 输入的图像。
    返回:
        img (PIL.Image): 处理后的图像。
    """
    # 转换为灰度图像
    img = img.convert("L")
    # 去除噪声
    img = img.filter(ImageFilter.MedianFilter(size=3))
    # 增强对比度
    img = ImageOps.autocontrast(img)
    return img


def extract_pdf_pages_as_images(pdf_path, start_page, end_page, dpi=1500):
    """
    提取 PDF 的部分页码并将其转换为图像。

    参数:
        pdf_path (str): PDF 文件路径
        start_page (int): 起始页码（从 1 开始）
        end_page (int): 结束页码（包含结束页）
        dpi (int): 图像的分辨率，默认 1500

    返回:
        images (list of PIL.Image): 提取的图像列表
    """
    # Convert specified range of pages to images
    images = convert_from_path(pdf_path, first_page=start_page, last_page=end_page, dpi=dpi)

    processed_images = []
    for i, img in enumerate(images):
        # 处理图像
        img = process_image(img)
        # **Convert to RGB before saving**
        img = img.convert('RGB')  # Add this line

        # 保存处理后的图像
        '''output_path = os.path.join(output_image_dir, f"page_{start_page + i}.png")
        img.save(output_path)'''
        processed_images.append(img)
        '''print(f"Saved: {output_path}")'''

    return processed_images


#  load model
model = lp.Detectron2LayoutModel(
    #  "lp://detectron2/PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x",
    config_path=os.path.expanduser("~/营销故事生成/data/config.yaml"),
    extra_config=["MODEL.WEIGHTS", os.path.expanduser("~/营销故事生成/data/model_final.pth"),  # 本地模型权重文件路
                  "MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.15,
                  "MODEL.ROI_HEADS.NMS_THRESH_TEST", 0.1,
                  "MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION", 7],
    label_map={0: "Text", 1: "Table", 2: "Figure"}
)
#  label_map={0:"Text", 1:"Title", 2:"List", 3:"Table", 4:"Figure"}
ocr_model = lp.TesseractAgent(languages='chi_sim')


def module_detect(images):
    img = np.asarray(images[0])
    detected = model.detect(img)
    # Convert detected elements to LayoutElements if they are strings
    detected = lp.Layout([lp.LayoutElement(block) if isinstance(block, str) else block for block in detected])
    lp.draw_box(img, detected, box_width=5, box_alpha=0.2, show_element_type=True)
    return detected, img


#  sort and collision detection
# 按第一个条件升序排序，第二个条件降序排序
def sort_collison_detecion(detected):
    new_detected = sorted(detected, key=lambda x: (x.coordinates[1], -(x.coordinates[1] - x.coordinates[3])))
    # print(new_detected[0].coordinates
    # assign ids
    i = 0
    while i < len(new_detected) and new_detected[i].type != 'Text':
        i += 1
    if i < len(new_detected):
        x_left, y_left, x_right, y_right = new_detected[i].coordinates[0], new_detected[i].coordinates[i], \
            new_detected[i].coordinates[2], \
            new_detected[i].coordinates[3]  # maintain the margin of module
        new_detected_tmp = []
        for idx, block in enumerate(new_detected):
            if block.type == 'Text':
                # 对于当前模块完全从属于之前模块，跳过
                if block.coordinates[1] > y_left and block.coordinates[3] < y_right:
                    continue
                # 识别并删除脚注
                elif new_detected_tmp and block.coordinates[1] > 9000 and\
                        block.coordinates[1] - new_detected_tmp[-1].coordinates[3] > 300:
                    continue
                else:
                    new_detected_tmp.append(block)
                    x_left, y_left, x_right, y_right = block.coordinates[0], block.coordinates[1], \
                        block.coordinates[2], block.coordinates[3]
        detected = lp.Layout([block.set(id=idx) for idx, block in enumerate(new_detected_tmp)])  # check
        for block in detected:
            print("---", str(block.id)+":", block.type, "---")
            print(block, end='\n\n')
    else:
        print("Not exist text! Please input the next page!")
    return detected


def parse_doc(dic):
    for k, v in dic.items():
        if "Title" in k:
            print('\x1b[1;31m' + v + '\x1b[0m')
        elif "Figure" in k:
            plt.figure(figsize=(10, 5))
            plt.imshow(v)
            plt.show()
        else:
            print(v)
        print(" ")


def process_pdf_images(images, detected, output_csv_path):
    """
    处理PDF的每一页，提取文本内容并将其保存到指定的CSV文件中。
    参数:
    images (list): PDF每一页的图像列表。
    detected (list): 检测到的文本块列表。
    output_csv_path (str): 输出的CSV文件路径。
    """

    all_text = ""
    dic_predicted = {}
    previous_text = ""  # 用于保存前一个块的文本

    for block in [block for block in detected if block.type in ["Text"]]:
        # 分割图像
        segmented = block.pad(left=1000, right=1000, top=70, bottom=70).crop_image(images)

        # 文本提取
        extracted = ocr_model.detect(segmented).replace('\n', ' ').strip()

        # 处理文本，遇到句号换行
        if previous_text and not previous_text.endswith('。') and not previous_text.endswith('. '):
            # 如果上一个模块不以句号结尾，则将当前模块的内容添加到上一个模块
            previous_text += extracted
        else:
            if previous_text:
                # 将上一个模块保存到字典
                dic_predicted[str(block.id - 1) + "-Text"] = previous_text.strip()

            # 开始新的模块
            previous_text = extracted

        # 如果遇到句号，则换行
        previous_text = previous_text.replace('。', '。\n')

    # 保存最后一个模块
    if previous_text:
        dic_predicted[str(block.id) + "-Text"] = previous_text.strip()

    # 移除多余的换行符
    for key, value in dic_predicted.items():
        # 将每行的文本拆分，再通过 join 合并，确保每行之间只有一个换行符
        dic_predicted[key] = "\n".join(line.strip() for line in value.splitlines() if line.strip())

    # 拼接当前页的文本内容到总文本中
    all_text += "".join(dic_predicted.values()) + "\n"

    # 写入CSV文件，并确保第一次创建时添加列标题
    file_exists = False
    try:
        with open(output_csv_path, "r", encoding="utf-8"):
            file_exists = True
    except FileNotFoundError:
        # 文件不存在则跳过检查
        file_exists = False

    with open(output_csv_path, "a", encoding="utf-8", newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            # 文件不存在，写入列标题
            writer.writerow(["Text"])

        # 写入文本内容
        if all_text.strip():
            writer.writerow([all_text.strip()])

    # 释放当前页相关的内存
    del detected
    del dic_predicted
    gc.collect()

    return


def ocr(pdf_file_path, start_page, end_page, output_csv_path):
    page_number = start_page
    images = extract_pdf_pages_as_images(pdf_file_path, start_page, end_page)
    print(f'Image {page_number} has been extracted!')
    detected, img = module_detect(images)
    print(f'Image {page_number} has been detected!')
    detected = sort_collison_detecion(detected)
    print(f'Sorting and collision detection for image {page_number} have been completed!')
    process_pdf_images(img, detected, output_csv_path)
    print(f'PDF page {page_number} has been processed!')
    del images
    del detected
    gc.collect()


def main():
    """
    主函数：处理用户交互，获取PDF文件路径、页码范围并执行OCR处理。
    """
    # 获取用户交互输入
    pdf_file_path, start_page, end_page = user_input_interactive()

    # 验证用户输入是否有效
    if pdf_file_path and start_page and end_page:
        # 定义输出图像目录和文本输出文件
        #  output_image_dir = os.path.expanduser("~/营销故事生成/data/output_images")
        #  os.makedirs(output_image_dir, exist_ok=True)
        output_csv_path = os.path.expanduser("~/营销故事生成/data/output.csv")

        # 执行OCR处理
        for i in range(end_page - start_page + 1):
            ocr(pdf_file_path, start_page + i, start_page + i, output_csv_path)

        print(f"所有页处理完毕，结果已保存至 {output_csv_path}")
    else:
        print("输入无效，程序终止。")


if __name__ == "__main__":
    main()
