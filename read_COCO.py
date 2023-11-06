import torchvision
from PIL import ImageDraw

# 导入coco 2017 验证集和对应annotations
coco_dataset = torchvision.datasets.CocoDetection(root="data/coco/train2017/",
                                                  annFile="data/coco/annotations/instances_train2017.json")

coco_dataset1 = torchvision.datasets.CocoDetection(root="data/coco/val2017/",
                                                  annFile="data/coco/annotations/instances_val2017.json")
# 图像和annotation分开读取
image, info = coco_dataset[11]
# ImageDraw 画图工具
image_handler = ImageDraw.ImageDraw(image)

for annotation in info:
    # bbox为检测框的位置坐标
    x_min, y_min, width, height = annotation['bbox']
    # ((), ())分别为左上角的坐标对和右上角的坐标对，image_handler.rectangle是指在图片是绘制方框
    image_handler.rectangle(((x_min, y_min), (x_min + width, y_min + height)))

image.show()

