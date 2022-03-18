from models import *

def return_fasterrcnn_resnet50(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_resnet50.create_model(
        num_classes, pretrained=pretrained, coco_model=coco_model
    )
    return model

def return_fasterrcnn_mobilenetv3_large_fpn(num_classes):
    model = fasterrcnn_mobilenetv3_large_fpn.create_model(num_classes)
    return model

create_model = {
    'fasterrcnn_resnet50': return_fasterrcnn_resnet50,
    'fasterrcnn_mobilenetv3_large_fpn': return_fasterrcnn_mobilenetv3_large_fpn,
}