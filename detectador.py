import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import torchvision,torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from detectron2.data.datasets import register_coco_instances
register_coco_instances("prueba1", {}, "./data/trainval.json", "./data/images")
register_coco_instances("prueba1v", {}, "./data/trainval.json", "./data/images")

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog,DatasetCatalog
prueba1_metadata = MetadataCatalog.get("prueba1")
dataset_dicts = DatasetCatalog.get("prueba1")

import random
from detectron2.utils.visualizer import Visualizer

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=prueba1_metadata, scale=0.2)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow("Nice",vis.get_image()[:, :, ::-1])
    
    from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

cfg = get_cfg()#se crea la clase que almacenara todo
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))#se importa la arquitectura de la red
cfg.DATASETS.TRAIN = ("prueba1",)#se establece la base de datos de entrenamiento
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 0#numero de nucleos de comunicacion con la cpu
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  #se inicializan los pesos de la red
cfg.SOLVER.IMS_PER_BATCH = 2#se establece el tamaño del lote de entrenamiento
cfg.SOLVER.BASE_LR = 0.01#factor de aprendizaje
cfg.SOLVER.MAX_ITER = (#numero de iteraciones
    300
)  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128
)  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # numero de clases

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()#empieza el entrenamiento
    
from detectron2.engine import DefaultPredictor
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # umbral minimo para que si se detecte el objeto como verdadero
cfg.DATASETS.TEST = ("prueba1", )
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode

for d in random.sample(dataset_dicts,1):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=prueba1_metadata, 
                   scale=1, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Nice",v.get_image()[:, :, ::-1])
    
    im = cv2.imread("./img1.jpeg")
cv2.imshow("Nice",im)
im = cv2.resize(im,(224,224))
cv2.imshow("test",im)

import time
t = time.time()
outputs = predictor(im)
elapsed = time.time() - t
print(elapsed)
outputs["instances"].pred_classes
outputs["instances"].pred_boxes

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.1)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Nice",v.get_image()[:, :, ::-1])