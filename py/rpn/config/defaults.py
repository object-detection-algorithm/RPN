from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = 'RPNDetector'
_C.MODEL.DEVICE = "cuda"
# match default boxes to any ground truth with jaccard overlap higher than a threshold (for example: 0.5)
# 正负样本阈值设置。大于正样本阈值的为正样本；小于负样本阈值的为负样本；居于正负样本阈值之间的不参与训练
_C.MODEL.POS_THRESHOLD = 0.7
_C.MODEL.NEG_THRESHOLD = 0.3
_C.MODEL.NUM_CLASSES = 2
# 用于训练的正负样本总数
_C.MODEL.N_CLS = 256
# 正负样本比例
_C.MODEL.POS_NEG_RATIO = 0.5
# 置信度损失和回归损失之间的超参数lambda
_C.MODEL.LAMBDA = 1.0

_C.MODEL.CENTER_VARIANCE = 0.1
_C.MODEL.SIZE_VARIANCE = 0.2

# ---------------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = 'vgg'

# -----------------------------------------------------------------------------
# ANCHORS
# -----------------------------------------------------------------------------
_C.MODEL.ANCHORS = CN()
_C.MODEL.ANCHORS.STRIDE = 16
_C.MODEL.ANCHORS.SIZES = [64, 128, 256, 512]
_C.MODEL.ANCHORS.ASPECT_RATIOS = [1.0, 2.0, 0.5]
_C.MODEL.ANCHORS.CLIP = True
_C.MODEL.ANCHORS.NUM = 12

# -----------------------------------------------------------------------------
# Box Head
# -----------------------------------------------------------------------------
_C.MODEL.BOX_HEAD = CN()
_C.MODEL.BOX_HEAD.NAME = 'RPNBoxHead'
_C.MODEL.BOX_HEAD.PREDICTOR = 'RPNBoxPredictor'
_C.MODEL.BOX_HEAD.FEATURE_MAP = 512
_C.MODEL.BOX_HEAD.CONV_OUTPUT = 512

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Image size
_C.INPUT.IMAGE_SHORT_SIDE = 600
# 如果使用了ImageNet预训练模型，使用以下归一化参数
_C.INPUT.PIXEL_MEAN = (0.485, 0.456, 0.406)
_C.INPUT.PIXEL_STD = (0.229, 0.224, 0.225)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ("voc_2007_trainval", "voc_2012_trainval")  # ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ("voc_2007_test",)  # ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATA_LOADER = CN()
# Number of data loading threads
_C.DATA_LOADER.NUM_WORKERS = 8
_C.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# train configs
_C.SOLVER.MAX_ITER = 100000
_C.SOLVER.LR_STEPS = [60000, 80000]
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.BATCH_SIZE = 1
_C.SOLVER.LR = 1e-3
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 3e-5
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.NMS_THRESHOLD = 0.7
_C.TEST.CONFIDENCE_THRESHOLD = 0.01
_C.TEST.MAX_PER_IMAGE = 300
_C.TEST.BATCH_SIZE = 1

_C.OUTPUT_DIR = 'outputs'
