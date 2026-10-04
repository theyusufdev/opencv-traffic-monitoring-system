# config.py
import cv2

# Video Kaynağı
VIDEO_PATH = "data/traffic_video.mp4"
VIDEO_SIZE = (3840,2160)

# Ön İşleme (PreProcessing)
GAUSSIAN_KERNEL = (5, 5)
MOG2_HISTORY = 500
MOG2_VAR_THRESHOLD = 50
MOG2_DETECT_SHADOWS = True
MIN_CONTOUR_AREA = 6000  # 4K için piksel alt sınırı artırıldı
MORPH_KERNEL_OPEN = (3, 3)
MORPH_KERNEL_CLOSE = (51, 51)  # 4K'da araç parçalarını birleştirmek için büyütüldü

# ROI
AUTO_ROI_ENABLED = False
ROI_HANDLE_VALUES = [(0.4552, 0.3995), (0.2609, 0.9495), (0.7339, 0.9519), (0.651, 0.7236), (0.7227, 0.6181), (0.6073, 0.463), (0.5474, 0.4236), (0.5352, 0.4037)]
CALIBRATION_FRAMES = 150
HEATMAP_THRESHOLD = 30
HEATMAP_KERNEL_CLOSE = (25,25)

# Geometrik Sınıflandırma
CAR_BUS_AREA_THRESHOLD = 22.0  # m^2 eşiği
ASPECT_RATIO_THRESHOLD = 2.8

# Takip Motoru (Tracking)
MAX_MISSED_FRAMES = 15
IOU_THRESHOLD = 0.3