import cv2
import numpy as np
import config
from core.AutoROIGenerator import AutoROIGenerator

class ROIManager:
    def __init__(self):

        # ROI alan hesabı yöntemi (True: Otonom - False: Manuel)
        self.auto_roi_enabled = config.AUTO_ROI_ENABLED
        if self.auto_roi_enabled:
            self.auto_roi_generator = AutoROIGenerator()
        else:
            self.auto_roi_generator = None
        
        self.locked_mask = None     # Kilitlenecek ROI alanı

    # ROI nin manuel girdiğimiz koordinatlarını çekmemize yarar
    def create_manual_roi(self, frame_shape):
        h, w = frame_shape[:2]                              # Videonun boyutlarını alır
        mask = np.zeros((h, w), dtype=np.uint8)      # Belirlenen boyutlarda arka plan için siyah tuval oluşturur

        # config dosyasındaki manuel belirlenen 0-1 arasındaki normalize edilmiş ROI koordinatlarını videonun o anki gerçek genişlik ve yükselikle çarpar
        pixel_points = []
        for x_norm, y_norm in config.ROI_HANDLE_VALUES:
            pixel_x = int(x_norm * w)
            pixel_y = int(y_norm * h)
            pixel_points.append([pixel_x, pixel_y])

        # Piksel noktalarını numpy array ine dönüştürür ve koordinatları gerçek video üzerinde tespit eder
        pts = np.array(pixel_points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

        return mask     # Tespit edilen ROI alanını çalıştırır

    # Kalibrasyon veya manuel ROI ile maskeyi kilitler
    def lock_mask(self, fg_mask):
        if self.locked_mask is None or self.locked_mask.shape != fg_mask.shape:     # ROI alanı hesaplanmadıysa veya ROI alanının boyutu, gerçek video boyutlarıyla uyuşmuyorsa
            if self.auto_roi_enabled:                                               # Seçili ROI hesaplama yöntemi otonom ise
                if not self.auto_roi_generator.is_calibrated:                       # Ve kalibrasyon tamamlanmadıysa
                    self.auto_roi_generator.update(fg_mask)                         # Kalibrasyonu devam ettir
                    return fg_mask, False
                else:
                    self.locked_mask = self.auto_roi_generator.final_roi_mask       # Kalibrasyon tamamlandıysa hesaplanan ROI alanını kilitle
            else:
                # Manuel ROI aktifse kalibrasyon beklemeden anında kilitle
                self.locked_mask = self.create_manual_roi(fg_mask.shape)

        if self.locked_mask is not None:                                            # Eğer kilitli bir ROI alanı mevcut ise bunu siyah tuval üzerine ekle
            fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=self.locked_mask)

        return fg_mask, True                                                        # Sonucu döndür

