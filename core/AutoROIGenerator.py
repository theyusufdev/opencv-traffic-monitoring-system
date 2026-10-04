import cv2
import numpy as np
import config

class AutoROIGenerator:
    def __init__(self):
        self.accumulate = None          # Trafik akışının olduğu alanı belirlemek için kalibrasyon bitene kadar video karelerin toplandığı değişken
        self.is_calibrated = False      # Kalibrasyonun tamamlanmadığını bildirir
        self.final_roi_mask = None      # Algoritma sonucu olarak belirlenecek ROI alanı
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, config.HEATMAP_KERNEL_CLOSE)
        self.frame_count = 0            # Kare sayısı

    # Trafik akışının belirlenmesi için MOG2 ile algılanan değişimin olduğu pikselleri kalibrasyon süreci boyunca toplar
    def update(self, final_mask):
        if self.is_calibrated:
            return

        if self.accumulate is None:
            self.accumulate = np.zeros(final_mask.shape, np.float32)

        # Siyah tuval üzerine her hareketli pikseli kalibrasyon kare sayısı bitene kadar toplayarak pikselleri biriktirir
        # Logaritmik toplama ile aşırı beyazlamaları dengelemeye yarar
        self.accumulate += np.log1p(final_mask)
        self.frame_count += 1   # Kalibrasyon yapıldıkça bir sonraki kareye geçer

        # Kalibrasyon sınırına ulaşınca _finalize_calibration() fonksiyonunu çalıştır
        if self.frame_count >= config.CALIBRATION_FRAMES:
            self._finalize_calibration()

    # Kalibrasyon sonundaki oluşan ROI alanını hesaplar
    def _finalize_calibration(self):
        # Piksellerdeki değerlerini içeren matrisi 0 ile 255 arasına sıkıştırarak hareketin olduğu pikselleri beyaz renge boyar
        normalized = cv2.normalize(self.accumulate, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, binary_thresh = cv2.threshold(normalized, config.HEATMAP_THRESHOLD, 255, cv2.THRESH_BINARY)

        morph_closed = cv2.morphologyEx(binary_thresh, cv2.MORPH_CLOSE, self.kernel_close)                  # Morfolojik kapanma ile küçük gürültüler giderilir
        contours, hierarchy = cv2.findContours(morph_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    # Harketin olduğu alanların konturunu belirler
        if contours:
            sorted_contours = max(contours, key=cv2.contourArea)                                            # Belirlenen en büyük konturu seçer
            hull = cv2.convexHull(sorted_contours, returnPoints=True)                                       # Hareketi algılanan nesnenin dış sınırları çizilir

            self.final_roi_mask = np.zeros(self.accumulate.shape, np.uint8)
            cv2.fillPoly(self.final_roi_mask, [hull], 255)                                        # Hareketleri siyah tuvala ekler

        self.is_calibrated = True                                                                           # Kalibrenin tamamlandığını bildirir