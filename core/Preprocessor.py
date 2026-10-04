import cv2
import numpy as np
import config
from core.AutoROIGenerator import AutoROIGenerator

class Preprocessor:
    def __init__(self):

        # Bu fonksiyon ile MOG2 algoritmasını kullanarak videodaki arka planı belirleyip araç tespit başarımını
        # arttırmak adına arkaplanı videodan çıkartırız.
        # MOG2 algortiması: Zaman çizelgesi içerisinde pikselleri takip eder ve değişimin algılanmadığı
        # pikselin rengini siyah, değişim algılandığı pikseli ise beyaz yapar. Bu şekilde hareket eden nesneler
        # beyaz renkli gözükürken arka plan siyah renkli gözükür. Bu sayede tespit algortimalarımızı hareketli nesneler
        # üzerinde yoğunlaştırabiliriz. Parametrelerini ise config.py dosyasından almaktadır.
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.MOG2_HISTORY,                # Bir piksel değişimini incelerken son 500 karedeki durumuna bakıp karar verir
            varThreshold=config.MOG2_VAR_THRESHOLD,     # Hareket düzeyinindeki değişim hassasiyet
            detectShadows=config.MOG2_DETECT_SHADOWS    # Hareket eden ana nesne ile gölgeleri biribirinden ayırır.
        )

        # Morfolojik open ve close işlemleri için kernel ayarları
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPH_KERNEL_OPEN)
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPH_KERNEL_CLOSE)

        # Karedeki harektin algılaması için incelenmesi gereken alanı belirtir
        self.roi_mask = None
        self.auto_roi_enabled = config.AUTO_ROI_ENABLED

        if self.auto_roi_enabled:
            self.auto_roi_generator = AutoROIGenerator()


    def create_manuel_roi(self, frame_shape):
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Normalize (0.0 - 1.0) koordinatları gerçek piksel (X, Y) değerleriyle çarpıyoruz
        pixel_points = []
        for x_norm, y_norm in config.ROI_HANDLE_VALUES:
            pixel_x = int(x_norm * w)  # Genişlik (Width) ile çarpım -> X koordinatı
            pixel_y = int(y_norm * h)  # Yükseklik (Height) ile çarpım -> Y koordinatı
            pixel_points.append([pixel_x, pixel_y])

        # OpenCV fillPoly için int32 formatına dönüştürüyoruz
        pts = np.array(pixel_points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

        return mask

    # Videoda her bir kare ilerledikçe bu fonksiyon çalışır.
    def process_frame(self, frame):
        if frame is None or frame.size == 0:
            raise ValueError("Geçersiz veya boş video karesi alındı.")

        # Contrast dengeleyici oluştur
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Video karesinin 3 kanallı olan BGR(Blue-Green-Red) renk kodlarını 1 kanallı olan griye çevirir
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Gaussian: Bir pikselin yeni değerini; merkezdeki piksele en yüksek, uzaklaşan komşu piksellere ise giderek azalan ağırlıklar vererek (ağırlıklı ortalama ile) yeniden hesaplar.
        gauss_blur = cv2.GaussianBlur(gray_frame, config.GAUSSIAN_KERNEL, 0)
        fg_mask = self.bg_subtractor.apply(gauss_blur)     # MOG2 algortimasını videoya uygular

        # MOG2 ile tespit edilen gölgeleri dışarı atar.
        _, binary_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

        # Morfolojik Open-Close işlemleri ile küçük boyutlu gölgeleri yok eder
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, self.kernel_open)
        final_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, self.kernel_close)


        if self.auto_roi_enabled:                                                      # Isı Haritası ile otonom roi alanını hesaplar
            if not self.auto_roi_generator.is_calibrated:                              # Seçilen yöntemde kalibrasyonun tamamlanmadığını kontrol eder
                self.auto_roi_generator.update(final_mask)                             # Kalibrasyon sırasında pikselleri günceller
                return final_mask, False                                               # Son pikseli çalıştır

            elif self.roi_mask is None:
                self.roi_mask = self.auto_roi_generator.final_roi_mask                 # ROI alanı belirlendiyse bu değer roi_mask değişkenine atar

        if self.roi_mask is None or self.roi_mask.shape != final_mask.shape:           # ROI alanı belirlenmediyse veya gelen alanın boyutu fina_mask değerinin boyutu ile eşit değilse
            self.roi_mask = self.create_manuel_roi(final_mask.shape)                   # Manuel hesaplanan ROI alan değerlerini kullanır

        if self.roi_mask is not None:
            final_mask = cv2.bitwise_and(final_mask, final_mask, mask=self.roi_mask)   # ROI hesaplandıyda siyah tuval üzerine algılanan hareketleri entegre eder

        return final_mask, True                                                        # Sonucu döndürür
