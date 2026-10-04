import cv2
import numpy as np

class TrafficCounter:
    def __init__(self):
        self.vehicle_count = 0          # Sayım çizgisinden geçen araç sayısı
        self.line_y = 1200              # Sayım çizgisinin varsayılan dikey(Y) koordinatı
        self.tracked_objects = {}       # Takip edilen araçların verisini tutan sözlük
        self.next_object_id = 0         # Tespit edilen araç ID

    # Sayım çizgisini videonun boyutlarına göre otonom olarak belirleyen fonksiyon
    def calculate_line_boundaries(self, locked_mask):
        if locked_mask is None:                                 # Eğer ROI alanı hesaplanmadıysa Y koordinatını 0 yapar ve hatayı engeller
            return 0, 0

        h, w = locked_mask.shape[:2]                            # Hesaplanan ROI alanın height, width değerlerini alır
        self.line_y = int(h * 0.7)                              # Sayım çigisini ROI alanına göre oranlar

        if self.line_y >= h:                                    # Eğer sayım çizgisinin yüksekliği videonun yüksekliğine göre eşitse veya fazlaysa hata oluşumunu engeller
            return 0, w

        # Çizginin roi alanındaki konumunu belirler
        row_pixels = locked_mask[self.line_y, :]
        white_pixels = np.where(row_pixels == 255)[0]

        if len(white_pixels) > 0:                               # Otonom çizginin tam olarak yolun sol şeridi ile sağ şeridi arasında kalmasını sağlayan sınır uçlarını belirler.
            return white_pixels[0], white_pixels[-1]

        return 0, w                                             # Eğer beklenmedik bir durumdan dolayı hata meydana gelirse ekranın kapanmasını engeller

    # Araç takibi yapar
    def update(self, boxes, locked_mask):
        line_start_x, line_end_x = self.calculate_line_boundaries(locked_mask)  # Otonom sayım çizgisinin x ile y eksenleri hesaplanır
        active_tracked_results = []                                             # Aktif olarak takip edilen araç sayısı

        # VehicleDetector da tespit edilen konturları gezer
        for box in boxes:
            xmin, ymin, xmax, ymax = box                        # Konturların koordinat değerleri hesaplanır
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2

            if not (line_start_x <= x_center <= line_end_x):    # Eğer kontur çizgiden geniş değilse devam eder
                continue

            # Basit ve kararlı takip / sayım yapısı
            matched = False
            for obj_id, data in self.tracked_objects.items():   # Tespit edilen araçların id ve data değerlerini döndürür.

                # Öklid formülünü, bir karede tespit ettiğimiz aracın merkez konumu ile bir önceki karede hafızaya kaydettiğimiz aracın konumu
                # arasındaki kuş uçuşu piksel mesafesini en doğru şekilde hesaplayıp aynı araç olduklarını eşleştirmek için kullandık.
                dist = np.sqrt((x_center - data["x"])**2 + (y_center - data["y"])**2)

                # Hesaplanan mesafe 100 pikselden az ise aynı araç olarak tespit eder ve konum verilerini günceller, aynı araç tekrardan sayılmaz
                if dist < 100:
                    matched = True
                    # Konumu güncelle
                    self.tracked_objects[obj_id]["x"] = x_center
                    self.tracked_objects[obj_id]["y"] = y_center
                    active_tracked_results.append((xmin, ymin, xmax, ymax, obj_id, data["is_counted"]))
                    break

            if not matched:                                     # Eğer eşleşme gerçekleşmediyse araç çizgiden geçince sayılır
                is_counted = False
                if y_center >= self.line_y:
                    is_counted = True
                    self.vehicle_count += 1

                self.tracked_objects[self.next_object_id] = {   # Yeni id ile sisteme kayıt edilir
                    "x": x_center,
                    "y": y_center,
                    "is_counted": is_counted
                }

                # Aktif takip edilen araç listesine eklenir ve bilgileri güncellenir
                active_tracked_results.append((xmin, ymin, xmax, ymax, self.next_object_id, is_counted))
                self.next_object_id += 1

        return active_tracked_results       # Aktif takip sonuçlarını döndürür
