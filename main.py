import cv2
import numpy as np
import config
from core.Preprocessor import Preprocessor
from tracking.ROIManager import ROIManager
from tracking.VehicleDetector import VehicleDetector
from tracking.TrafficCounter import TrafficCounter


FRAME_SIZE = (800, 450)  # Genişletilmiş video akış boyutu (Daha net görünüm)
PANEL_HEIGHT = 140       # Genişletilmiş modern alt bilgi paneli yüksekliği


def main():
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    preprocessor = Preprocessor()
    roi_manager = ROIManager()
    vehicle_detector = VehicleDetector()
    traffic_counter = TrafficCounter()

    print("Sistem başlatıldı. Genişletilmiş profesyonel dashboard modunda çalışıyor...")

    window_name = "Traffic Tracker - Professional CV Dashboard"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Genişlik: 800 + 800 = 1600, Yükseklik: 450 + 140 = 590
    cv2.resizeWindow(window_name, 1600, 590)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)

        # 1. Preprocessor'dan hareket maskesini al
        raw_mask, _ = preprocessor.process_frame(frame)

        # 2. RoiManager ile ROI kilitleme
        processed_mask, is_calibrated = roi_manager.lock_mask(raw_mask)
        processed_mask_resized = cv2.resize(processed_mask, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        processed_mask_bgr = cv2.cvtColor(processed_mask_resized, cv2.COLOR_GRAY2BGR)

        display_frame = frame_resized.copy()
        active_objects = []

        if not is_calibrated:
            status_text = "CALIBRATING ROI... (Learning Traffic Flow)"
            status_color = (0, 165, 255)  # Turuncu
        else:
            status_text = "SYSTEM ACTIVE - MONITORING & TRACKING"
            status_color = (0, 255, 0)  # Yeşil

            # 3. VehicleDetector ile araç kutularını bul
            boxes = vehicle_detector.detect_contour(processed_mask)

            # Koordinatları yeni ekran boyutuna ölçekle
            h_orig, w_orig = frame.shape[:2]
            scale_x = FRAME_SIZE[0] / w_orig
            scale_y = FRAME_SIZE[1] / h_orig

            # 4. TrafficCounter güncellemesi ve ID'li araçların alınması
            active_objects = traffic_counter.update(boxes, roi_manager.locked_mask)

            # Otonom sayım çizgisini çizdir
            line_start_x, line_end_x = traffic_counter.calculate_line_boundaries(roi_manager.locked_mask)
            if line_start_x > 0 and line_end_x > 0:
                s_line_y = int(traffic_counter.line_y * scale_y)
                s_start_x = int(line_start_x * scale_x)
                s_end_x = int(line_end_x * scale_x)
                cv2.line(display_frame, (s_start_x, s_line_y), (s_end_x, s_line_y), (0, 255, 255), 3)

            # Araç kutularını, ID'lerini ve durumlarını çiz
            for (xmin, ymin, xmax, ymax, obj_id, is_counted) in active_objects:
                sx_min = int(xmin * scale_x)
                sy_min = int(ymin * scale_y)
                sx_max = int(xmax * scale_x)
                sy_max = int(ymax * scale_y)

                box_color = (0, 255, 0) if is_counted else (0, 165, 255)
                cv2.rectangle(display_frame, (sx_min, sy_min), (sx_max, sy_max), box_color, 2)
                cv2.putText(display_frame, f"ID:{obj_id}", (sx_min, max(20, sy_min - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # --- YAN YANA GÖRÜNTÜ BİRLEŞTİRME ---
        top_row = np.hstack([display_frame, processed_mask_bgr])

        # --- ŞIK VE GENİŞLETİLMİŞ ALT BİLGİ PANELİ (DASHBOARD) ---
        dashboard = np.zeros((PANEL_HEIGHT, top_row.shape[1], 3), dtype=np.uint8)
        dashboard[:] = (25, 25, 25)  # Profesyonel koyu tema

        # Sol taraf: Durum ve Toplam Sayım
        cv2.putText(dashboard, f"STATUS: {status_text}", (30, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(dashboard, f"TOTAL VEHICLES COUNTED: {traffic_counter.vehicle_count}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Sağ taraf: Aktif Takip Edilen Araç Bilgileri (Daha düzenli ve detaylı)
        cv2.putText(dashboard, "ACTIVE TRACKED VEHICLES:", (850, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

        if active_objects:
            # En fazla 6 aktif aracı detaylı listele
            for idx, obj in enumerate(active_objects[:6]):
                obj_id = obj[4]
                cx = int((obj[0] + obj[2]) / 2)
                cy = int((obj[1] + obj[3]) / 2)
                status_str = "COUNTED" if obj[5] else "PENDING"
                color_str = (0, 255, 0) if obj[5] else (0, 165, 255)

                info_text = f"ID #{obj_id} | Pos:({cx},{cy}) | {status_str}"

                # İki sütun halinde yazdır
                col = idx % 2
                row = idx // 2
                pos_x = 850 + (col * 400)
                pos_y = 70 + (row * 30)

                cv2.putText(dashboard, info_text, (pos_x, pos_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_str, 1)
        else:
            cv2.putText(dashboard, "No vehicles currently active in ROI.", (850, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        full_dashboard = np.vstack([top_row, dashboard])

        cv2.imshow(window_name, full_dashboard)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
