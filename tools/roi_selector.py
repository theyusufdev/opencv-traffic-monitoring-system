import cv2
import numpy as np

# Tıklanan koordinatları tutacak liste
points = []


def mouse_callback(event, x, y, flags, param):
    global points
    # Sol fare tıkı: Yeni nokta ekler
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"[Nokta {len(points)}] Eklendi: ({x}, {y})")


def run_roi_selector(video_path):
    global points
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[HATA] Video okunamadı! Lütfen dosya yolunu kontrol edin.")
        return

    h, w = frame.shape[:2]
    window_name = "ROI Secici - Noktalari Tiklayin (Temizle: 'c' | Onayla: 'ENTER')"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 4K ekranlara sığacak boyutta açılır
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n--- ROI SEÇİM TALİMATLARI ---")
    print("1. Yolun etrafına saat yönünde tıklayarak istediğiniz sayıda köşe ekleyin (6, 7 vb.).")
    print("2. Yanlış tıklarsanız 'c' tuşuna basarak sıfırlayabilirsiniz.")
    print("3. Seçimi bitirmek için 'ENTER' veya 'SPACE' tuşuna basın.\n")

    while True:
        display_frame = frame.copy()

        # Tıklanan noktaları kırmızı daire olarak çiz
        for pt in points:
            cv2.circle(display_frame, pt, 6, (0, 0, 255), -1)

        # Noktalar arası yeşil çizgileri çiz
        if len(points) > 1:
            pts_array = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_frame, [pts_array], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        # 'c' tuşu: Noktaları temizler
        if key == ord('c'):
            points = []
            print("[BİLGİ] Tüm noktalar temizlendi.")

        # ENTER veya SPACE tuşu: Seçimi onaylar ve çıkar
        elif key in [13, 32]:
            if len(points) >= 3:
                break
            else:
                print("[UYARI] En az 3 nokta seçmelisiniz!")

    cv2.destroyAllWindows()

    # Koordinatları 4K / farklı çözünürlük uyumu için oranlara (0.0 - 1.0) çevir
    normalized_points = [(round(x / w, 4), round(y / h, 4)) for x, y in points]

    print("\n" + "=" * 50)
    print("SEÇİLEN ORANSAL ROI KOORDİNATLARI (config.py için):")
    print("=" * 50)
    print(f"ROI_POLYGON_NORMALIZED = {normalized_points}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    # Test etmek istediğiniz videonun yolunu girin
    VIDEO_PATH = "../data/traffic_video.mp4"
    run_roi_selector(VIDEO_PATH)