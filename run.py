import time
import cv2
from ultralytics import YOLO

# ---------- Settings ----------
CAM_INDEX = 0            # your webcam index
IMG_SIZE = 640           # YOLO input size
CONF = 0.25              # confidence threshold
DEVICE = None            # e.g. "cuda:0" or "cpu" (None = auto)
WINDOW = "YOLOv8n vs YOLOv11n | [Q]=quit  [C]=capture+eval  [L]=live toggle  [S]=save last capture"
# ------------------------------

def format_dets(results, topk=10):
    """Return a short text summary of detections: class conf (x1 y1 x2 y2)."""
    names = results.names
    lines = []
    if getattr(results, "boxes", None) is None or results.boxes is None:
        return "No boxes"
    boxes = results.boxes
    for i in range(min(len(boxes), topk)):
        cls = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        xyxy = boxes.xyxy[i].tolist()
        lines.append(f"{names[cls]} {conf:.2f} {tuple(round(v,1) for v in xyxy)}")
    return "\n".join(lines) if lines else "No boxes"

def infer_one(model, frame, imgsz=640, conf=0.25, device=None):
    t0 = time.time()
    res = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    dt = (time.time() - t0)
    return res, dt  # seconds

def put_text(img, text, org):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

def stack_side_by_side(a, b, width_each=640, height=480):
    a = cv2.resize(a, (width_each, height))
    b = cv2.resize(b, (width_each, height))
    return cv2.hconcat([a, b])

def main():
    # Load models (auto-download if needed)
    model_v8 = YOLO("yolov8n.pt")
    model_v11 = YOLO("yolo11n.pt")

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    live_mode = True         # True = run both per-frame; False = show raw preview until capture
    last_capture = None      # last still image captured
    last_annot_pair = None   # (img_v8, img_v11) for save
    print("[Q]=quit  [C]=capture+eval  [L]=toggle live/preview  [S]=save last capture")

    fps_hist_v8, fps_hist_v11 = [], []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        display = None

        if live_mode:
            # Run both models live on this frame
            res8, dt8 = infer_one(model_v8, frame, imgsz=IMG_SIZE, conf=CONF, device=DEVICE)
            res11, dt11 = infer_one(model_v11, frame, imgsz=IMG_SIZE, conf=CONF, device=DEVICE)

            ann8 = res8.plot()   # annotated image (BGR)
            ann11 = res11.plot()

            # Instant FPS
            fps8 = 1.0 / dt8 if dt8 > 0 else 0.0
            fps11 = 1.0 / dt11 if dt11 > 0 else 0.0
            fps_hist_v8 = (fps_hist_v8[-29:] + [fps8])
            fps_hist_v11 = (fps_hist_v11[-29:] + [fps11])

            # Compose side-by-side
            display = stack_side_by_side(ann8, ann11, 640, 480)
            put_text(display, f"YOLOv8n  {1000*dt8:.1f} ms  ({fps8:.1f} FPS)", (20, 30))
            put_text(display, f"YOLOv11n {1000*dt11:.1f} ms  ({fps11:.1f} FPS)", (680, 30))
            put_text(display, "[L]=live OFF  [C]=capture  [S]=save  [Q]=quit", (20, 470))

        else:
            # Preview only (no inference) until user captures
            display = cv2.resize(frame, (1280, 480))
            put_text(display, "Preview (no inference)  Press [C] to capture+eval", (20, 30))
            put_text(display, "[L]=live ON  [Q]=quit", (20, 470))

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('l'):
            live_mode = not live_mode

        elif key == ord('c'):
            # Capture current frame and evaluate both models once
            last_capture = frame.copy()
            print("\n================ CAPTURED FRAME ================")
            res8, dt8 = infer_one(model_v8, last_capture, imgsz=IMG_SIZE, conf=CONF, device=DEVICE)
            res11, dt11 = infer_one(model_v11, last_capture, imgsz=IMG_SIZE, conf=CONF, device=DEVICE)

            print(f"[YOLOv8n]  {1000*dt8:.1f} ms")
            print(format_dets(res8))

            print(f"\n[YOLOv11n] {1000*dt11:.1f} ms")
            print(format_dets(res11))

            ann8 = res8.plot()
            ann11 = res11.plot()
            last_annot_pair = (ann8.copy(), ann11.copy())

            comp = stack_side_by_side(ann8, ann11, 640, 480)
            put_text(comp, f"YOLOv8n {1000*dt8:.1f} ms", (20, 30))
            put_text(comp, f"YOLOv11n {1000*dt11:.1f} ms", (680, 30))
            cv2.imshow("Capture Comparison", comp)

        elif key == ord('s'):
            # Save last capture and annotations
            if last_capture is not None:
                cv2.imwrite("capture_raw.jpg", last_capture)
                print("💾 Saved: capture_raw.jpg")
            if last_annot_pair is not None:
                a8, a11 = last_annot_pair
                cv2.imwrite("capture_yolov8n.jpg", a8)
                cv2.imwrite("capture_yolov11n.jpg", a11)
                print("💾 Saved: capture_yolov8n.jpg, capture_yolov11n.jpg")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
