import argparse
import os
import sys

import cv2
import numpy as np

# Freeman 8-direction code: (dx, dy) -> code   (image y-axis points DOWN)
#   3 2 1
#   4 . 0      0=E 1=NE 2=N 3=NW 4=W 5=SW 6=S 7=SE
#   5 6 7
DIRS = {(1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3,
        (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7}


def make_test_image():
    img = np.full((360, 640, 3), 255, np.uint8)
    cv2.circle(img, (90, 100), 55, (0, 0, 200), -1)                      # circle
    cv2.rectangle(img, (190, 50), (310, 150), (200, 0, 0), -1)           # square
    cv2.fillPoly(img, [np.array([[450, 40], [390, 150], [510, 150]])],
                 (0, 130, 0))                                            # triangle
    cv2.circle(img, (100, 270), 60, (0, 0, 0), -1)                       # ring
    cv2.circle(img, (100, 270), 30, (255, 255, 255), -1)                 # 1 hole
    cv2.rectangle(img, (220, 210), (420, 330), (80, 80, 80), -1)         # 2 holes
    cv2.circle(img, (275, 270), 20, (255, 255, 255), -1)
    cv2.circle(img, (365, 270), 20, (255, 255, 255), -1)
    return img


class SyntheticVideo:
    """Fake camera: red ball bouncing left-right (used when no video is given)."""
    def __init__(self, n=600):
        self.i, self.n = 0, n

    def isOpened(self):
        return True

    def read(self):
        if self.i >= self.n:
            return False, None
        t, self.i = self.i, self.i + 1
        f = np.full((360, 640, 3), 230, np.uint8)
        x = int(60 + abs((t * 6) % 1040 - 520))
        y = int(180 + 100 * np.sin(t / 20))
        cv2.circle(f, (x, y), 22, (0, 0, 200), -1)
        return True, f

    def release(self):
        pass

def segment(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() > 127:                      # objects must be white
        bw = 255 - bw
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def classify(c, area):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    n = len(approx)
    circ = 4 * np.pi * area / (peri ** 2) if peri else 0
    if n == 3:
        return "Triangle"
    if n == 4:
        x, y, w, h = cv2.boundingRect(c)
        return "Square" if 0.9 <= w / float(h) <= 1.1 else "Rectangle"
    if n > 6 or circ > 0.8:
        return "Circle"
    return f"Polygon({n})"


# ------------------------------------------------- Practical 7: chain code
def chain_code(contour):
    pts = contour[:, 0, :]                          # (x, y) points
    steps = np.roll(pts, -1, axis=0) - pts          # closed boundary
    return np.array([DIRS[(int(dx), int(dy))] for dx, dy in steps
                     if (int(dx), int(dy)) in DIRS], dtype=int)


def first_difference(chain):                        # rotation invariant
    return np.mod(chain - np.roll(chain, 1), 8)


def shape_number(fd):                               # min circular rotation
    n = len(fd)
    if n == 0 or n > 1500:                          # skip huge contours (O(n^2))
        return None
    return min(tuple(np.roll(fd, -k)) for k in range(n))


# ------------------------------------------------- Practical 8: Euler number
def euler_bitquad(bw):
    A = np.pad((bw > 0).astype(np.int32), 1)
    a, b, c, d = A[:-1, :-1], A[:-1, 1:], A[1:, :-1], A[1:, 1:]
    s = a + b + c + d
    q1, q3 = (s == 1).sum(), (s == 3).sum()
    qd = ((s == 2) & (a == d)).sum()
    return int(round((q1 - q3 - 2 * qd) / 4)), int(round((q1 - q3 + 2 * qd) / 4))


def analyze(bw, min_area=200, min_hole=15):
    """Find objects; holes come from the contour hierarchy (RETR_CCOMP)."""
    contours, hier = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    objs = []
    if hier is None:
        return objs
    hier = hier[0]                                  # [next, prev, child, parent]
    for i, c in enumerate(contours):
        if hier[i][3] != -1:                        # this contour is a hole
            continue
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        holes = sum(1 for j in range(len(contours))
                    if hier[j][3] == i and cv2.contourArea(contours[j]) >= min_hole)
        m = cv2.moments(c)
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        ch = chain_code(c)
        fd = first_difference(ch)
        objs.append(dict(contour=c, label=classify(c, area), area=area,
                         centroid=(cx, cy), holes=holes, euler=1 - holes,
                         chain=ch, fd=fd, sn=shape_number(fd)))
    return objs



# ------------------------------------------------------------- image mode
def run_image(path, show=True):
    img = cv2.imread(path) if path else None
    if img is None:
        print("No image given/found -> using built-in demo image\n")
        img = make_test_image()
    bw = segment(img)
    objs = analyze(bw)
    out = img.copy()

    print(f"{'ID':<4}{'Shape':<12}{'Area':>8}{'Centroid':>16}{'Holes':>7}{'Euler':>7}{'ChainLen':>10}")
    report = []
    for k, o in enumerate(objs, 1):
        cx, cy = o["centroid"]
        print(f"{k:<4}{o['label']:<12}{o['area']:>8.0f}{f'({cx:.0f},{cy:.0f})':>16}"
              f"{o['holes']:>7}{o['euler']:>7}{len(o['chain']):>10}")
        cv2.drawContours(out, [o["contour"]], -1, (0, 0, 255), 2)
        x, y, w, h = cv2.boundingRect(o["contour"])
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(out, f"#{k} {o['label']} E={o['euler']}", (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        sn = "".join(map(str, o["sn"])) if o["sn"] else "(too long)"
        report.append(f"Object {k} ({o['label']})\n"
                      f"  Chain code   : {''.join(map(str, o['chain']))}\n"
                      f"  First diff   : {''.join(map(str, o['fd']))}\n"
                      f"  Shape number : {sn}\n")

    C, H = len(objs), sum(o["holes"] for o in objs)
    e8, e4 = euler_bitquad(bw)
    print(f"\nObjects C = {C}, Holes H = {H}  ->  Euler number E = C - H = {C - H}")
    print(f"Bit-quad check: E(8-conn) = {e8}, E(4-conn) = {e4}")

    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/annotated.png", out)
    cv2.imwrite("output/mask.png", bw)
    with open("output/report.txt", "w") as f:
        f.write("\n".join(report))
    print("Saved: output/annotated.png, output/mask.png, output/report.txt")

    if show:
        panel = np.hstack([img, cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR), out])
        s = min(1.0, 1600 / panel.shape[1])
        panel = cv2.resize(panel, None, fx=s, fy=s)
        cv2.imshow("Original | Segmented | Detected (any key to close)", panel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ------------------------------------------------- Practical 10: tracking
class Tracker:
    """Constant-velocity Kalman filter + nearest-centroid association."""
    def __init__(self, gate=80, max_lost=15):
        kf = cv2.KalmanFilter(4, 2)          # state [x y vx vy], meas [x y]
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                        [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf, self.gate, self.max_lost = kf, gate, max_lost
        self.active, self.lost, self.trail = False, 0, []

    def update(self, dets):                  # dets: [(cx, cy, (x,y,w,h)), ...]
        if not self.active and dets:
            cx, cy, _ = max(dets, key=lambda d: d[2][2] * d[2][3])   # largest blob
            self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
            self.active, self.lost, self.trail = True, 0, []
        if not self.active:
            return None, None
        pred = self.kf.predict()
        px, py = float(pred[0, 0]), float(pred[1, 0])
        best = None
        if dets:
            best = min(dets, key=lambda d: (d[0] - px) ** 2 + (d[1] - py) ** 2)
            if np.hypot(best[0] - px, best[1] - py) > self.gate:
                best = None
        bbox = None
        if best is not None:
            est = self.kf.correct(np.array([[best[0]], [best[1]]], np.float32))
            px, py = float(est[0, 0]), float(est[1, 0])
            bbox, self.lost = best[2], 0
        else:
            self.lost += 1
        self.trail.append((int(px), int(py)))
        if self.lost > self.max_lost:
            self.active = False
        return (int(px), int(py)), bbox

def run_video(source, save=False):
    if source is None:
        print("No video given -> using synthetic demo video")
        cap = SyntheticVideo()
    else:
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not cap.isOpened():
            sys.exit("Cannot open video source")

    sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40,
                                             detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tracker, writer = Tracker(), None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask = sub.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)   # drop shadows
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dets = []
        for c in cnts:
            if cv2.contourArea(c) > 400:
                x, y, w, h = cv2.boundingRect(c)
                dets.append((x + w / 2.0, y + h / 2.0, (x, y, w, h)))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        pos, bbox = tracker.update(dets)
        if tracker.active and pos:
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Tracked", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.circle(frame, pos, 4, (255, 255, 0), -1)
            for i in range(1, len(tracker.trail)):
                cv2.line(frame, tracker.trail[i - 1], tracker.trail[i], (0, 255, 255), 2)
        cv2.putText(frame, f"Detections: {len(dets)}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if save:
            if writer is None:
                os.makedirs("output", exist_ok=True)
                h_, w_ = frame.shape[:2]
                writer = cv2.VideoWriter("output/tracked.mp4",
                                         cv2.VideoWriter_fourcc(*"mp4v"), 30, (w_, h_))
            writer.write(frame)

        cv2.imshow("Detection + Tracking (ESC to quit)", frame)
        cv2.imshow("Foreground mask", mask)
        if cv2.waitKey(25) & 0xFF == 27:
            break

    cap.release()
    if writer:
        writer.release()
        print("Saved: output/tracked.mp4")
    cv2.destroyAllWindows()

# ------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DIP mini project: Shape Analyzer & Tracker")
    sub = ap.add_subparsers(dest="mode", required=True)
    pi = sub.add_parser("image", help="chain code + Euler number + detection")
    pi.add_argument("path", nargs="?", help="image file (omit for demo)")
    pi.add_argument("--no-show", action="store_true")
    pv = sub.add_parser("video", help="detect + track moving object")
    pv.add_argument("source", nargs="?", help="video file or 0 for webcam (omit for demo)")
    pv.add_argument("--save", action="store_true")
    a = ap.parse_args()

    if a.mode == "image":
        run_image(a.path, show=not a.no_show)
    else:
        run_video(a.source, save=a.save)