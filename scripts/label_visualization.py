from glob import glob
import argparse
import os
import signal
import sys
import zipfile

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def signal_handler(_, __):
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def ensure_extracted(folder_path, zip_path):
    if os.path.isdir(folder_path):
        return True
    if not os.path.isfile(zip_path):
        return False

    print(f"Extracting: {os.path.basename(zip_path)}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(folder_path))

    return os.path.isdir(folder_path)


def show_labels(flight):
    flight_type = "piloted" if "p-" in flight else "autonomous"
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    image_path = os.path.join(repo_root, "data", flight_type, flight, "camera_" + flight)
    label_path = os.path.join(repo_root, "data", flight_type, flight, "label_" + flight)

    ensure_extracted(image_path, image_path + ".zip")
    ensure_extracted(label_path, label_path + ".zip")

    images = sorted(glob(os.path.join(image_path, "*")))
    if not images:
        print(f"No images found in: {image_path}")
        print(f"Expected either folder or zip at: {image_path} / {image_path}.zip")
        return

    colors = ["red", "green", "blue", "gold", "purple", "teal", "orange"]
    keypoint_colors = ["yellow", "lime", "cyan", "magenta"]

    fig, ax = plt.subplots()
    plt.show(block=False)
    state = {"advance": False, "closed": False}

    def on_key(event):
        if event.key == ' ':
            state["advance"] = True

    def on_close(_):
        state["closed"] = True

    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('close_event', on_close)

    for idx, image in enumerate(images):
        if state["closed"]:
            break

        img = Image.open(image)  # read the target image
        ax.clear()
        ax.imshow(img)

        try:
            txt = open(os.path.join(label_path, os.path.basename(image).split(
                ".")[0] + ".txt"), "r")  # read Yolo label in txt format
            for index, label in enumerate(txt.readlines()):
                values = label.split(" ")

                bb = np.array(values[1:5], dtype=np.float32)
                kps = np.array(values[5:], dtype=np.float32).reshape((4, 3))

                # yolo uses x, y, w, h normalized to the image size. We need to convert them to pixel values.
                w = int(bb[2] * img.size[0])
                h = int(bb[3] * img.size[1])
                x = int(bb[0] * img.size[0] - w / 2)
                y = int(bb[1] * img.size[1] - h / 2)
                rect = patches.Rectangle((x, y), w, h, linewidth=3,
                                         edgecolor=colors[index], facecolor='none')
                ax.add_patch(rect)

                for kp_index, kp in enumerate(kps):
                    if kp[0] == 0 and kp[1] == 0:
                        continue
                    x = int(kp[0] * img.size[0])
                    y = int(kp[1] * img.size[1])
                    # non-visible corners are draw in white.
                    color = keypoint_colors[kp_index] if kp[2] == 2 else "w"
                    ax.scatter(x, y, marker="o", color=color, s=18)
            txt.close()
        except Exception:
            pass

        ax.set_title('Frame ' + str(idx) + ' (press SPACE to advance)')
        ax.axis('off')  # Turn off axis ticks and labels
        fig.canvas.draw_idle()

        state["advance"] = False
        while not state["advance"] and not state["closed"]:
            plt.pause(0.05)

    if not state["closed"]:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--flight', required=True, help="Flight ID (e.g., flight-01p-ellipse)")
    args = parser.parse_args()
    show_labels(args.flight)
