import sounddevice as sd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.io import wavfile as wf

duration = 2
samplerate = 48000
channels = 16

# --------------------------------
# Mikrofonlayout (spegelvänd)
# --------------------------------

layout = np.array([
    [15,16,1,2],
    [13,14,3,4],
    [11,12,5,6],
    [9,10,7,8]
]) - 1

mic_positions = []
mic_numbers = []

for y in range(4):
    for x in range(4):
        mic_positions.append((x,y))
        mic_numbers.append(layout[y,x])

mic_positions = np.array(mic_positions)
mic_numbers = np.array(mic_numbers)

# --------------------------------
# hitta UMA16
# --------------------------------

device_id = None

for i, dev in enumerate(sd.query_devices()):
    print(i, dev)
    if ("UMA16v2" in dev["name"] and dev["max_input_channels"] == 16) or ("miniDSP" in dev["name"] and dev["max_input_channels"] == 16):
        device_id = i
print(device_id)

if device_id is None:
    raise RuntimeError("UMA16 device not found")

# --------------------------------
# Kamera
# --------------------------------

print("Testing camera indices...")
for i in range(5):
    test_cam = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if test_cam.isOpened():
        print(f"✅ Camera found at index {i}")
        ret, test_frame = test_cam.read()
        if ret:
            print(f"   Resolution: {test_frame.shape}")
        test_cam.release()
    else:
        print(f"❌ No camera at index {i}")

cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cam.isOpened():
    raise RuntimeError("Camera not found")

for _ in range(5):
    cam.read()

ret, frame = cam.read()
cam.release()

if not ret:
    raise RuntimeError("Could not capture camera frame")

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# --------------------------------
# kvadratisk crop
# --------------------------------

h, w, _ = frame.shape
size = min(h, w)

start_x = (w - size) // 2
start_y = (h - size) // 2

frame = frame[start_y:start_y+size, start_x:start_x+size]

# --------------------------------
# spela in ljud
# --------------------------------

print("Recording audio...")

data = sd.rec(int(duration * samplerate),
              samplerate=samplerate,
              channels=channels,
              device=device_id,
              dtype='float32')

sd.wait()

wf.write("recording0.wav", samplerate, data)

print("Audio done")

# --------------------------------
# RMS energi
# --------------------------------

energy = np.zeros(channels)

for ch in range(channels):
    energy[ch] = np.sqrt(np.mean(data[:,ch]**2))

mic_energy = energy[mic_numbers]

# --------------------------------
# interpolation
# --------------------------------

grid_x, grid_y = np.mgrid[0:3:200j, 0:3:200j]

grid_energy = griddata(
    mic_positions,
    mic_energy,
    (grid_x, grid_y),
    method="cubic"
)

grid_energy = np.nan_to_num(grid_energy)

grid_energy -= grid_energy.min()
grid_energy /= grid_energy.max()

# --------------------------------
# plot
# --------------------------------

plt.figure(figsize=(7,7))

plt.imshow(frame)

plt.imshow(
    np.flipud(grid_energy.T),
    cmap="inferno",
    alpha=0.5,
    extent=(0, frame.shape[1], frame.shape[0], 0)
)

# mikrofonpositioner

for i,(x,y) in enumerate(mic_positions):

    px = x / 3 * frame.shape[1]
    py = (3-y) / 3 * frame.shape[0]

    mic = mic_numbers[i] + 1

    plt.scatter(px, py, c="cyan", s=80)

    plt.text(px, py, str(mic),
             color="white",
             ha="center",
             va="center",
             fontsize=11,
             fontweight="bold")

plt.title("UMA-16 Acoustic Heatmap Overlay")
plt.axis("off")

plt.show()