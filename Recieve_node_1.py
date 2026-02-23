#cd ESP_train/
import socket
import struct
import cv2
import numpy as np

HOST = "0.0.0.0"
PORT = 5001

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))

print("UDP Receiver listening on port", PORT)

while True:
    data, addr = sock.recvfrom(65535)  # UDP max safe size

    if len(data) < 9:
        continue

    cam_id = data[0]
    size = struct.unpack("!Q", data[1:9])[0]
    jpg = data[9:9+size]

    if len(jpg) != size:
        continue

    frame = cv2.imdecode(
        np.frombuffer(jpg, np.uint8),
        cv2.IMREAD_COLOR
    )

    if frame is None:
        continue

    if cam_id == 1:
        cv2.imshow("CAM 1 - AI NODE 1", frame)
    elif cam_id == 2:
        cv2.imshow("CAM 2 - RAW", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

sock.close()
cv2.destroyAllWindows()


