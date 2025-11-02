# sends realsense camera video to rtsp server

import pyrealsense2 as rs
import numpy as np
import cv2
import subprocess

# Configure RealSense
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(cfg)

# FFmpeg command to take raw video from stdin and send to RTSP
ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-f",
    "rawvideo",
    "-vcodec",
    "rawvideo",
    "-pix_fmt",
    "bgr24",
    "-s",
    "640x480",
    "-r",
    "30",
    "-i",
    "-",  # read from stdin
    "-c:v",
    "libx264",
    "-preset",
    "ultrafast",
    "-tune",
    "zerolatency",
    "-f",
    "rtsp",
    "rtsp://localhost:8554/mystream",
]

# Launch FFmpeg process
proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

try:
    while True:
        frame = pipe.wait_for_frames()
        color_frame = frame.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow("rgb", color_image)
        # Send raw bytes to ffmpeg
        proc.stdin.write(color_image.tobytes())

        if cv2.waitKey(1) == ord("q"):
            break
except KeyboardInterrupt:
    pass
finally:
    pipe.stop()
    proc.stdin.close()
    proc.wait()
    cv2.destroyAllWindows()
