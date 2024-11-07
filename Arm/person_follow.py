#!/usr/bin/env python
# coding: utf-8
import cv2 as cv
from ultralytics import YOLO
from utils.Arm_Lib import Arm_Device
from utils.PID import PositionalPID


class face_follow:
    def __init__(self):
        self.Arm = Arm_Device()
        self.target_servox = 90
        self.target_servoy = 45
        self.a = 0
        self.b = 0
        self.xservo_pid = PositionalPID(0.25, 0.1, 0.05)
        self.yservo_pid = PositionalPID(0.25, 0.1, 0.05)
        self.model = YOLO("./models/yolov8s-pose.pt")
        self.conf_threshold = 0.5

    def follow_function(self, img):
        self.a = 0
        self.b = 0
        img = cv.resize(img, (640, 480))

        # 使用YOLOv8进行姿态估计
        results = self.model(img, conf=self.conf_threshold)

        if len(results[0].keypoints.xy[0]) > 0:
            # 获取第一个检测到的人的关键点
            keypoints = results[0].keypoints.xy[0]

            # 使用鼻子的位置作为跟踪点 (关键点索引0为鼻子)
            nose = keypoints[0]
            x, y = nose

            # 在原图上绘制关键点
            annotated_frame = results[0].plot()

            cv.putText(
                annotated_frame,
                "Person",
                (280, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (105, 105, 105),
                2,
            )

            if (260 < x < 380) and (180 < y < 300):
                self.b = 1
                self.a = 1

            if not (
                self.target_servox >= 180
                and x <= 320
                and self.a == 1
                or self.target_servox <= 0
                and x >= 320
                and self.a == 1
            ):
                if self.a == 0:
                    self.xservo_pid.SystemOutput = x
                    self.xservo_pid.SetStepSignal(320)
                    self.xservo_pid.SetInertiaTime(0.01, 0.1)
                    self.target_servox = int((self.xservo_pid.SystemOutput + 1000) / 10)

                    if self.target_servox > 180:
                        self.target_servox = 180
                    if self.target_servox < 0:
                        self.target_servox = 0

            if not (
                self.target_servoy >= 180
                and y <= 240
                and self.b == 1
                or self.target_servoy <= 0
                and y >= 240
                and self.b == 1
            ):
                if self.b == 0:
                    self.yservo_pid.SystemOutput = y
                    self.yservo_pid.SetStepSignal(240)
                    self.yservo_pid.SetInertiaTime(0.01, 0.1)
                    self.target_servoy = (
                        int((self.yservo_pid.SystemOutput + 1000) / 10) - 45
                    )

                    if self.target_servoy > 360:
                        self.target_servoy = 360
                    if self.target_servoy < 0:
                        self.target_servoy = 0

            joints_0 = [
                self.target_servox / 1,
                135,
                self.target_servoy / 2,
                self.target_servoy / 2,
                90,
                90,
            ]
            self.Arm.Arm_serial_servo_write6_array(joints_0, 500)

            print(joints_0)

            return annotated_frame

        return img


if __name__ == "__main__":
    face_follow = face_follow()
    cap = cv.VideoCapture(2)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated_frame = face_follow.follow_function(frame)
        cv.imshow("YOLOv8 Pose", annotated_frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()
