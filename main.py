from typing import Tuple, List, Any

import cv2
import yolov5
from dataclasses import dataclass


EPS = 2500
COLOUR = (255, 0, 0)
CAM_URLS = {
    'october_crossroads': 'https://live.cmirit.ru:443/live/0kt-raahe2_1920x1080.stream/playlist.m3u8',
    'arha_bridge': 'https://live.cmirit.ru:443/live/arh-most-002.stream/playlist.m3u8',
    'prospect_pobedi': 'https://live.cmirit.ru:443/live/axis6_704x576.stream/playlist.m3u8'
}


@dataclass
class Data:
    cars_number: int
    avg_speed: int

    def get_info(self):
        if self.avg_speed <= 500 and self.cars_number >= 10:
            return f'Сars on this stretch of road: {self.cars_number}. Possible traffic jam'
        else:
            return f'Cars on this strect of road: {self.cars_number}. There is no traffic jam'

    def __str__(self):
        return f'Cars on road: {self.cars_number}. Avg speed: {self.avg_speed}'


def object_detection(model, img, prev_cars) -> Tuple[Data, List[int], Any]:

    data: Data = Data(0, 0)
    results = model(img)
    predictions = results.pred[0]
    boxes = predictions[:, :4]  # x1, y1, x2, y2

    data.cars_number = len(boxes)
    cur_cars = []

    for i in boxes:
        cur_cars.append(int(i[0]) * int(i[3]) - int(i[1]) * int(i[2]))

    # print('Previous cars: ', prev_cars)

    speeds = []

    if prev_cars:
        for i in prev_cars:
            for j in cur_cars:
                diff = abs(i - j)
                if diff < EPS:
                    speeds.append(diff)

    if speeds:
        data.avg_speed = sum(speeds) / len(speeds)

    return data, cur_cars, boxes


def main():
    freq = 1

    model = yolov5.load('yolov5s.pt')

    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = False
    model.classes = [2., 3.]
    model.multi_label = False
    model.max_det = 1000

    cam_url = CAM_URLS['prospect_pobedi']
    capture = cv2.VideoCapture(cam_url)

    cars = []
    count = 0
    while True:
        success, frame = capture.read()
        if count == freq:
            if not success:
                continue
            ans, cars, boxes = object_detection(model, frame, cars)
            if ans.cars_number > 0:
                for p_group in boxes:
                    first_point = (int(p_group[0]), int(p_group[1]))
                    second_point = (int(p_group[2]), int(p_group[3]))
                    frame = cv2.rectangle(frame, first_point, second_point, COLOUR, 2)
            frame = cv2.resize(frame, (960, 540))
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            count = -1
            # print(ans.get_info())
            # print(ans)
        count += 1


if __name__ == '__main__':
    main()
