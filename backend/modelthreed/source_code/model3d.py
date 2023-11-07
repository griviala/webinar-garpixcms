from math import sqrt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from django.conf import settings
from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d
import json
from .consts import IMAGE_PATH, BOX_QUEUE


class PalletDetect:
    def __init__(self, image: np.array, N: int, M: int):
        self.image = image
        self.grid = np.array([M, N])
        self.minSquareLimit = 200
        self.box = None

    def run(self):
        image = self.image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 100, 13, cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        max_box = None

        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) <= self.minSquareLimit:
                continue
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Вычисление площади
            area = int(abs((box[2][0] - box[0][0]) * (box[2][1] - box[0][1])))

            # Проверка на максимальную площадь
            if area > max_area:
                max_area = area
                max_box = box

        if max_box is not None:
            cv2.drawContours(image, [max_box], 0, (0, 0, 255), 2)
            self.box = [max_box]

        self.result_image = image

    def show(self):
        image = cv2.resize(self.result_image, [640, 480])
        cv2.imshow("Name", image)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    def get_box(self):
        """Returns box of pallet"""
        return self.box


class Pallet():
    def __init__(self, imagePath, x=75, y=75):
        self.x = x
        self.y = y
        self.imagePath = imagePath

    def find_bounding_box(self):
        img = cv2.imread(self.imagePath)
        pDet = PalletDetect(img, self.x, self.y)
        pDet.run()
        boundingBox = pDet.get_box()
        self.startX = min(boundingBox[0][0][1], boundingBox[0][2][1])
        self.startY = min(boundingBox[0][0][0], boundingBox[0][2][0])
        self.endX = max(boundingBox[0][0][1], boundingBox[0][2][1])
        self.endY = max(boundingBox[0][0][0], boundingBox[0][2][0])
        self.dx = self.endX - self.startX
        self.dy = self.endY - self.startY


class Box():
    def __init__(self, boxId, x, y, z, imagePath):
        self.id = boxId
        self.x = x
        self.y = y
        self.z = z
        self.imagePath = imagePath

    def find_box_start_point_3d(self, pallet: Pallet, queue):
        """Поиск стартовой точки для отрисовки (x и y переставлены местами)"""
        self.startPointX = self.startCoordinate[1] / 361 * pallet.x * 0.01
        self.startPointY = self.startCoordinate[0] / 349 * pallet.x * 0.01
        if self.underBoxId == -1:
            self.startPointZ = 0
        # TODO find z coordinate using boxes query
        else:
            for box in queue:
                if self.underBoxId == box.id:
                    self.startPointZ = box.z * 0.01

    def find_box_sizes_in_3d(self):
        """Поиск сторон для отрисовки (x и y переставлены местами)"""
        self.dx = round(float(self.y) * 0.01, 4)
        self.dy = round(float(self.x) * 0.01, 4)
        self.dz = round(float(self.z) * 0.01, 4)

    def find_box_axes(self, pallet: Pallet):
        """"Поиск настоящей ориентации сторон коробки при укладки (плоскость, которой положили на паллет)"""
        # Процентное соотношение сторон коробки к сторонам паллета, подразумевается, что паллет квадратный
        percentX = self.x / pallet.x
        percentY = self.y / pallet.x
        percentZ = self.z / pallet.x
        # Процентное соотношение разницы координат коробки к разнице координат паллета
        percentCoordinateX = (self.endCoordinate[0] - self.startCoordinate[0]) / 349
        percentCoordinateY = (self.endCoordinate[1] - self.startCoordinate[1]) / 361
        # Стартовые размеры коробки без учета ориентации сторон
        realX, realY, realZ = self.x, self.y, self.z
        if abs(percentX - percentCoordinateX) > abs(percentY - percentCoordinateX):
            realX, realY, realZ = realY, realX, realZ
            percentX, percentY, percentZ = percentY, percentX, percentZ
            if abs(percentX - percentCoordinateX) > abs(percentZ - percentCoordinateX):
                realX, realY, realZ = realZ, realY, realX
                percentX, percentY, percentZ = percentZ, percentY, percentX
        if abs(percentY - percentCoordinateY) > abs(percentZ - percentCoordinateY):
            realX, realY, realZ = realX, realZ, realY
            percentX, percentY, percentZ = percentX, percentZ, percentY
            if abs(percentX - percentCoordinateY) > abs(percentY - percentCoordinateY):
                realX, realY, realZ = realY, realX, realZ
                percentX, percentY, percentZ = percentZ, percentY, percentX
        if abs(percentX - percentCoordinateX) > abs(percentZ - percentCoordinateX):
            realX, realY, realZ = realZ, realY, realX

        # if abs(percentX - percentCoordinateX) > (percentY - percentCoordinateX):
        #     realX, realY, realZ = realY, realX, realZ
        #     if abs(percentX - percentCoordinateX) > (percentZ - percentCoordinateX):
        #         realX, realY, realZ = realZ, realY, realX
        #         if abs(percentY - percentCoordinateY) > abs(percentZ - percentCoordinateY):
        #            realX, realY, realZ = realX, realZ, realY 
        # elif abs(percentX - percentCoordinateX) > (percentZ - percentCoordinateX):
        #     realX, realY, realZ = realZ, realY, realX
        #     if abs(percentY - percentCoordinateY) > abs(percentZ - percentCoordinateY):
        #         realX, realY, realZ = realX, realZ, realY
        # elif abs(percentY - percentCoordinateY) > abs(percentZ - percentCoordinateY):
        #         realX, realY, realZ = realX, realZ, realY

        self.x = realX
        self.y = realY
        self.z = realZ

    def find_underBoxId(self, packer, lastBoxQueueNumber):
        """Поиск коробки внизу, если коробка есть, присваиваем id коробки в атрибут underBoxId, если нет, то присваиваем -1 в тот же атрибут"""
        underBoxId = -1
        maxInterception = 0
        for box in packer.queue[1:lastBoxQueueNumber:]:
            # Левый верхний
            if self.startCoordinate[0] > box.startCoordinate[0] and self.startCoordinate[1] > box.startCoordinate[1] and \
                    self.startCoordinate[0] < box.endCoordinate[0] and self.startCoordinate[1] < box.endCoordinate[1]:
                intersection_area = (box.endCoordinate[0] - self.startCoordinate[0]) * (
                        box.endCoordinate[1] - self.startCoordinate[1])
                iou = intersection_area / float(self.area + box.area - intersection_area)
                if abs(iou) > 1:
                    if abs(iou) > maxInterception:
                        maxInterception = abs(iou)
                        underBoxId = box.id
            # Правый верхний
            if self.endCoordinate[0] > box.startCoordinate[0] and self.startCoordinate[1] > box.startCoordinate[1] and \
                    self.endCoordinate[0] < box.endCoordinate[0] and self.startCoordinate[1] < box.endCoordinate[1]:
                intersection_area = (self.endCoordinate[0] - box.startCoordinate[0]) * (
                        box.endCoordinate[1] - self.startCoordinate[1])
                iou = intersection_area / float(self.area + box.area - intersection_area)
                if abs(iou) > 1:
                    if abs(iou) > maxInterception:
                        maxInterception = abs(iou)
                        underBoxId = box.id
            # Левый нижний
            if self.startCoordinate[0] > box.startCoordinate[0] and self.endCoordinate[1] > box.startCoordinate[1] and \
                    self.startCoordinate[0] < box.endCoordinate[0] and self.endCoordinate[1] < box.endCoordinate[1]:
                intersection_area = (box.endCoordinate[0] - self.startCoordinate[0]) * (
                        self.startCoordinate[1] - box.startCoordinate[1])
                iou = intersection_area / float(self.area + box.area - intersection_area)
                if abs(iou) > 1:
                    if abs(iou) > maxInterception:
                        maxInterception = abs(iou)
                        underBoxId = box.id
            # Правый нижний
            if self.endCoordinate[0] > box.startCoordinate[0] and self.endCoordinate[1] > box.startCoordinate[1] and \
                    self.endCoordinate[0] < box.endCoordinate[0] and self.endCoordinate[1] < box.endCoordinate[1]:
                intersection_area = (self.endCoordinate[0] - box.startCoordinate[0]) * (
                        self.endCoordinate[1] - box.startCoordinate[1])
                iou = intersection_area / float(self.area + box.area - intersection_area)
                if abs(iou) > 1:
                    if abs(iou) > maxInterception:
                        maxInterception = abs(iou)
                        underBoxId = box.id
        self.underBoxId = underBoxId

    def get_bounding_box_coordinates(self, previousObject, pallet: Pallet):
        """Поиск координат контура новой коробки"""
        # Загрузка изображений
        image1 = cv2.imread(self.imagePath)
        image2 = cv2.imread(previousObject.imagePath)

        image1 = image1[pallet.startX:pallet.endX, pallet.startY:pallet.endY]
        image2 = image2[pallet.startX:pallet.endX, pallet.startY:pallet.endY]
        # Преобразование изображений в оттенки серого
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Убираем тени
        smooth1 = cv2.GaussianBlur(gray1, (95, 95), 0)
        smooth2 = cv2.GaussianBlur(gray2, (95, 95), 0)

        division1 = cv2.divide(gray1, smooth1, scale=192)
        division2 = cv2.divide(gray2, smooth2, scale=192)

        # # Вычитание фонового изображения
        diff = cv2.absdiff(division2, division1)
        # diff = cv2.absdiff(gray2, gray1)
        # Применение пороговой обработки
        threshold = 35
        _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Нахождение контуров объектов
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Выделение объекта на втором изображении
        maxContour = contours[0]
        for contour in contours[1::]:
            # Searching max area of contours
            if cv2.contourArea(contour) > cv2.contourArea(maxContour):
                maxContour = contour

        (x, y, w, h) = cv2.boundingRect(maxContour)
        cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        self.area = cv2.contourArea(maxContour)
        self.startCoordinate = (x, y)
        self.endCoordinate = (x + w, y + h)

    def prepare_image():
        pass


class Packer():
    def __init__(self, pallet: Pallet):
        self.pallet = pallet
        self.queue = [pallet]

    def append_box(self, box: Box):
        self.queue.append(box)


class Painter():
    def __init__(self):
        pass

    def draw_box(self, box: Box, ax):
        x = box.startPointX
        y = box.startPointY
        z = box.startPointZ
        dx = box.dx
        dy = box.dy
        dz = box.dz

        p = Rectangle((x, y), dx, dy, fc="red", ec='black')  # низ
        p2 = Rectangle((x, y), dx, dy, fc="red", ec='black')  # вверх
        # one side
        p3 = Rectangle((y, z), dy, dz, fc="red", ec='black')  #
        p4 = Rectangle((y, z), dy, dz, fc="red", ec='black')  #
        # another side
        p5 = Rectangle((x, z), dx, dz, fc="red", ec='black')  #
        p6 = Rectangle((x, z), dx, dz, fc="red", ec='black')  #
        ax.add_patch(p)
        ax.add_patch(p2)
        ax.add_patch(p3)
        ax.add_patch(p4)
        ax.add_patch(p5)
        ax.add_patch(p6)
        art3d.pathpatch_2d_to_3d(p, z=z, zdir="z")
        art3d.pathpatch_2d_to_3d(p2, z=z + dz, zdir="z")
        art3d.pathpatch_2d_to_3d(p3, z=x, zdir="x")
        art3d.pathpatch_2d_to_3d(p4, z=x + dx, zdir="x")
        art3d.pathpatch_2d_to_3d(p5, z=y, zdir="y")
        art3d.pathpatch_2d_to_3d(p6, z=y + dy, zdir="y")


class Model3d():
    def __init__(self, imagesPath=IMAGE_PATH, boxQueue=BOX_QUEUE):
        plt.switch_backend('Agg')
        self.imagePath = imagesPath
        self.boxQueue = boxQueue
        self.fig = plt.figure()
        self.axGlob = plt.axes(projection='3d')

    def get_boxes(self):
        boxes = {"boxes": []}
        with open(self.boxQueue) as f:
            lines = f.readlines()
            for row in lines[1::]:
                row = row.split(',')
                box = {"id": int(row[0]), "x": float(row[1]), "y": float(row[2]), "z": float(row[3])}
                boxes["boxes"].append(box)
        self.boxes = boxes

    def get_images(self):
        dirname = self.imagePath
        imagePathes = []
        imagesNumbers = []
        for file in os.listdir(dirname):
            filename = dirname + file
            imagePathes.append(filename)
            imagesNumbers.append(int(file.split('_')[0]))
        imagePathes = [x for y, x in sorted(zip(imagesNumbers, imagePathes))]
        self.imagePathes = imagePathes

    def find_pallet_bounding_box(self):
        self.pallet = Pallet(self.imagePathes[0])
        self.pallet.find_bounding_box()

    def create_packer(self):
        self.packer = Packer(self.pallet)

    def create_painter(self):
        self.painter = Painter()

    def create_model(self):
        self.packedBoxesInfo = []
        self.unpacked_cargos_info = []
        i = 2
        iteration = 0
        for box in self.boxes["boxes"]:
            boxesInfo = {}
            imagePath = self.imagePathes[i - 1]
            i += 1
            box = Box(boxId=box["id"], x=box["x"], y=box["y"], z=box["z"], imagePath=imagePath)
            # TODO поле имен
            boxesInfo["calculated_size"] = {
                "width": round(float(box.x) * 0.01, 4),
                "length": round(float(box.y) * 0.01, 4),
                "height": round(float(box.z) * 0.01, 4)}
            boxesInfo["cargo_id"] = box.id
            boxesInfo["id"] = box.id
            boxesInfo["mass"] = 1

            self.packer.append_box(box)
            box.get_bounding_box_coordinates(self.packer.queue[iteration], self.pallet)
            box.find_underBoxId(self.packer, i - 1)
            iteration += 1
            box.find_box_axes(self.pallet)
            box.find_box_sizes_in_3d()
            box.find_box_start_point_3d(pallet=self.pallet, queue=self.packer.queue[1::])
            boxesInfo["position"] = {
                "x": round(float(box.startPointZ) + round(float(box.z)) * 0.01 / 2, 4),
                "y": round(float(box.startPointY) + round(float(box.y)) * 0.01 / 2, 4),
                "z": round(float(box.startPointX) + round(float(box.x)) * 0.01 / 2, 4)}
            boxesInfo["size"] = {
                "width": round(float(box.x) * 0.01, 4),
                "height": round(float(box.y) * 0.01, 4),
                "length": round(float(box.z) * 0.01, 4)
            }
            boxesInfo["sort"] = 1
            boxesInfo["stacking"] = True
            boxesInfo["turnover"] = True
            boxesInfo["type"] = "box"
            self.packedBoxesInfo.append(boxesInfo)
            self.painter.draw_box(box, self.axGlob)
        plt.savefig(os.path.join(settings.MEDIA_ROOT, 'plot.png'), bbox_inches='tight', pad_inches=0)

    def create_output_json(self):
        outputDict = {
            "cargoSpace": {
                "loading_size": {
                    "length": 0 * 0.01,
                    "height": self.pallet.y * 0.01,
                    "width": self.pallet.x * 0.01
                },
                "position": [
                    0 * 0.01 / 2,
                    self.pallet.y * 0.01 / 2,
                    self.pallet.x * 0.01 / 2
                ],
                "type": "pallet"
            },
            "cargos": self.packedBoxesInfo,
            "unpacked": self.unpacked_cargos_info
        }

        # with open("./Output/output.json", 'w') as fp:
        #     json.dump(outputDict, fp)

        return outputDict

    def run(self):
        self.get_boxes()
        self.get_images()
        self.find_pallet_bounding_box()
        self.create_packer()
        self.create_painter()
        self.create_model()
        return self.create_output_json()
