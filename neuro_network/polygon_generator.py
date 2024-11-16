import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random
import os
#import wtf

def is_far_enough(point, vertices, min_dist):
    for v in vertices:
        if np.linalg.norm(np.array(point) - np.array(v)) < min_dist:
            return False
    return True

def is_convex_polygon(vertices):
    num_vertices = len(vertices)
    # Функция для вычисления векторного произведения трех точек
    def cross_product(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    sign = None
    for i in range(num_vertices):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % num_vertices]
        p3 = vertices[(i + 2) % num_vertices]
        cp = cross_product(p1, p2, p3)
        if cp != 0:  # Если произведение не равно нулю
            if sign is None:
                sign = cp > 0
            elif sign != (cp > 0):
                return False  # Найдено противоположное направление, значит многоугольник не выпуклый
    return True

def generate_random_polygon(num_points, i, flag, size=100, margin=25):
    """
    Генерация случайного многоугольника из заданного числа точек с заливкой.
    
    :param num_points: Количество точек многоугольника
    :param i: Номер текущего многоугольника для именования файла
    :param flag: Флаг для определения выпуклого или вогнутого многоугольника (1 для выпуклого, 0 для вогнутого)
    :param size: Размер изображения (по умолчанию 400x400 пикселей)
    :param margin: Отступ от краев изображения (по умолчанию 25 пикселей)
    :return: 1 если успешное создание выпуклого/вогнутого многоугольника, иначе 0
    """
    if num_points<4: 
        print("Для определения выпуклости у многоугольника должно быть больше 3 точек")
        return 0

    # Создание случайных координат с учетом отступов
    points = (np.random.rand(num_points, 2) * (size - 2 * margin)) + margin
    
    # Найдем центр точек
    centroid = np.mean(points, axis=0)
    
    # Рассчитаем углы от центра для сортировки точек
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    
    # Сортировка точек по углам против часовой стрелки
    sorted_points = points[np.argsort(angles)]
    
    # Построение многоугольника с заливкой
    plt.figure(figsize=(size / 100, size / 100), dpi=100)
    plt.fill(sorted_points[:, 0], sorted_points[:, 1], edgecolor='black', fill=True, facecolor='skyblue')
    plt.plot(sorted_points[:, 0], sorted_points[:, 1], 'o', color='black', markersize=0)
    
    # Настройка графика
    plt.xlim(0, size)
    plt.ylim(0, size)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    
    # Сохранение изображения во временный файл
    temp_filename = 'temp_image.png'
    plt.savefig(temp_filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Загрузка сохраненного изображения для обработки 
    img = cv.imread(temp_filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV) # Применение пороговой бинаризации и инверсия цветов
    
    # Поиск контуров на изображении
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        os.remove(temp_filename)
        return 0

    vertices = []
    for cnt in contours:
        # Аппроксимация контура до многоугольника
        epsilon = 0.01 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)    # Результат аппроксимации содержащий вершины контура
        
        for vertex in approx:
            p = tuple(vertex.ravel())  # Преобразование координат вершины в кортеж
            if is_far_enough(p, vertices, min_dist=15):
                vertices.append(p)  # Добавление вершины в список, если она на достаточном расстоянии
    
    if len(vertices) < 4:
        os.remove(temp_filename)
        return 0
    
    is_convex = is_convex_polygon(vertices)

    # создает выпуклые и вогнутые многоугольники в соотношении 1 к 1 
    if flag == 1 and is_convex:
        # Переименование временного файла в целевую папку
        target_filename = f"data1/polygon_test.png"
        os.rename(temp_filename, target_filename)
        return 1
    elif flag == 0 and not is_convex:
        target_filename = f"data1/polygon_test.png"
        os.rename(temp_filename, target_filename)
        return 1

    os.remove(temp_filename)
    return 0

if __name__ == '__main__':
    os.makedirs("data1", exist_ok=True)
    cou = 1
    while cou != 2:
        num_points = random.randint(4, 7)  # Случайное количество точек для многоугольника
        flag = cou % 2  # 1 для нечетных, 0 для четных
        if generate_random_polygon(num_points, cou, flag) == 1:
            cou += 1
