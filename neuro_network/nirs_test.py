import numpy as np
import cv2 as cv

# Функция проверки, является ли многоугольник выпуклым
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
    return True  # Многоугольник выпуклый

# Функция проверки, находится ли точка на достаточном расстоянии от других точек
def is_far_enough(point, vertices, min_dist=10):
    for v in vertices:
        if np.linalg.norm(np.array(point) - np.array(v)) < min_dist:
            return False
    return True

# Функция обработки изображения
def process_image(filename):
    # Чтение изображения
    img = cv.imread(filename)
    if img is None:
        print("Error: Image not found or could not be loaded.")
        return

    # Преобразование изображения в оттенки серого
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Применение пороговой бинаризации и инверсия цветов
    _, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)

    # Поиск контуров на изображении
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("Контуры не найдены")
        return

    vertices = []
    for cnt in contours:
        # Аппроксимация контура до многоугольника
        epsilon = 0.01 * cv.arcLength(cnt, True)  # Степень апроксимации, чем меньше, тем полученный многоугольник к исходному
        approx = cv.approxPolyDP(cnt, epsilon, True)

        for vertex in approx:
            p = tuple(vertex.ravel())  # Преобразование координат вершины в кортеж
            if is_far_enough(p, vertices, min_dist=10):
                vertices.append(p)  # Добавление вершины в список, если она на достаточном расстоянии

    print("Всего точек - ", len(vertices))

    if len(vertices) < 4:
        print("Найдено меньше 4 точек, что нам не подхлодит", end="\n\n")
        return

    # Проверка, является ли многоугольник выпуклым
    if is_convex_polygon(vertices):
        convex_text = "Convex"
        #print("Выпуклый", end="\n\n")
        return 1
    else:
        convex_text = "Not convex"
        #print("Вогнутый", end="\n\n")
        return 0

    # Отображение индексов вершин на изображении
    for i, p in enumerate(vertices):
        cv.putText(img, str(i), p, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Отрисовка контуров на изображении
    cv.drawContours(img, contours, -1, (255, 0, 0), 1, cv.LINE_AA, hierarchy, 1)

    # Добавление текста о выпуклости многоугольника внизу изображения
    text_position = (10, img.shape[0] - 10)
    cv.putText(img, convex_text, text_position, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Отображение изображения с контурами и текстом
    cv.imshow('contours', img)

    # Ожидание нажатия клавиши для закрытия окна
    key = cv.waitKey()
    if key == 27:  # Клавиша 'Esc'
        cv.destroyAllWindows()