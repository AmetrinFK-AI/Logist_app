# app.py

import os
import openai
import csv
from typing import List, Tuple
from flask import Flask, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv
import io
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения из .env файла
load_dotenv()

# Установите ваш OpenAI API ключ
openai.api_key = os.getenv("OPENAI_API_KEY")  # Рекомендуется хранить ключ в переменных окружения

app = Flask(__name__)

# Установка секретного ключа для сессий Flask
app.secret_key = os.getenv("SECRET_KEY")  # Убедитесь, что SECRET_KEY определен в вашем .env файле

# Проверка, что секретный ключ установлен
if not app.secret_key:
    raise RuntimeError("Необходимо установить SECRET_KEY. Пожалуйста, добавьте его в ваш .env файл.")


def is_valid_coordinate(lat: float, lon: float) -> bool:
    """
    Проверяет, находятся ли координаты в допустимом диапазоне.

    Args:
        lat (float): Широта.
        lon (float): Долгота.

    Returns:
        bool: True, если координаты валидны, иначе False.
    """
    return -90 <= lat <= 90 and -180 <= lon <= 180


def load_coordinates_from_csv(file_stream) -> List[Tuple[float, float]]:
    """
    Загружает координаты из CSV файла с автоматическим определением разделителя.

    Args:
        file_stream: Файловый объект.

    Returns:
        List[Tuple[float, float]]: Список кортежей с широтой и долготой.
    """
    coordinates = []
    try:
        # Обернем file_stream в текстовый поток с указанием newline=''
        text_stream = io.TextIOWrapper(file_stream, encoding='utf-8', newline='')
        # Чтение образца для определения диалекта
        sample = text_stream.read(1024)
        text_stream.seek(0)
        # Определение разделителя
        dialect = csv.Sniffer().sniff(sample)
        reader = csv.DictReader(text_stream, dialect=dialect)
        for row in reader:
            # Попытка получить разные варианты заголовков
            lat_str = row.get('latitude') or row.get('lat') or row.get('Latitude') or row.get('Lat')
            lon_str = row.get('longitude') or row.get('lon') or row.get('Longitude') or row.get('Lon')
            if lat_str is None or lon_str is None:
                flash("CSV файл должен содержать заголовки 'latitude' и 'longitude'.", 'danger')
                logger.error("CSV файл не содержит необходимых заголовков.")
                return []
            try:
                lat = float(lat_str)
                lon = float(lon_str)
                if is_valid_coordinate(lat, lon):
                    coordinates.append((lat, lon))
                else:
                    flash(f"Координаты вне допустимого диапазона: {lat}, {lon}", 'warning')
                    logger.warning(f"Координаты вне диапазона: {lat}, {lon}")
            except ValueError:
                flash(f"Некорректный формат координат: {lat_str}, {lon_str}", 'warning')
                logger.warning(f"Некорректный формат координат: {lat_str}, {lon_str}")
    except csv.Error as e:
        flash(f"Ошибка при разборе CSV файла: {e}", 'danger')
        logger.error(f"Ошибка при разборе CSV файла: {e}")
    except Exception as e:
        flash(f"Неизвестная ошибка при загрузке файла CSV: {e}", 'danger')
        logger.error(f"Неизвестная ошибка при загрузке файла CSV: {e}")
    finally:
        # Закрываем текстовый поток
        text_stream.detach()
    return coordinates


from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2


def optimize_route(coordinates: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], str]:
    """
    Оптимизирует маршрут с помощью OpenAI API или OR-Tools в зависимости от количества точек.

    Args:
        coordinates (List[Tuple[float, float]]): Исходный список координат.

    Returns:
        Tuple[List[Tuple[float, float]], str]: Оптимизированный список координат и название использованной технологии.
    """
    if not coordinates:
        return [], ""

    if len(coordinates) <= 15:
        # Используем OpenAI API
        optimized_coords = optimize_route_with_openai(coordinates)
        technology_used = "OpenAI GPT-4"
    else:
        # Используем OR-Tools
        optimized_coords = optimize_route_with_ortools(coordinates)
        technology_used = "Google OR-Tools"

    return optimized_coords, technology_used


def optimize_route_with_openai(coordinates: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Оптимизирует маршрут с использованием OpenAI API.

    Args:
        coordinates (List[Tuple[float, float]]): Исходный список координат.

    Returns:
        List[Tuple[float, float]]: Оптимизированный список координат.
    """
    if not coordinates:
        return []

    # Формируем строку с координатами
    coord_str = '\n'.join([f"{idx + 1}. {lat},{lon}" for idx, (lat, lon) in enumerate(coordinates)])

    prompt = (
        "Оптимизируй порядок следующих координат для минимизации общего расстояния маршрута. "
        "Предоставь оптимизированный порядок как список номеров из исходного списка, без каких-либо дополнительных слов или символов.\n\n"
        f"Исходные точки:\n{coord_str}\n\n"
        "Ответь только номерами точек в оптимизированном порядке, разделенными запятыми. Не добавляй никаких комментариев."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": "Ты помощник для оптимизации маршрутов. Отвечай только запрошенными данными без лишнего текста."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0
        )
        optimized_order_str = response.choices[0].message['content'].strip()
        logger.info(f"Ответ OpenAI API: {optimized_order_str}")

        # Используем регулярное выражение для извлечения чисел
        import re
        numbers = re.findall(r'\d+', optimized_order_str)
        optimized_indices = []
        for num_str in numbers:
            idx = int(num_str) - 1  # Индексация с 0
            if 0 <= idx < len(coordinates):
                optimized_indices.append(idx)

        if len(optimized_indices) != len(coordinates):
            logger.warning("Получен неполный или некорректный порядок оптимизации. Используется исходный порядок.")
            flash('Получен неполный или некорректный порядок оптимизации. Используется исходный порядок.', 'warning')
            return coordinates

        optimized_coords = [coordinates[idx] for idx in optimized_indices]
        return optimized_coords
    except Exception as e:
        logger.error(f"Ошибка при обращении к OpenAI API: {e}")
        flash('Ошибка при оптимизации маршрута с помощью OpenAI API. Пожалуйста, попробуйте позже.', 'danger')
        return coordinates


def compute_euclidean_distance_matrix(locations):
    """Создаёт матрицу евклидовых расстояний между точками."""
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                distances[from_counter][to_counter] = int(
                    ((from_node[0] - to_node[0]) ** 2 + (from_node[1] - to_node[1]) ** 2) ** 0.5 * 100000
                )  # Умножаем на 100000 для преобразования в целые числа
    return distances


def optimize_route_with_ortools(coordinates: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Оптимизирует маршрут с использованием OR-Tools."""
    # Создаём данные
    data = {}
    data['locations'] = coordinates
    data['num_vehicles'] = 1
    data['depot'] = 0

    # Создаём менеджер маршрутов
    manager = pywrapcp.RoutingIndexManager(len(data['locations']), data['num_vehicles'], data['depot'])

    # Создаём модель маршрутов
    routing = pywrapcp.RoutingModel(manager)

    # Создаём матрицу расстояний
    distance_matrix = compute_euclidean_distance_matrix(data['locations'])

    def distance_callback(from_index, to_index):
        """Возвращает расстояние между двумя узлами."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Определяем стоимость ребра
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Настраиваем параметры поиска
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    # Ищем решение
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(data['locations'][node_index])
            index = solution.Value(routing.NextVar(index))
        # Добавляем последнюю точку (возвращение в депо)
        node_index = manager.IndexToNode(index)
        route.append(data['locations'][node_index])
        return route
    else:
        logger.error('Не удалось найти решение с помощью OR-Tools.')
        flash('Не удалось оптимизировать маршрут с помощью OR-Tools. Используется исходный порядок.', 'warning')
        return coordinates


def create_google_maps_link(coordinates: List[Tuple[float, float]]) -> str:
    """
    Формирует ссылку на Google Maps с заданными координатами.

    Args:
        coordinates (List[Tuple[float, float]]): Список координат.

    Returns:
        str: Ссылка на Google Maps с маршрутом.
    """
    base_url = "https://www.google.com/maps/dir/"
    coord_parts = [f"{lat},{lon}" for lat, lon in coordinates]
    route = "/".join(coord_parts)
    return f"{base_url}{route}"


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        method = request.form.get('method')

        coordinates = []

        if method == 'manual':
            # Получение координат из формы
            latitudes = request.form.getlist('latitude')
            longitudes = request.form.getlist('longitude')
            for lat_str, lon_str in zip(latitudes, longitudes):
                try:
                    lat = float(lat_str)
                    lon = float(lon_str)
                    if is_valid_coordinate(lat, lon):
                        coordinates.append((lat, lon))
                    else:
                        flash(f"Координаты вне допустимого диапазона: {lat}, {lon}", 'danger')
                except ValueError:
                    flash(f"Неверный формат координат: {lat_str}, {lon_str}", 'danger')

        elif method == 'csv':
            # Получение файла CSV
            if 'csv_file' not in request.files:
                flash('Файл не загружен.', 'danger')
                return redirect(request.url)
            file = request.files['csv_file']
            if file.filename == '':
                flash('Файл не выбран.', 'danger')
                return redirect(request.url)
            coordinates = load_coordinates_from_csv(file.stream)
            if not coordinates:
                # flash уже был вызван в load_coordinates_from_csv
                return redirect(request.url)

        elif method == 'bulk':
            # Получение координат из текстовой области
            bulk_input = request.form.get('bulk_coordinates')
            if bulk_input:
                parts = bulk_input.strip().split()
                for part in parts:
                    lat_lon = part.split(',')
                    if len(lat_lon) != 2:
                        flash(f"Неверный формат координаты: {part}. Пропускаем.", 'warning')
                        continue
                    try:
                        lat = float(lat_lon[0])
                        lon = float(lat_lon[1])
                        if is_valid_coordinate(lat, lon):
                            coordinates.append((lat, lon))
                        else:
                            flash(f"Координаты вне допустимого диапазона: {lat}, {lon}", 'danger')
                    except ValueError:
                        flash(f"Неверный формат чисел в координате: {part}. Пропускаем.", 'warning')
            else:
                flash('Нет координат для обработки.', 'danger')

        else:
            flash('Неверный метод ввода.', 'danger')
            return redirect(request.url)

        if not coordinates:
            flash('Нет введенных координат. Пожалуйста, попробуйте снова.', 'danger')
            return redirect(request.url)

        # Ограничение на количество координат (например, 100)
        if len(coordinates) > 100:
            flash('Максимальное количество координат — 100.', 'danger')
            return redirect(request.url)

        # Оптимизация маршрута
        optimized_coords, technology_used = optimize_route(coordinates)

        if not optimized_coords:
            flash('Не удалось оптимизировать маршрут.', 'danger')
            return redirect(request.url)

        # Создание ссылки на Google Maps
        google_maps_link = create_google_maps_link(optimized_coords)

        return render_template('result.html', optimized_coords=optimized_coords, google_maps_link=google_maps_link,
                               technology_used=technology_used)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
