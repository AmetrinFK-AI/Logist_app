# app.py

import os
import openai
import csv
from typing import List, Tuple, Optional
from flask import Flask, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv
import io
import logging
import urllib.parse

# Дополнительные библиотеки
from geopy.distance import geodesic
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения из .env файла
load_dotenv()

# Установите ваш OpenAI API ключ
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Установка секретного ключа для сессий Flask
app.secret_key = os.getenv("SECRET_KEY")

# Проверка, что секретный ключ установлен
if not app.secret_key:
    raise RuntimeError("Необходимо установить SECRET_KEY. Пожалуйста, добавьте его в ваш .env файл.")

def is_valid_coordinate(lat: float, lon: float) -> bool:
    """
    Проверяет, находятся ли координаты в допустимом диапазоне.
    """
    return -90 <= lat <= 90 and -180 <= lon <= 180

def load_coordinates_from_csv(file_stream) -> List[Tuple[float, float]]:
    """
    Загружает координаты из CSV файла с автоматическим определением разделителя.
    """
    coordinates = []
    try:
        text_stream = io.TextIOWrapper(file_stream, encoding='utf-8', newline='')
        sample = text_stream.read(1024)
        text_stream.seek(0)
        dialect = csv.Sniffer().sniff(sample)
        reader = csv.DictReader(text_stream, dialect=dialect)
        for row in reader:
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
        try:
            text_stream.detach()
        except:
            pass
    return coordinates

def optimize_route(coordinates: List[Tuple[float, float]]) -> Tuple[Optional[List[Tuple[float, float]]], str]:
    """
    Оптимизирует маршрут с помощью OpenAI API или OR-Tools в зависимости от количества точек.
    Возвращает оптимизированный список координат и использованную технологию.
    """
    if not coordinates:
        return [], ""

    if len(coordinates) <= 10:
        optimized_coords = optimize_route_with_openai(coordinates)
        if optimized_coords is not None:
            technology_used = "OpenAI GPT-3.5-turbo"
            return optimized_coords, technology_used
        else:
            flash('Ошибка при оптимизации с помощью OpenAI API. Используется OR-Tools.', 'warning')

    # Используем OR-Tools
    optimized_coords = optimize_route_with_ortools(coordinates)
    if optimized_coords is not None:
        technology_used = "OR-Tools"
        return optimized_coords, technology_used
    else:
        flash('Ошибка при оптимизации маршрута с помощью OR-Tools.', 'danger')
        return None, ""

def optimize_route_with_openai(coordinates: List[Tuple[float, float]]) -> Optional[List[Tuple[float, float]]]:
    """
    Оптимизирует маршрут с использованием OpenAI API.
    """
    if not coordinates:
        return []

    coord_str = '\n'.join([f"{idx + 1}. {lat},{lon}" for idx, (lat, lon) in enumerate(coordinates)])

    prompt = (
        "Оптимизируй порядок следующих координат для минимизации общего расстояния маршрута. "
        "Предоставь оптимизированный порядок как список номеров из исходного списка, без каких-либо дополнительных слов или символов.\n\n"
        f"Исходные точки:\n{coord_str}\n\n"
        "Ответь только номерами точек в оптимизированном порядке, разделенными запятыми. Не добавляй никаких комментариев."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
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

        import re
        if not re.match(r'^(\d+\s*,\s*)*\d+\s*$', optimized_order_str):
            logger.warning("Формат ответа некорректен.")
            flash('Некорректный ответ от OpenAI API.', 'warning')
            return None

        numbers = re.findall(r'\d+', optimized_order_str)
        optimized_indices = [int(num) - 1 for num in numbers]

        # Проверка индексов
        if any(idx < 0 or idx >= len(coordinates) for idx in optimized_indices):
            logger.warning("Некорректные индексы в ответе от OpenAI API.")
            flash('Некорректные индексы в ответе от OpenAI API.', 'warning')
            return None

        # Проверка на дубликаты и пропуски
        if len(set(optimized_indices)) != len(coordinates):
            logger.warning("Получены дубликаты или отсутствующие индексы.")
            flash('Получены дубликаты или отсутствующие индексы в ответе от OpenAI API.', 'warning')
            return None

        optimized_coords = [coordinates[idx] for idx in optimized_indices]
        return optimized_coords
    except openai.error.RateLimitError as e:
        logger.error(f"OpenAI API RateLimitError: {e}", exc_info=True)
        flash('Превышен лимит запросов к OpenAI API. Попробуйте позже или используйте OR-Tools.', 'danger')
        return None
    except Exception as e:
        logger.error(f"Ошибка при обращении к OpenAI API: {e}", exc_info=True)
        flash('Ошибка при обращении к OpenAI API. Пожалуйста, проверьте ваш API-ключ и доступ к модели.', 'danger')
        return None

def compute_distance_matrix(locations: List[Tuple[float, float]]) -> List[List[int]]:
    """
    Создает матрицу расстояний между точками с использованием геодезических расчетов.
    Возвращает матрицу расстояний как список списков.
    """
    size = len(locations)
    distance_matrix = []

    for from_node in range(size):
        row = []
        for to_node in range(size):
            if from_node == to_node:
                row.append(0)
            else:
                lat1, lon1 = locations[from_node]
                lat2, lon2 = locations[to_node]
                distance = geodesic((lat1, lon1), (lat2, lon2)).meters
                row.append(int(distance))
        distance_matrix.append(row)
    return distance_matrix

def optimize_route_with_ortools(coordinates: List[Tuple[float, float]]) -> Optional[List[Tuple[float, float]]]:
    """
    Оптимизирует маршрут с использованием OR-Tools.
    Возвращает оптимизированный список координат.
    """
    try:
        data = {'locations': coordinates, 'num_vehicles': 1, 'depot': 0}
        manager = pywrapcp.RoutingIndexManager(len(data['locations']), data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)
        distance_matrix = compute_distance_matrix(data['locations'])

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = 30

        solution = routing.SolveWithParameters(search_parameters)
        if solution:
            route = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(data['locations'][node_index])
                index = solution.Value(routing.NextVar(index))
            # Добавляем конечную точку
            node_index = manager.IndexToNode(index)
            route.append(data['locations'][node_index])
            return route
        else:
            logger.error('Не удалось найти решение с помощью OR-Tools.')
            flash('Не удалось найти решение с помощью OR-Tools.', 'danger')
            return None
    except Exception as e:
        logger.error(f"Ошибка в optimize_route_with_ortools: {e}", exc_info=True)
        flash('Ошибка при оптимизации маршрута с помощью OR-Tools.', 'danger')
        return None

def create_google_maps_links(coordinates: List[Tuple[float, float]]) -> List[str]:
    """
    Создает ссылки на Google Карты, где каждая ссылка содержит не более 24 точек координат.
    """
    MAX_POINTS_PER_LINK = 24  # Максимальное количество точек в одной ссылке

    links = []
    total_points = len(coordinates)

    # Разбиваем список координат на части по MAX_POINTS_PER_LINK точек
    for i in range(0, total_points, MAX_POINTS_PER_LINK):
        chunk = coordinates[i:i + MAX_POINTS_PER_LINK]

        if not chunk:
            continue

        # Используем функционал из первого кода для генерации ссылки
        base_url = "https://www.google.com/maps/dir/"
        coord_parts = [f"{lat},{lon}" for lat, lon in chunk]
        route = "/".join(coord_parts)
        url = f"{base_url}{route}"
        links.append(url)

    return links

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        method = request.form.get('method')

        coordinates = []

        if method == 'manual':
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
            if 'csv_file' not in request.files:
                flash('Файл не загружен.', 'danger')
                return redirect(request.url)
            file = request.files['csv_file']
            if file.filename == '':
                flash('Файл не выбран.', 'danger')
                return redirect(request.url)
            coordinates = load_coordinates_from_csv(file.stream)
            if not coordinates:
                return redirect(request.url)

        elif method == 'bulk':
            bulk_input = request.form.get('bulk_coordinates')
            if bulk_input:
                lines = bulk_input.strip().split('\n')
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) != 2:
                        flash(f"Неверный формат координаты: {line}. Пропускаем.", 'warning')
                        continue
                    try:
                        lat = float(parts[0].strip())
                        lon = float(parts[1].strip())
                        if is_valid_coordinate(lat, lon):
                            coordinates.append((lat, lon))
                        else:
                            flash(f"Координаты вне допустимого диапазона: {lat}, {lon}", 'danger')
                    except ValueError:
                        flash(f"Неверный формат чисел в координате: {line}. Пропускаем.", 'warning')
            else:
                flash('Нет координат для обработки.', 'danger')

        else:
            flash('Неверный метод ввода.', 'danger')
            return redirect(request.url)

        if not coordinates:
            flash('Нет введенных координат. Пожалуйста, попробуйте снова.', 'danger')
            return redirect(request.url)

        # Оптимизация маршрута
        optimized_coords, technology_used = optimize_route(coordinates)

        if not optimized_coords:
            flash('Не удалось оптимизировать маршрут.', 'danger')
            return redirect(request.url)

        # Создание ссылок на Google Карты
        google_maps_links = create_google_maps_links(optimized_coords)

        if not google_maps_links:
            flash('Не удалось создать ссылки на Google Карты.', 'danger')
            return redirect(request.url)

        return render_template('result.html', optimized_coords=optimized_coords,
                               google_maps_links=google_maps_links,
                               technology_used=technology_used)

    # Если метод запроса GET, просто отобразить главную страницу
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
