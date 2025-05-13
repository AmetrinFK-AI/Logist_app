# streamlit_app.py
# ─────────────────────────────────────────────────────────────────────
# Запуск:  streamlit run streamlit_app.py
# Требует: streamlit, python-dotenv, openai, geopy, ortools
# ─────────────────────────────────────────────────────────────────────
import os
import io
import csv
import re
import logging
from typing import List, Tuple, Optional

import streamlit as st
from dotenv import load_dotenv

import openai
from geopy.distance import geodesic
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ──────────────── 1 · Загрузка переменных окружения ────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ──────────────── 2 · Настройка логирования ────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────── 3 · Вспомогательные функции  ─────────────────────
def is_valid_coordinate(lat: float, lon: float) -> bool:
    return -90 <= lat <= 90 and -180 <= lon <= 180


def load_coordinates_from_csv(file) -> List[Tuple[float, float]]:
    """Читает координаты из загруженного CSV-файла Streamlit-uploader."""
    coordinates = []
    text_stream = io.TextIOWrapper(file, encoding="utf-8", newline="")
    sample = text_stream.read(1024)
    text_stream.seek(0)
    dialect = csv.Sniffer().sniff(sample)
    reader = csv.DictReader(text_stream, dialect=dialect)

    for row in reader:
        lat_str = row.get("latitude") or row.get("lat") or row.get("Latitude") or row.get("Lat")
        lon_str = row.get("longitude") or row.get("lon") or row.get("Longitude") or row.get("Lon")
        if lat_str is None or lon_str is None:
            st.warning("CSV-файл должен содержать колонки latitude / longitude.")
            return []
        try:
            lat = float(lat_str)
            lon = float(lon_str)
            if is_valid_coordinate(lat, lon):
                coordinates.append((lat, lon))
            else:
                st.warning(f"Координаты вне диапазона: {lat}, {lon}")
        except ValueError:
            st.warning(f"Некорректный формат координаты: {lat_str}, {lon_str}")
    return coordinates


def compute_distance_matrix(locations: List[Tuple[float, float]]) -> List[List[int]]:
    size = len(locations)
    matrix = []
    for i in range(size):
        row = []
        for j in range(size):
            if i == j:
                row.append(0)
            else:
                row.append(int(geodesic(locations[i], locations[j]).meters))
        row  # noqa: B018
        matrix.append(row)
    return matrix


def optimize_route_with_ortools(coordinates: List[Tuple[float, float]]) -> Optional[List[Tuple[float, float]]]:
    try:
        data = {"locations": coordinates, "num_vehicles": 1, "depot": 0}
        manager = pywrapcp.RoutingIndexManager(len(data["locations"]), data["num_vehicles"], data["depot"])
        routing = pywrapcp.RoutingModel(manager)
        distance_matrix = compute_distance_matrix(data["locations"])

        def distance_callback(from_index, to_index):
            return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

        routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(0)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = 30

        solution = routing.SolveWithParameters(search_parameters)
        if not solution:
            return None

        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(data["locations"][node_index])
            index = solution.Value(routing.NextVar(index))
        route.append(data["locations"][manager.IndexToNode(index)])
        return route
    except Exception as e:
        logger.error(f"Ошибка OR-Tools: {e}", exc_info=True)
        return None


def optimize_route_with_openai(coordinates: List[Tuple[float, float]]) -> Optional[List[Tuple[float, float]]]:
    coord_str = "\n".join(f"{i+1}. {lat},{lon}" for i, (lat, lon) in enumerate(coordinates))
    prompt = (
        "Оптимизируй порядок следующих координат для минимизации общего расстояния маршрута. "
        "Верни только список номеров из исходного списка, разделённых запятыми, без лишнего текста.\n\n"
        f"{coord_str}"
    )
    try:
        rsp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты помощник по оптимизации маршрутов."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=150,
        )
        order_str = rsp.choices[0].message.content.strip()
        if not re.match(r"^(\d+\s*,\s*)*\d+\s*$", order_str):
            return None
        idxs = [int(n) - 1 for n in re.findall(r"\d+", order_str)]
        if len(set(idxs)) != len(coordinates) or any(i >= len(coordinates) for i in idxs):
            return None
        return [coordinates[i] for i in idxs]
    except openai.error.RateLimitError:
        st.warning("Достигнут лимит OpenAI, переключаемся на OR-Tools.")
        return None
    except Exception as e:
        logger.error(f"OpenAI error: {e}", exc_info=True)
        return None


def optimize_route(coordinates: List[Tuple[float, float]]) -> Tuple[Optional[List[Tuple[float, float]]], str]:
    if not coordinates:
        return None, ""
    if len(coordinates) <= 10:
        res = optimize_route_with_openai(coordinates)
        if res:
            return res, "OpenAI GPT-3.5-turbo"
    res = optimize_route_with_ortools(coordinates)
    return res, "OR-Tools" if res else ""


def create_google_maps_links(coordinates: List[Tuple[float, float]]) -> List[str]:
    MAX = 24
    links = []
    for i in range(0, len(coordinates), MAX):
        chunk = coordinates[i : i + MAX]
        route = "/".join(f"{lat},{lon}" for lat, lon in chunk)
        links.append(f"https://www.google.com/maps/dir/{route}")
    return links

# ──────────────── 4 · UI Streamlit ─────────────────────────────────
st.set_page_config(page_title="Оптимизация маршрута", layout="wide")
st.title("🚚 Оптимизация маршрута доставок")

with st.expander("Как пользоваться приложением?"):
    st.markdown(
        """
1. Выберите способ ввода координат:  
   • **Вручную** — введите широту и долготу для каждой точки.  
   • **CSV-файл** — загрузите файл с колонками `latitude` и `longitude`.  
   • **Bulk** — вставьте список точек (`lat,lon`) построчно.  
2. Нажмите **«Оптимизировать маршрут»**.  
3. Получите оптимизированный порядок точек и готовые ссылки для Google Maps.  
"""
    )

method = st.radio(
    "Способ ввода координат",
    ["Вручную", "CSV-файл", "Bulk-ввод"],
    horizontal=True,
)

coordinates: List[Tuple[float, float]] = []

# «Вручную»
if method == "Вручную":
    cols_qty = st.number_input("Количество точек", min_value=1, max_value=25, value=3, step=1)
    st.markdown("Введите координаты:")
    lat_inputs, lon_inputs = [], []
    for i in range(int(cols_qty)):
        c1, c2 = st.columns(2)
        lat = c1.text_input(f"Широта {i+1}", key=f"lat_{i}")
        lon = c2.text_input(f"Долгота {i+1}", key=f"lon_{i}")
        lat_inputs.append(lat)
        lon_inputs.append(lon)
    if st.button("Добавить ещё строку"):
        st.session_state["lat_" + str(cols_qty)] = ""
        st.session_state["lon_" + str(cols_qty)] = ""
        st.experimental_rerun()

    for lat_str, lon_str in zip(lat_inputs, lon_inputs):
        if lat_str and lon_str:
            try:
                lat, lon = float(lat_str), float(lon_str)
                if is_valid_coordinate(lat, lon):
                    coordinates.append((lat, lon))
                else:
                    st.error(f"Координаты вне диапазона: {lat}, {lon}")
            except ValueError:
                st.error(f"Некорректный формат: {lat_str}, {lon_str}")

# «CSV-файл»
elif method == "CSV-файл":
    uploaded_file = st.file_uploader("Загрузите CSV", type=["csv"])
    if uploaded_file:
        coordinates = load_coordinates_from_csv(uploaded_file)

# «Bulk»
else:  # Bulk-ввод
    bulk_text = st.text_area("Вставьте координаты (каждая строка: lat,lon)")
    if bulk_text.strip():
        for line in bulk_text.strip().splitlines():
            parts = line.split(",")
            if len(parts) != 2:
                st.warning(f"Пропускаю строку (не 2 элемента): {line}")
                continue
            try:
                lat, lon = float(parts[0].strip()), float(parts[1].strip())
                if is_valid_coordinate(lat, lon):
                    coordinates.append((lat, lon))
                else:
                    st.warning(f"Вне диапазона: {lat}, {lon}")
            except ValueError:
                st.warning(f"Неверное число в строке: {line}")

# ──────────────── 5 · Запуск оптимизации ───────────────────────────
if st.button("🚀 Оптимизировать маршрут"):
    if not coordinates:
        st.error("Нет корректных координат.")
        st.stop()

    with st.spinner("Считаем оптимальный маршрут…"):
        optimized, tech = optimize_route(coordinates)

    if not optimized:
        st.error("Не удалось оптимизировать маршрут.")
        st.stop()

    # Вывод результатов
    st.success(f"Маршрут оптимизирован ({tech}).")
    st.subheader("Порядок точек:")
    for i, (lat, lon) in enumerate(optimized, 1):
        st.write(f"{i}. {lat}, {lon}")

    st.subheader("Ссылки для Google Maps:")
    for idx, link in enumerate(create_google_maps_links(optimized), 1):
        st.markdown(f"[Маршрут #{idx}]({link})")

# ───────────────────────────────────────────────────────────────────
