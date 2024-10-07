// static/js/scripts.js

// Скрипты для динамического отображения полей ввода и управления 'required'
document.addEventListener('DOMContentLoaded', function () {
    const methodRadios = document.querySelectorAll('input[name="method"]');
    const manualInput = document.getElementById('manual-input');
    const csvInput = document.getElementById('csv-input');
    const bulkInput = document.getElementById('bulk-input');

    methodRadios.forEach(radio => {
        radio.addEventListener('change', function () {
            if (this.value === 'manual') {
                manualInput.style.display = 'block';
                csvInput.style.display = 'none';
                bulkInput.style.display = 'none';
                setRequired('manual');
            } else if (this.value === 'csv') {
                manualInput.style.display = 'none';
                csvInput.style.display = 'block';
                bulkInput.style.display = 'none';
                setRequired('csv');
            } else if (this.value === 'bulk') {
                manualInput.style.display = 'none';
                csvInput.style.display = 'none';
                bulkInput.style.display = 'block';
                setRequired('bulk');
            }
        });
    });

    // Добавление новых полей для координат
    const addCoordinateBtn = document.querySelector('.add-coordinate');
    const coordinateFields = document.getElementById('coordinate-fields');

    addCoordinateBtn.addEventListener('click', function () {
        const newField = document.createElement('div');
        newField.className = 'row mb-2';
        newField.innerHTML = `
            <div class="col">
                <input type="text" name="latitude" class="form-control" placeholder="Широта" required>
            </div>
            <div class="col">
                <input type="text" name="longitude" class="form-control" placeholder="Долгота" required>
            </div>
            <div class="col-auto">
                <button type="button" class="btn btn-danger remove-coordinate">-</button>
            </div>
        `;
        coordinateFields.appendChild(newField);
    });

    // Удаление полей для координат
    coordinateFields.addEventListener('click', function (e) {
        if (e.target && e.target.classList.contains('remove-coordinate')) {
            const removed = e.target.closest('.row').querySelectorAll('input');
            removed.forEach(input => input.required = false);
            e.target.closest('.row').remove();
        }
    });

    // Функция для установки 'required' атрибутов
    function setRequired(method) {
        // Удаляем 'required' с всех
        document.querySelectorAll('input[name="latitude"], input[name="longitude"], textarea[name="bulk_coordinates"], input[name="csv_file"]').forEach(element => {
            element.required = false;
        });

        if (method === 'manual') {
            document.querySelectorAll('input[name="latitude"], input[name="longitude"]').forEach(element => {
                element.required = true;
            });
        } else if (method === 'csv') {
            // Только файл CSV должен быть обязательным
            const csvFileInput = document.querySelector('input[name="csv_file"]');
            csvFileInput.required = true;
        } else if (method === 'bulk') {
            const bulkInput = document.querySelector('textarea[name="bulk_coordinates"]');
            bulkInput.required = true;
        }
    }

    // Инициализация 'required' для начального метода
    setRequired('manual');

    // Валидация формы перед отправкой
    const form = document.getElementById('route-form');
    form.addEventListener('submit', function (e) {
        // Проверка выбранного метода
        const method = document.querySelector('input[name="method"]:checked').value;
        let valid = true;

        if (method === 'manual') {
            const latitudes = document.querySelectorAll('input[name="latitude"]');
            const longitudes = document.querySelectorAll('input[name="longitude"]');
            latitudes.forEach((lat, index) => {
                const lon = longitudes[index];
                if (!isValidCoordinate(lat.value, lon.value)) {
                    alert(`Некорректные координаты в строке ${index + 1}. Широта должна быть от -90 до 90, долгота от -180 до 180.`);
                    valid = false;
                }
            });
        } else if (method === 'bulk') {
            const bulkInput = document.querySelector('textarea[name="bulk_coordinates"]').value.trim();
            if (bulkInput.length === 0) {
                alert('Пожалуйста, введите координаты или выберите другой метод ввода.');
                valid = false;
            } else {
                const parts = bulkInput.split(' ');
                parts.forEach((part, index) => {
                    const [lat, lon] = part.split(',');
                    if (!isValidCoordinate(lat, lon)) {
                        alert(`Некорректные координаты в позиции ${index + 1}. Широта должна быть от -90 до 90, долгота от -180 до 180.`);
                        valid = false;
                    }
                });
            }
        }

        if (!valid) {
            e.preventDefault(); // Отменяет отправку формы
        }
    });

    // Функция проверки корректности координаты
    function isValidCoordinate(lat, lon) {
        const latNum = parseFloat(lat);
        const lonNum = parseFloat(lon);
        return !isNaN(latNum) && !isNaN(lonNum) && latNum >= -90 && latNum <= 90 && lonNum >= -180 && lonNum <= 180;
    }
});
