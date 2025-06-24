🎯 People Counter - Руководство по установке и использованию
📋 Описание
Система подсчета людей, проходящих через определенную линию в видео, используя YOLOv8 для детекции и DeepSORT для трекинга объектов.
🚀 Возможности

✅ Подсчет входящих и выходящих людей
✅ Интерактивное рисование линии подсчета
✅ Веб-интерфейс для загрузки видео
✅ Поддержка различных форматов видео
✅ Визуализация траекторий движения
✅ Сохранение результатов в JSON
✅ Экспорт обработанного видео

🛠️ Установка
1. Установка Python зависимостей
bash# Создать виртуальное окружение (рекомендуется)
python -m venv people_counter_env
source people_counter_env/bin/activate  # Linux/Mac
# или
people_counter_env\Scripts\activate  # Windows

# Установить зависимости
pip install ultralytics
pip install deep-sort-realtime
pip install opencv-python
pip install torch torchvision
pip install numpy
pip install pathlib
2. Скачать модель YOLO
bash# Модель скачается автоматически при первом запуске
# Или можно скачать заранее:
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
🎮 Использование
Метод 1: Веб-интерфейс (Рекомендуется)

Откройте interface.html в браузере
Загрузите видео файл
Нарисуйте линию подсчета на видео
Скопируйте сгенерированную команду
Запустите команду в терминале

Метод 2: Интерактивный режим
bash# Запуск с интерактивным выбором линии
python people_counter.py --source video.mp4 --interactive

# Откроется окно, где можно кликами выбрать линию
Метод 3: Командная строка с предустановленной линией
bash# Базовое использование
python people_counter.py --source video.mp4 --line 0.3,0.5,0.7,0.5

# С сохранением результата
python people_counter.py --source video.mp4 --line 0.3,0.5,0.7,0.5 --save

# Без отображения окна (для серверов)
python people_counter.py --source video.mp4 --line 0.3,0.5,0.7,0.5 --no-display
📊 Параметры командной строки
bash--source        # Источник видео (файл, веб-камера, RTSP)
--model         # Путь к модели YOLO (по умолчанию: yolov8n.pt)
--line          # Координаты линии (x1,y1,x2,y2) в относительных координатах [0-1]
--classes       # ID классов для детекции (по умолчанию: 0 - люди)
--save          # Сохранить обработанное видео
--display       # Показать окно с видео (по умолчанию: True)
--interactive   # Интерактивный выбор линии
--conf          # Порог уверенности для детекции (по умолчанию: 0.35)
--output-dir    # Папка для сохранения результатов
🎯 Как работает система
1. Координаты линии

Используются относительные координаты от 0 до 1
Формат: x1,y1,x2,y2
Пример: 0.3,0.5,0.7,0.5 - горизонтальная линия посередине

2. Направления подсчета

ВХОД: движение слева направо через линию
ВЫХОД: движение справа налево через линию
Направление определяется автоматически на основе геометрии линии

3. Алгоритм работы

YOLOv8 детектирует людей в каждом кадре
DeepSORT отслеживает каждого человека (присваивает ID)
Система отслеживает пересечение линии каждым треком
Подсчитывает входы и выходы

📁 Структура файлов
people_counter/
├── people_counter.py      # Основной скрипт
├── interface.html         # Веб-интерфейс
├── README.md             # Это руководство
├── requirements.txt      # Зависимости
└── results/              # Папка с результатами
    ├── result.mp4        # Обработанное видео
    ├── results_*.json    # Статистика
    └── screenshot_*.jpg  # Скриншоты
🔧 Примеры использования
Веб-камера в реальном времени
bashpython people_counter.py --source 0 --interactive
RTSP поток
bashpython people_counter.py --source "rtsp://username:password@ip:port/stream" --line 0.4,0.6,0.6,0.4
Пакетная обработка
bash# Обработать все видео в папке
for video in *.mp4; do
    python people_counter.py --source "$video" --line 0.3,0.5,0.7,0.5 --save
done
Высокоточная детекция
bash# Использовать более точную модель
python people_counter.py --source video.mp4 --model yolov8x.pt --conf 0.5
📊 Интерпретация результатов
В реальном времени

Зеленый счетчик: ВХОД - люди, прошедшие слева направо
Желтый счетчик: ВЫХОД - люди, прошедшие справа налево
Синий счетчик: ИТОГО - разница между входом и выходом

JSON файл результатов
json{
  "in_count": 15,
  "out_count": 12,
  "net_count": 3,
  "total_detections": 450,
  "elapsed_time": 120.5,
  "frames_processed": 3600,
  "fps": 29.87,
  "timestamp": "2024-01-15T14:30:00",
  "line_coordinates": [150, 200, 450, 300]
}
🔍 Устранение неполадок
Проблема: Модель не загружается
bash# Решение: Установить правильную версию PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
Проблема: Низкая точность детекции
bash# Решение: Использовать более точную модель и настроить порог
python people_counter.py --model yolov8m.pt --conf 0.6
Проблема: Двойной подсчет

Убедитесь, что линия пересекает зону движения только один раз
Проверьте, что линия не слишком близко к краям видео

Проблема: Пропущенные детекции

Уменьшите порог уверенности: --conf 0.25
Используйте более мощную модель: --model yolov8l.pt

⚡ Оптимизация производительности
Для слабых компьютеров
bash# Использовать самую легкую модель
python people_counter.py --model yolov8n.pt --conf 0.4
Для высокой точности
bash# Использовать самую точную модель
python people_counter.py --model yolov8x.pt --conf 0.6
Для серверов без GUI
bash# Отключить отображение
python people_counter.py --source video.mp4 --no-display --save
🎨 Настройка визуализации
Горячие клавиши в режиме отображения

Q или ESC - Выход
S - Сохранить скриншот
Space - Пауза/продолжить (если поддерживается)

Цветовая схема

🟢 Зеленый: Вход, начальная точка линии
🔴 Красный: Выход, конечная точка линии
🔵 Синий: Центры треков
🟡 Желтый: ID треков, стрелка направления

📈 Рекомендации по настройке
Размещение линии

Линия должна быть перпендикулярна направлению движения
Избегайте размещения на краях видео
Учитывайте перспективу камеры

Настройки детекции

Для переполненных сцен: увеличьте --conf
Для разреженных сцен: уменьшите --conf
Для быстрых объектов: используйте более частые кадры

Оптимальные условия

Хорошее освещение
Минимальные окклюзии (перекрытия)
Стабильная камера
Четкое разделение направлений движения

🤝 Поддержка
При возникновении проблем:

Проверьте версии зависимостей
Убедитесь в корректности формата видео
Проверьте права доступа к файлам
Протестируйте на простом видео

📝 Changelog
v2.0

✅ Добавлен веб-интерфейс
✅ Интерактивный выбор линии
✅ Улучшенная визуализация
✅ Сохранение статистики в JSON
✅ Поддержка горячих клавиш

v1.0

✅ Базовый функционал подсчета
✅ Интеграция YOLOv8 + DeepSORT









# People IN/OUT Counter

Real-time people counting system that tracks individuals crossing a user-defined line in video streams using YOLOv8 and DeepSORT.

## Quick Start

1. **Setup environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt

Run examples:
bash# Count people in video file with GUI
python main.py --source videos/people.mp4 --display

# Use webcam and save output video
python main.py --source 0 --save

# RTSP stream without display (headless)
python main.py --source rtsp://camera-ip/stream --save