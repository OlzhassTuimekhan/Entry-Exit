#!/usr/bin/env python3
"""
People IN/OUT Counter using YOLOv8 + DeepSORT
Counts people crossing a user-defined line in video/RTSP streams.
Enhanced version with better interface support.
"""

import argparse
import sys
import os
from pathlib import Path
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import json
from datetime import datetime


def apply_torch_patch():
    """Apply PyTorch 2.6 compatibility patch for YOLOv8 weights loading."""
    original_load = torch.load

    def patched_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_load(*args, **kwargs)

    torch.load = patched_load

    try:
        from ultralytics.nn.modules import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
    except ImportError:
        pass


class LineCounter:
    """Manages people counting across a defined line."""

    def __init__(self, line_coords, frame_shape):
        """
        Initialize line counter.

        Args:
            line_coords: (x1, y1, x2, y2) in relative coordinates [0-1]
            frame_shape: (height, width) of video frame
        """
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = line_coords

        # Convert relative to absolute coordinates
        self.line = (
            int(x1 * w), int(y1 * h),
            int(x2 * w), int(y2 * h)
        )

        self.in_count = 0
        self.out_count = 0
        self.track_positions = {}  # track_id -> last_position
        self.track_history = {}  # track_id -> list of positions
        self.crossed_tracks = set()  # tracks that have crossed the line

        # Statistics
        self.start_time = time.time()
        self.total_detections = 0
        self.frame_count = 0

    def update(self, track_id, center):
        """
        Update counter based on track movement.

        Args:
            track_id: Unique tracker ID
            center: (x, y) center point of detection
        """
        self.total_detections += 1

        # Store track history
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        self.track_history[track_id].append(center)

        # Keep only last 10 positions to prevent memory issues
        if len(self.track_history[track_id]) > 10:
            self.track_history[track_id] = self.track_history[track_id][-10:]

        if track_id in self.track_positions:
            prev_pos = self.track_positions[track_id]
            curr_pos = center

            # Check if track crossed the line
            if self._line_crossed(prev_pos, curr_pos) and track_id not in self.crossed_tracks:
                # Determine direction based on line orientation
                if self._is_going_in(prev_pos, curr_pos):
                    self.in_count += 1
                    print(f"  🟢 ВХОД: ID {track_id} | Всего входов: {self.in_count}")
                else:
                    self.out_count += 1
                    print(f"  🔴 ВЫХОД: ID {track_id} | Всего выходов: {self.out_count}")

                # Mark this track as having crossed to prevent double counting
                self.crossed_tracks.add(track_id)

        self.track_positions[track_id] = center

    def _line_crossed(self, p1, p2):
        """Check if line segment p1-p2 intersects counting line."""
        x1, y1, x2, y2 = self.line

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        line_p1, line_p2 = (x1, y1), (x2, y2)
        return ccw(line_p1, p1, p2) != ccw(line_p2, p1, p2) and \
            ccw(p1, line_p1, line_p2) != ccw(p2, line_p1, line_p2)

    def _is_going_in(self, prev_pos, curr_pos):
        """Determine if movement is 'IN' direction (left to right across line)."""
        x1, y1, x2, y2 = self.line

        def side_of_line(point):
            return (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)

        prev_side = side_of_line(prev_pos)
        curr_side = side_of_line(curr_pos)

        return prev_side < 0 and curr_side > 0

    def draw_line(self, frame):
        """Draw counting line and counters on frame."""
        x1, y1, x2, y2 = self.line
        frame_height, frame_width = frame.shape[:2]

        # Draw thick counting line in bright red
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 6)

        # Draw line endpoints
        cv2.circle(frame, (x1, y1), 10, (0, 255, 0), -1)  # Green start
        cv2.circle(frame, (x2, y2), 10, (0, 0, 255), -1)  # Red end

        # Draw direction arrows
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Calculate line direction
        line_angle = np.arctan2(y2 - y1, x2 - x1)

        # IN arrow (perpendicular to line, pointing "up")
        arrow_length = 40
        in_arrow_angle = line_angle - np.pi / 2
        in_end_x = int(mid_x + arrow_length * np.cos(in_arrow_angle))
        in_end_y = int(mid_y + arrow_length * np.sin(in_arrow_angle))

        cv2.arrowedLine(frame, (mid_x, mid_y), (in_end_x, in_end_y), (0, 255, 0), 4)
        cv2.putText(frame, "ВХОД", (in_end_x + 10, in_end_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # OUT arrow (perpendicular to line, pointing "down")
        out_arrow_angle = line_angle + np.pi / 2
        out_end_x = int(mid_x + arrow_length * np.cos(out_arrow_angle))
        out_end_y = int(mid_y + arrow_length * np.sin(out_arrow_angle))

        cv2.arrowedLine(frame, (mid_x, mid_y), (out_end_x, out_end_y), (0, 255, 255), 4)
        cv2.putText(frame, "ВЫХОД", (out_end_x + 10, out_end_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Enhanced counter display
        counter_width = 400
        counter_height = 120

        # Background with transparency effect
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + counter_width, 10 + counter_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Border
        cv2.rectangle(frame, (10, 10), (10 + counter_width, 10 + counter_height), (255, 255, 255), 3)

        # Counter text with larger font
        cv2.putText(frame, f"ВХОД: {self.in_count}", (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"ВЫХОД: {self.out_count}", (25, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        # Net count
        net_count = self.in_count - self.out_count
        net_color = (0, 255, 0) if net_count >= 0 else (0, 0, 255)
        cv2.putText(frame, f"ИТОГО: {net_count}", (220, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, net_color, 3)

        # Statistics in bottom right
        stats_text = [
            f"Время: {time.time() - self.start_time:.0f}s",
            f"Кадров: {self.frame_count}",
            f"Детекций: {self.total_detections}"
        ]

        stats_y = frame_height - 80
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (frame_width - 200, stats_y + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Line coordinates info
        cv2.putText(frame, f"Линия: ({x1},{y1})-({x2},{y2})",
                    (10, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_track_trails(self, frame):
        """Draw track trails to visualize movement."""
        for track_id, positions in self.track_history.items():
            if len(positions) > 1:
                # Draw trail
                points = np.array(positions, dtype=np.int32)
                cv2.polylines(frame, [points], False, (100, 100, 255), 2)

                # Draw track ID at current position
                if positions:
                    current_pos = positions[-1]
                    cv2.putText(frame, f"{track_id}",
                                (current_pos[0] - 10, current_pos[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    def get_statistics(self):
        """Get counting statistics."""
        elapsed_time = time.time() - self.start_time
        return {
            'in_count': self.in_count,
            'out_count': self.out_count,
            'net_count': self.in_count - self.out_count,
            'total_detections': self.total_detections,
            'elapsed_time': elapsed_time,
            'frames_processed': self.frame_count,
            'fps': self.frame_count / elapsed_time if elapsed_time > 0 else 0
        }

    def save_results(self, output_file="counting_results.json"):
        """Save counting results to JSON file."""
        stats = self.get_statistics()
        stats['timestamp'] = datetime.now().isoformat()
        stats['line_coordinates'] = self.line

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"📊 Результаты сохранены в {output_file}")


def parse_line_coords(line_str):
    """Parse line coordinates from string format 'x1,y1,x2,y2'."""
    try:
        coords = [float(x) for x in line_str.split(',')]
        if len(coords) != 4:
            raise ValueError
        # Validate coordinates are in range [0,1]
        for coord in coords:
            if not (0 <= coord <= 1):
                raise ValueError("Coordinates must be between 0 and 1")
        return coords
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Line must be in format 'x1,y1,x2,y2' with values 0-1"
        )


def create_interactive_line_selector(video_path):
    """Create interactive window for line selection."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Не удалось открыть видео: {video_path}")
        return None

    # Get first frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Не удалось прочитать первый кадр")
        return None

    # Global variables for mouse callback
    points = []
    temp_frame = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal points, temp_frame

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
                points.append((x, y))
                temp_frame = frame.copy()

                # Draw points
                for i, point in enumerate(points):
                    color = (0, 255, 0) if i == 0 else (0, 0, 255)
                    cv2.circle(temp_frame, point, 8, color, -1)
                    label = "НАЧАЛО" if i == 0 else "КОНЕЦ"
                    cv2.putText(temp_frame, label, (point[0] + 10, point[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Draw line if we have 2 points
                if len(points) == 2:
                    cv2.line(temp_frame, points[0], points[1], (0, 0, 255), 3)
                    cv2.putText(temp_frame, "Нажмите ENTER для подтверждения или R для сброса",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow('Выберите линию подсчета', temp_frame)

    cv2.namedWindow('Выберите линию подсчета', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Выберите линию подсчета', 1000, 700)
    cv2.setMouseCallback('Выберите линию подсчета', mouse_callback)

    # Instructions
    instructions = frame.copy()
    cv2.putText(instructions, "1. Кликните для установки НАЧАЛЬНОЙ точки линии",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(instructions, "2. Кликните для установки КОНЕЧНОЙ точки линии",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(instructions, "3. Нажмите ENTER для подтверждения",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(instructions, "4. Нажмите R для сброса, ESC для выхода",
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Выберите линию подсчета', instructions)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == 13 or key == 10:  # Enter
            if len(points) == 2:
                break
        elif key == ord('r') or key == ord('R'):  # Reset
            points = []
            temp_frame = frame.copy()
            cv2.imshow('Выберите линию подсчета', temp_frame)
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()

    if len(points) == 2:
        h, w = frame.shape[:2]
        # Convert to relative coordinates
        x1, y1 = points[0][0] / w, points[0][1] / h
        x2, y2 = points[1][0] / w, points[1][1] / h
        return [x1, y1, x2, y2]

    return None


def main():
    """Main function to run people counter."""
    # Apply PyTorch compatibility patch
    apply_torch_patch()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='People IN/OUT Counter with Interactive Line Selection')
    parser.add_argument('--source', default='0',
                        help='Video source: file path, webcam (0), or RTSP URL')
    parser.add_argument('--model', default='yolov8n.pt',
                        help='YOLO model path')
    parser.add_argument('--line', type=parse_line_coords,
                        help='Counting line coords: x1,y1,x2,y2 (0-1 relative). If not provided, interactive selection will be used.')
    parser.add_argument('--classes', default='0',
                        help='Class IDs to detect (comma-separated)')
    parser.add_argument('--save', action='store_true',
                        help='Save output video as result.mp4')
    parser.add_argument('--display', action='store_true', default=True,
                        help='Display video window')
    parser.add_argument('--interactive', action='store_true',
                        help='Force interactive line selection')
    parser.add_argument('--conf', type=float, default=0.35,
                        help='Confidence threshold for detection')
    parser.add_argument('--output-dir', default='.',
                        help='Output directory for results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("🎯 People Counter - Система подсчета людей")
    print("=" * 50)

    # Interactive line selection if no line provided or forced
    if args.line is None or args.interactive:
        print("🖱️  Запуск интерактивного выбора линии...")
        line_coords = create_interactive_line_selector(args.source)
        if line_coords is None:
            print("❌ Линия не была выбрана. Выход.")
            return 1
        args.line = line_coords
        print(f"✅ Линия выбрана: {args.line}")

    # Parse class IDs
    class_ids = [int(x) for x in args.classes.split(',')]

    print(f"🤖 Загрузка YOLO модели: {args.model}")
    try:
        model = YOLO(args.model)
        print("✅ YOLO модель загружена успешно")
    except Exception as e:
        print(f"❌ Ошибка загрузки YOLO модели: {e}")
        return 1

    # Initialize DeepSORT tracker
    print("🔄 Инициализация DeepSORT трекера...")
    tracker = DeepSort(max_age=50, n_init=3)

    # Open video source
    print(f"📹 Открытие видео источника: {args.source}")
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print(f"❌ Не удалось открыть видео источник: {args.source}")
        return 1

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"✅ Видео открыто: {width}x{height} @ {fps} FPS")
    if total_frames > 0:
        print(f"📊 Общее количество кадров: {total_frames}")

    # Initialize line counter
    counter = LineCounter(args.line, (height, width))

    # Setup video writer if saving
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = output_dir / 'result.mp4'
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"💾 Сохранение результата в: {output_path}")

    # Setup display window
    if args.display:
        cv2.namedWindow('People Counter', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('People Counter', 1200, 800)
        print("🖥️  Окно отображения создано")

    print("\n▶️  Начало обработки видео... Нажмите 'q' для выхода")
    print("-" * 50)

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n📹 Достигнут конец видео")
                break

            frame_count += 1
            counter.frame_count = frame_count

            # Run YOLO detection
            results = model(frame, classes=class_ids, conf=args.conf, verbose=False)

            # Extract detections for DeepSORT
            detections = []
            boxes = results[0].boxes

            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()

                    # Convert to DeepSORT format: [x, y, w, h]
                    w, h = x2 - x1, y2 - y1
                    detections.append([[x1, y1, w, h], conf, 'person'])

            # Update tracker
            tracks = tracker.update_tracks(detections, frame=frame)

            # Process tracks and update counter
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = ltrb

                # Calculate center point
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                # Update counter
                counter.update(track_id, center)

                # Draw bounding box with enhanced styling
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 3)

                # Draw ID with background
                label = f'ID: {track_id}'
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (int(x1), int(y1) - text_height - 10),
                              (int(x1) + text_width + 10, int(y1)), (0, 255, 0), -1)
                cv2.putText(frame, label, (int(x1) + 5, int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                # Draw center point
                cv2.circle(frame, center, 6, (255, 0, 0), -1)

            # Draw track trails
            counter.draw_track_trails(frame)

            # Draw counting line and counters
            counter.draw_line(frame)

            # Save frame if requested
            if writer:
                writer.write(frame)

            # Display frame
            if args.display:
                cv2.imshow('People Counter', frame)

                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("\n⏹️  Остановлено пользователем")
                    break
                elif key == ord('s'):  # Save screenshot
                    screenshot_path = output_dir / f'screenshot_{frame_count}.jpg'
                    cv2.imwrite(str(screenshot_path), frame)
                    print(f"📸 Скриншот сохранен: {screenshot_path}")

            # Print progress
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0

                print(f"📊 Кадр {frame_count:,} | ВХОД: {counter.in_count} | ВЫХОД: {counter.out_count} | "
                      f"FPS: {fps_current:.1f} | Прогресс: {progress:.1f}%")

    except KeyboardInterrupt:
        print("\n⏹️  Остановлено пользователем (Ctrl+C)")

    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

        # Final statistics
        stats = counter.get_statistics()
        print("\n" + "=" * 60)
        print("🎯 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
        print("=" * 60)
        print(f"✅ ВХОД: {stats['in_count']}")
        print(f"❌ ВЫХОД: {stats['out_count']}")
        print(f"📊 ИТОГО: {stats['net_count']}")
        print(f"⏱️  Время обработки: {stats['elapsed_time']:.1f} секунд")
        print(f"🎬 Обработано кадров: {stats['frames_processed']:,}")
        print(f"⚡ Средний FPS: {stats['fps']:.2f}")
        print(f"🔍 Всего детекций: {stats['total_detections']:,}")

        # Save results
        results_file = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        counter.save_results(results_file)

    return 0


if __name__ == '__main__':
    sys.exit(main())