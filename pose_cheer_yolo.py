import cv2
from ultralytics import YOLO
import socket
import time
import numpy as np

MODEL_PATH = "yolo26n-pose.pt"
UDP_HOST = "0.0.0.0"
UDP_RECEIVE_PORT = 5005
UDP_SEND_PORT = 5006
UDP_SEND_HOST = "127.0.0.1"
UDP_BUFFER_SIZE = 1024
UDP_CONNECTED_TIMEOUT = 5.0
MAX_PEOPLE = 3
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
DEFAULT_LANDMARK_COLOR = (0, 0, 255)
DEFAULT_CONNECTION_COLOR = (224, 224, 224)
MIN_CONFIDENCE = 0.5
ABOVE_HEAD_MARGIN = 0.04
ACTIVE_TRACKING_DISTANCE = 0.4
LOST_TRACKING_DISTANCE = 0.7
MAX_MISSING_SECONDS = 8.0
MAX_MISSING_SECONDS_AFTER_COUNT = 20.0
ACTIVE_TRACKING_TIME_THRESHOLD = 0.5
DRAW_SKELETON = True

# YOLO keypoint indices
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def is_visible(keypoint_conf):
    return keypoint_conf >= MIN_CONFIDENCE


def is_cheering_pose(keypoints):
    """
    Detecta se a pose é de torcida:
    - Punhos acima da cabeça
    - Cotovelos levantados acima dos ombros
    """
    if len(keypoints) < 17:
        return False

    nose_conf = keypoints[NOSE, 2]
    left_wrist_conf = keypoints[LEFT_WRIST, 2]
    right_wrist_conf = keypoints[RIGHT_WRIST, 2]
    left_elbow_conf = keypoints[LEFT_ELBOW, 2]
    right_elbow_conf = keypoints[RIGHT_ELBOW, 2]
    left_shoulder_conf = keypoints[LEFT_SHOULDER, 2]
    right_shoulder_conf = keypoints[RIGHT_SHOULDER, 2]

    important_conf = (
        nose_conf,
        left_wrist_conf,
        right_wrist_conf,
        left_elbow_conf,
        right_elbow_conf,
        left_shoulder_conf,
        right_shoulder_conf,
    )

    if not all(is_visible(conf) for conf in important_conf):
        return False

    nose_y = keypoints[NOSE, 1]
    left_wrist_y = keypoints[LEFT_WRIST, 1]
    right_wrist_y = keypoints[RIGHT_WRIST, 1]
    left_elbow_y = keypoints[LEFT_ELBOW, 1]
    right_elbow_y = keypoints[RIGHT_ELBOW, 1]
    left_shoulder_y = keypoints[LEFT_SHOULDER, 1]
    right_shoulder_y = keypoints[RIGHT_SHOULDER, 1]

    wrists_above_head = (
        left_wrist_y < nose_y - ABOVE_HEAD_MARGIN
        and right_wrist_y < nose_y - ABOVE_HEAD_MARGIN
    )
    elbows_lifted = (
        left_elbow_y < left_shoulder_y
        and right_elbow_y < right_shoulder_y
    )

    return wrists_above_head and elbows_lifted


def pose_center(keypoints):
    """Calcula o centro da pose baseado nos ombros e quadris visíveis"""
    center_indices = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
    visible_points = [
        keypoints[i, :2] for i in center_indices
        if is_visible(keypoints[i, 2])
    ]

    if not visible_points:
        return None

    center = np.mean(visible_points, axis=0)
    return tuple(center)


def pose_size(keypoints):
    """Calcula o tamanho da pose (bounding box diagonal)"""
    visible_points = [
        keypoints[i, :2] for i in range(len(keypoints))
        if is_visible(keypoints[i, 2])
    ]

    if not visible_points:
        return 0.0

    visible_points = np.array(visible_points)
    min_x = visible_points[:, 0].min()
    max_x = visible_points[:, 0].max()
    min_y = visible_points[:, 1].min()
    max_y = visible_points[:, 1].max()

    return max(max_x - min_x, max_y - min_y)


def distance(point_a, point_b):
    return ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5


def matching_score(center, size, person):
    """Calcula score de correspondência entre detecção e pessoa rastreada.
    Menor score = melhor correspondência.
    """
    center_distance = distance(center, person["center"])
    
    # Penalidade de tamanho relativa (não absoluta)
    if person["size"] > 0:
        size_ratio = abs(size - person["size"]) / person["size"]
    else:
        size_ratio = 0
    
    # Dar mais peso à distância do centro, menos peso ao tamanho
    score = center_distance * 0.8 + size_ratio * 0.2
    return score


def track_people(detected_keypoints, tracked_people, next_person_id, now):
    for person in tracked_people:
        person["visible"] = False

    current_people = []
    available_people = set(range(len(tracked_people)))
    
    # Ordena detectadas por confiança (se disponível) ou mantém ordem
    detected_list = list(detected_keypoints[:MAX_PEOPLE])

    for keypoints in detected_list:
        center = pose_center(keypoints)
        if center is None:
            continue

        size = pose_size(keypoints)
        best_index = None
        best_score = float('inf')
        
        # Threshold dinâmico baseado no tempo desde a última detecção
        for index in available_people:
            person = tracked_people[index]
            missing_time = now - person["last_seen"]
            
            # Usar limites diferentes baseado no tempo de ausência
            if missing_time < ACTIVE_TRACKING_TIME_THRESHOLD:
                max_distance = ACTIVE_TRACKING_DISTANCE
            else:
                max_distance = LOST_TRACKING_DISTANCE

            center_distance = distance(center, person["center"])
            
            # Se muito longe, não considerar
            if center_distance > max_distance:
                continue

            current_score = matching_score(center, size, person)
            
            # Preferir pessoas que estão sendo rastreadas (não perdidas)
            if missing_time > 1.0:
                current_score *= 1.5  # Penalidade para pessoas perdidas
            
            if current_score < best_score:
                best_index = index
                best_score = current_score

        if best_index is None:
            # Criar nova pessoa
            person = {
                "id": next_person_id,
                "center": center,
                "size": size,
                "counted": False,
                "last_seen": now,
                "visible": True,
            }
            tracked_people.append(person)
            next_person_id += 1
        else:
            # Atualizar pessoa existente
            person = tracked_people[best_index]
            person["center"] = center
            person["size"] = size
            person["last_seen"] = now
            person["visible"] = True
            available_people.remove(best_index)

        current_people.append((keypoints, person))

    tracked_people[:] = [
        person
        for person in tracked_people
        if now - person["last_seen"]
        <= (
            MAX_MISSING_SECONDS_AFTER_COUNT
            if person["counted"]
            else MAX_MISSING_SECONDS
        )
    ]

    return current_people, next_person_id


def draw_pose(frame, keypoints, person, is_cheering, debug_labels):
    """Desenha a pose no frame"""
    if not DRAW_SKELETON and not debug_labels:
        return

    # Cores baseadas no estado de torcida
    landmark_color = GREEN if is_cheering else DEFAULT_LANDMARK_COLOR
    connection_color = GREEN if is_cheering else DEFAULT_CONNECTION_COLOR

    h, w = frame.shape[:2]

    if DRAW_SKELETON:
        # Desenhar keypoints
        for i, (x, y, conf) in enumerate(keypoints):
            if conf >= MIN_CONFIDENCE:
                x_px = int(x * w)
                y_px = int(y * h)
                cv2.circle(frame, (x_px, y_px), 5, landmark_color, -1)
                cv2.circle(frame, (x_px, y_px), 5, (255, 255, 255), 1)

    # Conexões principais do corpo (COCO format)
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # cabeça
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # braços
        (5, 11), (6, 12), (11, 12),  # tronco
        (11, 13), (13, 15), (12, 14), (14, 16),  # pernas
    ]

    for start_idx, end_idx in connections:
        if (DRAW_SKELETON and
                keypoints[start_idx, 2] >= MIN_CONFIDENCE and
                keypoints[end_idx, 2] >= MIN_CONFIDENCE):
            start_pos = (int(keypoints[start_idx, 0] * w), int(keypoints[start_idx, 1] * h))
            end_pos = (int(keypoints[end_idx, 0] * w), int(keypoints[end_idx, 1] * h))
            cv2.line(frame, start_pos, end_pos, connection_color, 2)

    # Label de debug
    if debug_labels:
        label_x = int(person["center"][0] * w)
        label_y = int(person["center"][1] * h)
        label = f"Pessoa {person['id']}: {1 if person['counted'] else 0}"
        cv2.putText(
            frame,
            label,
            (max(10, label_x - 60), max(30, label_y - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            GREEN if person["counted"] else WHITE,
            2,
            cv2.LINE_AA,
        )


def draw_counter(frame, cheer_count):
    cv2.putText(
        frame,
        f"Torcedores contados: {cheer_count}  |  D: debug  R: reset  Esc: sair",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        WHITE,
        2,
        cv2.LINE_AA,
    )


def draw_last_udp_message(frame, message, connected):
    height = frame.shape[0]
    cv2.putText(
        frame,
        f"UDP: {message}",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        GREEN if connected else RED,
        2,
        cv2.LINE_AA,
    )


def create_udp_sockets():
    receive_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receive_socket.bind((UDP_HOST, UDP_RECEIVE_PORT))
    receive_socket.setblocking(False)

    send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"UDP recebendo em {UDP_HOST}:{UDP_RECEIVE_PORT}")
    print(f"UDP enviando para {UDP_SEND_HOST}:{UDP_SEND_PORT}")
    return receive_socket, send_socket


def reset_values(state):
    state["tracked_people"] = []
    state["next_person_id"] = 1
    state["cheer_count"] = 0


def handle_udp_messages(receive_socket, send_socket, callbacks, state):
    while True:
        try:
            data, address = receive_socket.recvfrom(UDP_BUFFER_SIZE)
        except BlockingIOError:
            break

        message = data.decode("utf-8", errors="ignore").strip().lower()
        state["last_udp_message"] = message
        state["last_udp_time"] = time.monotonic()
        print(f"UDP recebido de {address[0]}:{address[1]} -> {message}")
        callback = callbacks.get(message)
        if callback is None:
            continue

        response = callback()
        if response is not None:
            target_address = (UDP_SEND_HOST, UDP_SEND_PORT)
            send_socket.sendto(response.encode("utf-8"), target_address)
            print(f"UDP enviado para {target_address[0]}:{target_address[1]} -> {response}")


# Inicialização
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
udp_receive_socket, udp_send_socket = create_udp_sockets()

# Carregar modelo YOLO
model = YOLO(MODEL_PATH)

state = {
    "tracked_people": [],
    "next_person_id": 1,
    "cheer_count": 0,
    "last_udp_message": "sem mensagem",
    "last_udp_time": None,
}
udp_callbacks = {
    "reset": lambda: reset_values(state),
    "values": lambda: f"cheer,{min(state['cheer_count'], 3)}",
}
debug_labels = False

while cap.isOpened():
    handle_udp_messages(
        udp_receive_socket,
        udp_send_socket,
        udp_callbacks,
        state,
    )

    ret, frame = cap.read()
    if not ret:
        break

    # Detecção de poses com YOLO
    results = model(frame, conf=MIN_CONFIDENCE, verbose=False)
    now = time.monotonic()

    # Extrair keypoints normalizados
    detected_keypoints = []
    if results[0].keypoints is not None:
        for keypoints in results[0].keypoints.data:
            # Normalizar para [0, 1]
            h, w = frame.shape[:2]
            keypoints_normalized = keypoints.cpu().numpy().copy()
            keypoints_normalized[:, 0] /= w
            keypoints_normalized[:, 1] /= h
            detected_keypoints.append(keypoints_normalized)

    current_people, state["next_person_id"] = track_people(
        detected_keypoints,
        state["tracked_people"],
        state["next_person_id"],
        now,
    )

    for keypoints, person in current_people:
        is_cheering = is_cheering_pose(keypoints)
        if is_cheering and not person["counted"]:
            person["counted"] = True
            state["cheer_count"] += 1

        draw_pose(frame, keypoints, person, is_cheering, debug_labels)

    if debug_labels:
        udp_connected = (
            state["last_udp_time"] is not None
            and time.monotonic() - state["last_udp_time"] <= UDP_CONNECTED_TIMEOUT
        )
        draw_counter(frame, state["cheer_count"])
        draw_last_udp_message(
            frame,
            state["last_udp_message"],
            udp_connected,
        )

    cv2.imshow("YOLO Pose Detection", frame)
    key = cv2.waitKey(5) & 0xFF
    if key == 27:
        break
    if key in (ord("d"), ord("D")):
        debug_labels = not debug_labels
    if key in (ord("r"), ord("R")):
        reset_values(state)

udp_receive_socket.close()
udp_send_socket.close()
cap.release()
cv2.destroyAllWindows()
