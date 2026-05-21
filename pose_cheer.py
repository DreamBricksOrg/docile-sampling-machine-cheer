import cv2
import mediapipe as mp
import time
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker_lite.task"
MAX_PEOPLE = 3
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
DEFAULT_LANDMARK_COLOR = (0, 0, 255)
DEFAULT_CONNECTION_COLOR = (224, 224, 224)
MIN_VISIBILITY = 0.5
ABOVE_HEAD_MARGIN = 0.04
ACTIVE_TRACKING_DISTANCE = 0.25
LOST_TRACKING_DISTANCE = 0.45
MAX_MISSING_SECONDS = 8.0
MAX_MISSING_SECONDS_AFTER_COUNT = 20.0


def is_visible(landmark):
    return getattr(landmark, "visibility", 1.0) >= MIN_VISIBILITY


def is_cheering_pose(landmarks):
    pose_landmark = vision.PoseLandmark
    nose = landmarks[pose_landmark.NOSE.value]
    left_wrist = landmarks[pose_landmark.LEFT_WRIST.value]
    right_wrist = landmarks[pose_landmark.RIGHT_WRIST.value]
    left_elbow = landmarks[pose_landmark.LEFT_ELBOW.value]
    right_elbow = landmarks[pose_landmark.RIGHT_ELBOW.value]
    left_shoulder = landmarks[pose_landmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[pose_landmark.RIGHT_SHOULDER.value]

    important_points = (
        nose,
        left_wrist,
        right_wrist,
        left_elbow,
        right_elbow,
        left_shoulder,
        right_shoulder,
    )
    if not all(is_visible(point) for point in important_points):
        return False

    wrists_above_head = (
        left_wrist.y < nose.y - ABOVE_HEAD_MARGIN
        and right_wrist.y < nose.y - ABOVE_HEAD_MARGIN
    )
    elbows_lifted = (
        left_elbow.y < left_shoulder.y
        and right_elbow.y < right_shoulder.y
    )
    return wrists_above_head and elbows_lifted


def pose_center(landmarks):
    pose_landmark = vision.PoseLandmark
    center_points = (
        landmarks[pose_landmark.LEFT_SHOULDER.value],
        landmarks[pose_landmark.RIGHT_SHOULDER.value],
        landmarks[pose_landmark.LEFT_HIP.value],
        landmarks[pose_landmark.RIGHT_HIP.value],
    )
    visible_points = [point for point in center_points if is_visible(point)]
    if not visible_points:
        return None

    x = sum(point.x for point in visible_points) / len(visible_points)
    y = sum(point.y for point in visible_points) / len(visible_points)
    return x, y


def pose_size(landmarks):
    visible_points = [point for point in landmarks if is_visible(point)]
    if not visible_points:
        return 0.0

    min_x = min(point.x for point in visible_points)
    max_x = max(point.x for point in visible_points)
    min_y = min(point.y for point in visible_points)
    max_y = max(point.y for point in visible_points)
    return max(max_x - min_x, max_y - min_y)


def distance(point_a, point_b):
    return ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5


def matching_score(center, size, person):
    center_distance = distance(center, person["center"])
    size_distance = abs(size - person["size"])
    return center_distance + (size_distance * 0.4)


def track_people(detected_landmarks, tracked_people, next_person_id, now):
    for person in tracked_people:
        person["visible"] = False

    current_people = []
    available_people = set(range(len(tracked_people)))

    for landmarks in detected_landmarks[:MAX_PEOPLE]:
        center = pose_center(landmarks)
        if center is None:
            continue

        size = pose_size(landmarks)
        best_index = None
        best_score = None
        for index in available_people:
            person = tracked_people[index]
            missing_time = now - person["last_seen"]
            max_distance = (
                ACTIVE_TRACKING_DISTANCE
                if missing_time < 0.25
                else LOST_TRACKING_DISTANCE
            )
            if distance(center, person["center"]) > max_distance:
                continue

            current_score = matching_score(center, size, person)
            if best_score is None or current_score < best_score:
                best_index = index
                best_score = current_score

        if best_index is None:
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
            person = tracked_people[best_index]
            person["center"] = center
            person["size"] = size
            person["last_seen"] = now
            person["visible"] = True
            available_people.remove(best_index)

        current_people.append((landmarks, person))

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


def draw_pose(frame, landmarks, person, is_cheering, debug_labels):
    landmark_color = GREEN if is_cheering else DEFAULT_LANDMARK_COLOR
    connection_color = GREEN if is_cheering else DEFAULT_CONNECTION_COLOR

    vision.drawing_utils.draw_landmarks(
        frame,
        landmarks,
        vision.PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=vision.drawing_utils.DrawingSpec(
            color=landmark_color,
            thickness=2,
            circle_radius=2,
        ),
        connection_drawing_spec=vision.drawing_utils.DrawingSpec(
            color=connection_color,
            thickness=2,
            circle_radius=2,
        ),
    )

    if debug_labels:
        height, width = frame.shape[:2]
        label_x = int(person["center"][0] * width)
        label_y = int(person["center"][1] * height)
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


cap = cv2.VideoCapture(0)

options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=MAX_PEOPLE,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

with vision.PoseLandmarker.create_from_options(options) as pose:
    start_time = time.monotonic()
    tracked_people = []
    next_person_id = 1
    cheer_count = 0
    debug_labels = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.monotonic() - start_time) * 1000)
        results = pose.detect_for_video(mp_image, timestamp_ms)
        now = time.monotonic()

        current_people, next_person_id = track_people(
            results.pose_landmarks,
            tracked_people,
            next_person_id,
            now,
        )

        for landmarks, person in current_people:
            is_cheering = is_cheering_pose(landmarks)
            if is_cheering and not person["counted"]:
                person["counted"] = True
                cheer_count += 1

            draw_pose(frame, landmarks, person, is_cheering, debug_labels)

        if debug_labels:
            draw_counter(frame, cheer_count)

        cv2.imshow("MediaPipe Pose", frame)
        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            break
        if key in (ord("d"), ord("D")):
            debug_labels = not debug_labels
        if key in (ord("r"), ord("R")):
            tracked_people = []
            next_person_id = 1
            cheer_count = 0

cap.release()
cv2.destroyAllWindows()
