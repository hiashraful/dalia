import depthai as dai
import cv2
import mediapipe as mp
import numpy as np
import asyncio
import websockets
import json
import threading
import time
import logging
from collections import Counter, deque
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.point_history_classifier.point_history_classifier import PointHistoryClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

connected_clients = set()
data_lock = threading.Lock()
current_data = {
    "person_count": 0,
    "gesture": "unknown",
    "timestamp": time.time(),
    "status": "starting",
    "detection_triggered": False,
    "last_trigger_time": 0,
    "person_detected_time": 0,
    "person_stable": False,
    "gesture_detected_time": 0,
    "gesture_stable": False,
    "last_detected_gesture": "unknown",
    "depth_enabled": True,
    "distance_to_person": 0.0
}

MIN_DISTANCE = 0.5
MAX_DISTANCE = 2.0
CENTER_THRESHOLD = 0.3
COOLDOWN_TIME = 10
PERSON_STABLE_THRESHOLD = 3.0
GESTURE_STABLE_THRESHOLD = 1.2

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def create_oak_d_s2_pipeline():
    """Create DepthAI pipeline for Oak D S2 camera"""
    # Create pipeline
    pipeline = dai.Pipeline()
    
    # Define sources and outputs
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    depth = pipeline.create(dai.node.MonoCamera)
    depth2 = pipeline.create(dai.node.MonoCamera)
    depth_output = pipeline.create(dai.node.StereoDepth)
    
    # RGB camera properties
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    cam_rgb.setFps(30)
    
    # Mono cameras for depth
    depth.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    depth.setBoardSocket(dai.CameraBoardSocket.LEFT)
    depth2.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    depth2.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    
    # Create depth output
    depth_output.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    depth_output.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    depth_output.setLeftRightCheck(True)
    depth_output.setSubpixel(False)
    depth_output.setExtendedDisparity(False)
    
    # Link cameras to depth
    depth.out.link(depth_output.left)
    depth2.out.link(depth_output.right)
    
    # Create outputs
    rgb_out = pipeline.create(dai.node.XLinkOut)
    depth_out = pipeline.create(dai.node.XLinkOut)
    
    rgb_out.setStreamName("rgb")
    depth_out.setStreamName("depth")
    
    # Link to outputs
    cam_rgb.preview.link(rgb_out.input)
    depth_output.depth.link(depth_out.input)
    
    return pipeline

def load_keypoint_classifier():
    try:
        keypoint_classifier = KeyPointClassifier()
        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            keypoint_classifier_labels = [row[0] for row in [line.strip().split(',') for line in f]]
        return keypoint_classifier, keypoint_classifier_labels
    except Exception as e:
        logger.error(f"Error loading keypoint classifier: {e}")
        return None, ["unknown"]

def load_point_history_classifier():
    try:
        point_history_classifier = PointHistoryClassifier()
        with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
            point_history_classifier_labels = [row[0] for row in [line.strip().split(',') for line in f]]
        return point_history_classifier, point_history_classifier_labels
    except Exception as e:
        logger.error(f"Error loading point history classifier: {e}")
        return None, ["unknown"]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]
        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
    return temp_point_history

def detect_person(pose_landmarks):
    """Enhanced person detection using pose landmarks"""
    if not pose_landmarks:
        return False
    
    # Check if key body parts are visible
    key_points = [
        pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE],
        pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER],
        pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER],
        pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW],
        pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    ]
    
    visible_count = sum(1 for point in key_points if point.visibility > 0.5)
    return visible_count >= 3

def is_person_centered_and_in_range(pose_landmarks, depth_frame, frame_width, frame_height):
    """Check if person is centered and within optimal distance using depth data"""
    if not pose_landmarks:
        return False, 0.0
    
    # Get nose position for center detection
    nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    nose_x = int(nose.x * frame_width)
    nose_y = int(nose.y * frame_height)
    
    # Check if person is centered
    center_x = frame_width // 2
    center_y = frame_height // 2
    
    distance_from_center_x = abs(nose_x - center_x) / frame_width
    distance_from_center_y = abs(nose_y - center_y) / frame_height
    
    is_centered = (distance_from_center_x < CENTER_THRESHOLD and 
                   distance_from_center_y < CENTER_THRESHOLD)
    
    # Get depth at nose position
    distance = 0.0
    if depth_frame is not None and 0 <= nose_x < depth_frame.shape[1] and 0 <= nose_y < depth_frame.shape[0]:
        # Sample area around nose for more stable reading
        sample_size = 20
        x1 = max(0, nose_x - sample_size)
        x2 = min(depth_frame.shape[1], nose_x + sample_size)
        y1 = max(0, nose_y - sample_size)
        y2 = min(depth_frame.shape[0], nose_y + sample_size)
        
        depth_region = depth_frame[y1:y2, x1:x2]
        valid_depths = depth_region[depth_region > 0]
        
        if len(valid_depths) > 0:
            distance = np.median(valid_depths) / 1000.0  # Convert mm to meters
    
    is_in_range = MIN_DISTANCE <= distance <= MAX_DISTANCE
    
    return is_centered and is_in_range, distance

async def websocket_handler(websocket, path):
    """Handle WebSocket connections"""
    try:
        connected_clients.add(websocket)
        client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
        logger.info(f"Client {client_ip} connected. Total clients: {len(connected_clients)}")
        
        await websocket.wait_closed()
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        connected_clients.discard(websocket)
        client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
        logger.info(f"Client {client_ip} cleaned up")

async def broadcast_data():
    """Broadcast detection data to all connected clients"""
    while True:
        if connected_clients:
            with data_lock:
                message = json.dumps(current_data)
            
            disconnected_clients = set()
            for client in connected_clients.copy():
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
                except Exception:
                    disconnected_clients.add(client)
            
            for client in disconnected_clients:
                connected_clients.discard(client)
        
        await asyncio.sleep(1/30)

def start_websocket_server():
    """Start WebSocket server in a separate thread"""
    async def server():
        try:
            server = await websockets.serve(
                websocket_handler, 
                "localhost", 
                8765,
                ping_interval=20,
                ping_timeout=10,
                max_size=2**16,
                compression=None,
                process_request=None
            )
            
            logger.info("WebSocket server started on localhost:8765")
            broadcast_task = asyncio.create_task(broadcast_data())
            
            try:
                await server.wait_closed()
            except KeyboardInterrupt:
                logger.info("Shutting down WebSocket server...")
                broadcast_task.cancel()
                server.close()
                await server.wait_closed()
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server())
    except Exception as e:
        logger.error(f"Failed to start WebSocket server: {e}")

def main():
    global current_data
    
    logger.info("Starting WebSocket server...")
    websocket_thread = threading.Thread(target=start_websocket_server, daemon=True)
    websocket_thread.start()
    
    time.sleep(2)
    
    logger.info("Loading ML models...")
    keypoint_classifier, keypoint_classifier_labels = load_keypoint_classifier()
    point_history_classifier, point_history_classifier_labels = load_point_history_classifier()
    
    if not keypoint_classifier:
        logger.error("Failed to load keypoint classifier")
        return
    
    # Create Oak D S2 pipeline and device
    logger.info("Initializing Oak D S2 camera...")
    pipeline = create_oak_d_s2_pipeline()
    
    try:
        device = dai.Device(pipeline)
        logger.info("Oak D S2 camera connected successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Oak D S2 camera: {e}")
        logger.info("Make sure the Oak D S2 is connected via USB")
        return
    
    # Output queues
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    
    # Initialize MediaPipe
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.85,
        min_tracking_confidence=0.5,
        model_complexity=0
    )
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)
    
    show_video = True
    
    logger.info("Starting main detection loop...")
    
    try:
        while True:
            # Get RGB and depth frames
            rgb_frame = rgb_queue.get().getCvFrame()
            depth_frame = depth_queue.get().getFrame()
            
            if rgb_frame is None:
                continue
            
            # Process RGB frame
            img = np.copy(rgb_frame)
            img = cv2.flip(img, 1)  # Mirror for better UX
            debug_img = img.copy()
            
            # Convert depth to proper format for processing
            if depth_frame is not None:
                depth_frame = cv2.flip(depth_frame, 1)  # Mirror depth too
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            
            frame_height, frame_width = img.shape[:2]
            
            # MediaPipe processing
            hand_result = hands.process(img_rgb)
            pose_result = pose.process(img_rgb)
            
            img_rgb.flags.writeable = True
            
            person_count = 0
            detection_triggered = False
            distance_to_person = 0.0
            
            # STRICT distance-based person detection
            if pose_result.pose_landmarks:
                if detect_person(pose_result.pose_landmarks):
                    # Check distance FIRST - only detect if in range
                    is_in_range, distance_to_person = is_person_centered_and_in_range(
                        pose_result.pose_landmarks, depth_frame, frame_width, frame_height
                    )
                    
                    # ONLY count as detected if within distance range
                    if is_in_range:
                        current_time = time.time()
                        
                        if person_count == 0:
                            with data_lock:
                                if current_data["person_detected_time"] == 0:
                                    current_data["person_detected_time"] = current_time
                        
                        person_count = 1
                        
                        with data_lock:
                            time_detected = current_time - current_data["person_detected_time"]
                            person_stable = time_detected >= PERSON_STABLE_THRESHOLD
                            current_data["person_stable"] = person_stable
                            current_data["distance_to_person"] = distance_to_person
                            
                            if (current_data["person_stable"] and 
                                current_time - current_data["last_trigger_time"] > COOLDOWN_TIME):
                                detection_triggered = True
                                current_data["detection_triggered"] = True
                                current_data["last_trigger_time"] = current_time
                                logger.info(f"Person detected and stable at {distance_to_person:.2f}m")
                    else:
                        # Person visible but outside range - reset detection
                        with data_lock:
                            current_data["person_detected_time"] = 0
                            current_data["person_stable"] = False
                            current_data["distance_to_person"] = distance_to_person
            else:
                # No person detected OR person outside range
                with data_lock:
                    current_data["person_detected_time"] = 0
                    current_data["person_stable"] = False
                    current_data["distance_to_person"] = 0.0
            
            # Gesture recognition
            real_time_gesture = "unknown"
            gesture_stability = False
            
            if hand_result.multi_hand_landmarks:
                for hand_landmarks in hand_result.multi_hand_landmarks:
                    landmark_list = calc_landmark_list(img, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    real_time_gesture = keypoint_classifier_labels[hand_sign_id]
                    
                    finger_gesture_history.append(hand_sign_id)
                    most_common_fg = Counter(finger_gesture_history).most_common()
                    
                    if most_common_fg:
                        stable_gesture = keypoint_classifier_labels[most_common_fg[0][0]]
                        gesture_stability = most_common_fg[0][1] >= (history_length * 0.7)
                        
                        current_time = time.time()
                        
                        if stable_gesture != current_data["last_detected_gesture"]:
                            with data_lock:
                                current_data["gesture_detected_time"] = current_time
                                current_data["last_detected_gesture"] = stable_gesture
                        
                        with data_lock:
                            time_stable = current_time - current_data["gesture_detected_time"]
                            current_data["gesture_stable"] = (
                                gesture_stability and 
                                time_stable >= GESTURE_STABLE_THRESHOLD
                            )
                            
                            if current_data["gesture_stable"]:
                                current_data["gesture"] = stable_gesture
                                logger.info(f"Stable gesture detected: {stable_gesture}")
            else:
                with data_lock:
                    current_data["gesture_detected_time"] = 0
                    current_data["gesture_stable"] = False
                    current_data["last_detected_gesture"] = "unknown"
            
            # Update global data
            with data_lock:
                current_data["person_count"] = person_count
                current_data["timestamp"] = time.time()
                current_data["status"] = "running"
            
            # Display video if enabled
            if show_video:
                display_img = debug_img.copy()
                
                # Display depth info if available
                if depth_frame is not None:
                    cv2.putText(display_img, f"Distance: {distance_to_person:.2f}m", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Display person detection status
                person_status = f"Person: {person_count}"
                if current_data["person_stable"]:
                    person_status += " (STABLE)"
                    person_color = (0, 255, 0)
                else:
                    person_status += " (DETECTING...)"
                    person_color = (0, 255, 255)
                
                cv2.putText(display_img, person_status, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, person_color, 2)
                
                # Display gesture info
                is_stable = current_data["gesture_stable"]
                stability_text = "STABLE" if is_stable else "DETECTING..."
                stability_color = (0, 255, 0) if is_stable else (0, 255, 255)
                
                cv2.putText(display_img, f"Gesture: {real_time_gesture}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display_img, stability_text, (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, stability_color, 2)
                
                # Draw landmarks
                if hand_result.multi_hand_landmarks:
                    for hand_landmarks in hand_result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            display_img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                        )
                
                if pose_result.pose_landmarks:
                    mp_draw.draw_landmarks(
                        display_img, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2)
                    )
                
                cv2.imshow("Oak D S2 - Person & Gesture Detection", display_img)
                
                # Show depth visualization (optional)
                if depth_frame is not None:
                    # Normalize depth for display
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET
                    )
                    cv2.imshow("Depth View", depth_colormap)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                show_video = not show_video
                if not show_video:
                    cv2.destroyAllWindows()
            elif key == ord('d'):
                # Toggle depth view
                pass
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        with data_lock:
            current_data["status"] = "stopped"
        
        device.close()
        cv2.destroyAllWindows()
        hands.close()
        pose.close()
        logger.info("Oak D S2 cleanup completed")

if __name__ == "__main__":
    try:
        import depthai as dai
        import websockets
        import copy
        import itertools
        main()
    except ImportError as e:
        logger.error(f"Required library not installed: {e}")
        logger.info("Install with: pip install depthai websockets")
        exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)