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

# DepthAI for OAK-D S2 camera
import depthai as dai

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
    "confidence": 0.0
}

# REMOVED DISTANCE CONDITIONS FOR TESTING
PERSON_STABLE_THRESHOLD = 2.0  # Reduced from 3.0
GESTURE_STABLE_THRESHOLD = 1.5  # 1.5 seconds for all gestures - STABILITY REQUIRED

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def create_oak_pipeline():
    """Create DepthAI pipeline for OAK-D S2 camera"""
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define source and output
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutRgb = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("rgb")

    # Properties
    camRgb.setPreviewSize(640, 480)  # Set preview size
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    camRgb.setFps(30)  # Set FPS

    # Link nodes
    camRgb.preview.link(xoutRgb.input)

    return pipeline

def load_keypoint_classifier():
    try:
        keypoint_classifier = KeyPointClassifier()
        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            keypoint_classifier_labels = [row[0] for row in [line.strip().split(',') for line in f]]
        logger.info(f"Loaded {len(keypoint_classifier_labels)} gesture classes: {keypoint_classifier_labels}")
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
    temp_landmark_list = landmark_list.copy()
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(np.array(temp_landmark_list).flatten())
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value if max_value != 0 else 0
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def detect_person(pose_landmarks):
    """Simplified person detection - just check if we have pose landmarks"""
    required_landmarks = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER
    ]
    
    visible_count = 0
    for landmark_id in required_landmarks:
        landmark = pose_landmarks.landmark[landmark_id]
        if landmark.visibility > 0.3:  # Lowered threshold
            visible_count += 1
    
    return visible_count >= 2  # Lowered requirement

async def websocket_handler(websocket, path=""):
    client_ip = "unknown"
    try:
        client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
        logger.info(f"Client connected from {client_ip}")
        connected_clients.add(websocket)
        
        welcome_msg = {
            "type": "connection",
            "message": "Connected to OAK-D S2 person and gesture detection",
            "timestamp": time.time()
        }
        await websocket.send(json.dumps(welcome_msg))
        
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get("type") == "ping":
                    await websocket.send(json.dumps({
                        "type": "pong", 
                        "timestamp": time.time()
                    }))
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from {client_ip}")
                continue
            except Exception as e:
                logger.error(f"Error processing message from {client_ip}: {e}")
                break
        
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client {client_ip} disconnected normally")
    except websockets.exceptions.InvalidMessage as e:
        logger.info(f"Invalid WebSocket handshake from {client_ip}: {e}")
    except websockets.exceptions.ConnectionClosedError:
        logger.info(f"Connection closed unexpectedly for {client_ip}")
    except Exception as e:
        logger.error(f"Unexpected error for client {client_ip}: {e}")
    finally:
        connected_clients.discard(websocket)
        logger.info(f"Client {client_ip} cleaned up")

async def broadcast_data():
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
            
            logger.info("WebSocket server started on localhost:8765 (OAK-D S2 mode)")
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
    
    # Create DepthAI pipeline and connect to OAK-D S2
    logger.info("Initializing OAK-D S2 camera...")
    pipeline = create_oak_pipeline()
    
    # Check for available devices
    found_devices = dai.Device.getAllAvailableDevices()
    if len(found_devices) == 0:
        logger.error("No OAK-D devices found! Make sure your OAK-D S2 is connected.")
        return
    
    logger.info(f"Found {len(found_devices)} OAK-D device(s)")
    for device in found_devices:
        logger.info(f"Device: {device.name} - {device.mxid}")
    
    # Connect to device and start pipeline
    try:
        with dai.Device(pipeline) as device:
            logger.info("Connected to OAK-D S2 successfully!")
            logger.info(f"Gesture stability threshold: {GESTURE_STABLE_THRESHOLD} seconds")
            
            # Output queue will be used to get the rgb frames from the output defined above
            q = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            
            # Initialize MediaPipe
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,  # Lowered for testing
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
            
            logger.info("Starting main detection loop with OAK-D S2...")
            
            try:
                while True:
                    inRgb = q.get()  # Get RGB frame from OAK-D S2
                    
                    if inRgb is not None:
                        # Convert to OpenCV format
                        img = inRgb.getCvFrame()
                        
                        # Flip horizontally (mirror effect)
                        img = cv2.flip(img, 1)
                        debug_img = img.copy()
                        
                        # Convert BGR to RGB for MediaPipe
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_rgb.flags.writeable = False
                        
                        frame_height, frame_width = img.shape[:2]
                        
                        hand_result = hands.process(img_rgb)
                        pose_result = pose.process(img_rgb)
                        
                        img_rgb.flags.writeable = True
                        
                        person_count = 0
                        current_time = time.time()
                        
                        # SIMPLIFIED PERSON DETECTION - NO DISTANCE CONDITIONS
                        if pose_result.pose_landmarks:
                            if detect_person(pose_result.pose_landmarks):
                                if person_count == 0:
                                    with data_lock:
                                        if current_data["person_detected_time"] == 0:
                                            current_data["person_detected_time"] = current_time
                                
                                person_count = 1
                                
                                with data_lock:
                                    time_detected = current_time - current_data["person_detected_time"]
                                    person_stable = time_detected >= PERSON_STABLE_THRESHOLD
                                    current_data["person_stable"] = person_stable
                                    # Always trigger detection when person is stable (no distance conditions)
                                    if person_stable:
                                        current_data["detection_triggered"] = True
                                    else:
                                        current_data["detection_triggered"] = False
                        else:
                            with data_lock:
                                current_data["person_detected_time"] = 0
                                current_data["person_stable"] = False
                                current_data["detection_triggered"] = False

                        # GESTURE DETECTION WITH 1.5 SECOND STABILITY REQUIREMENT
                        detected_gesture = "unknown"
                        gesture_confidence = 0.0
                        
                        if hand_result.multi_hand_landmarks:
                            for hand_landmarks in hand_result.multi_hand_landmarks:
                                landmark_list = calc_landmark_list(img, hand_landmarks)
                                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                                
                                if hand_sign_id < len(keypoint_classifier_labels):
                                    detected_gesture = keypoint_classifier_labels[hand_sign_id]
                                    gesture_confidence = 1.0  # You can implement actual confidence if available
                                    logger.debug(f"OAK-D S2 Detected gesture: {detected_gesture}")
                                
                                break  # Only process first hand
                        
                        # UPDATE GESTURE STABILITY WITH 1.5 SECOND REQUIREMENT
                        with data_lock:
                            if detected_gesture != current_data["last_detected_gesture"]:
                                current_data["last_detected_gesture"] = detected_gesture
                                current_data["gesture_detected_time"] = current_time
                                current_data["gesture_stable"] = False
                                current_data["confidence"] = gesture_confidence
                                logger.debug(f"OAK-D S2 Gesture changed to: {detected_gesture}, starting stability timer")
                            else:
                                time_stable = current_time - current_data["gesture_detected_time"]
                                if time_stable >= GESTURE_STABLE_THRESHOLD and detected_gesture != "unknown":
                                    if not current_data["gesture_stable"]:
                                        logger.info(f"OAK-D S2 Gesture '{detected_gesture}' stable for {GESTURE_STABLE_THRESHOLD}s - sending to websocket")
                                    current_data["gesture_stable"] = True
                                else:
                                    current_data["gesture_stable"] = False
                                current_data["confidence"] = gesture_confidence
                        
                        # ONLY SEND STABLE GESTURES TO WEBSOCKET (1.5 SECOND REQUIREMENT)
                        stable_gesture = "unknown"
                        with data_lock:
                            # Only send gestures that have been stable for 1.5 seconds
                            if current_data["gesture_stable"] and current_data["last_detected_gesture"] != "unknown":
                                stable_gesture = current_data["last_detected_gesture"]
                        
                        # UPDATE WEBSOCKET DATA
                        with data_lock:
                            current_data.update({
                                "person_count": person_count,
                                "gesture": stable_gesture,  # Only stable gestures are sent to websocket
                                "timestamp": current_time,
                                "status": "running"
                            })
                            
                            # Log for debugging when stable gestures are sent
                            if stable_gesture != "unknown":
                                logger.info(f"OAK-D S2 WebSocket sending: person_count={person_count}, person_stable={current_data['person_stable']}, gesture={stable_gesture}")
                        
                        # VIDEO DISPLAY WITH ENHANCED FEEDBACK
                        if show_video:
                            display_img = debug_img.copy()
                            
                            with data_lock:
                                real_time_gesture = current_data["last_detected_gesture"]
                                is_stable = current_data["gesture_stable"]
                                person_stable_status = current_data["person_stable"]
                                time_detecting = current_time - current_data["gesture_detected_time"]
                                websocket_gesture = current_data["gesture"]
                                
                                if real_time_gesture != "unknown":
                                    stability_text = f"STABLE ({time_detecting:.1f}s)" if is_stable else f"DETECTING... ({time_detecting:.1f}s/{GESTURE_STABLE_THRESHOLD}s)"
                                    stability_color = (0, 255, 0) if is_stable else (0, 255, 255)
                                else:
                                    stability_text = "NO HAND"
                                    stability_color = (128, 128, 128)
                            
                            cv2.putText(display_img, f"OAK-D S2 Gesture: {real_time_gesture}", (10, 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                            cv2.putText(display_img, stability_text, (10, 90), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, stability_color, 2)
                            cv2.putText(display_img, f"Person Stable: {person_stable_status}", (10, 130), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                            cv2.putText(display_img, f"Person Count: {person_count}", (10, 170), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                            # Show what gets sent to websocket
                            cv2.putText(display_img, f"WebSocket: {websocket_gesture}", (10, 210), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                            
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
                            
                            cv2.imshow("OAK-D S2 Person & Gesture Detection", display_img)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('s'):
                            show_video = not show_video
                            if not show_video:
                                cv2.destroyAllWindows()
            
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
            finally:
                with data_lock:
                    current_data["status"] = "stopped"
                
                cv2.destroyAllWindows()
                hands.close()
                pose.close()
                logger.info("Cleanup completed")
                
    except Exception as e:
        logger.error(f"Failed to connect to OAK-D S2: {e}")
        logger.error("Make sure:")
        logger.error("1. OAK-D S2 is connected via USB")
        logger.error("2. DepthAI library is installed: pip install depthai")
        logger.error("3. No other applications are using the camera")

if __name__ == "__main__":
    try:
        import websockets
        import depthai as dai
        main()
    except ImportError as e:
        if "depthai" in str(e):
            logger.error("DepthAI library not installed. Install with: pip install depthai")
        elif "websockets" in str(e):
            logger.error("websockets library not installed. Install with: pip install websockets")
        else:
            logger.error(f"Import error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)