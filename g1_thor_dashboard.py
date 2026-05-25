import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import threading
import numpy as np
import time
from flask import Flask, Response, render_template_string

app = Flask(__name__)

# --- HTML/CSS Frontend Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AGX Thor | G1 Vision Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0f172a;
            --glass-bg: rgba(30, 41, 59, 0.7);
            --glass-border: rgba(255, 255, 255, 0.1);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: radial-gradient(circle at 50% -20%, #1e293b, var(--bg-dark) 80%);
            color: #f8fafc;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem 1rem;
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
            animation: fadeIn 1s ease-out;
        }

        .title {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(to right, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.75rem;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(16, 185, 129, 0.1);
            color: #10b981;
            padding: 0.5rem 1rem;
            border-radius: 2rem;
            font-weight: 600;
            font-size: 0.875rem;
            border: 1px solid rgba(16, 185, 129, 0.2);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .pulse {
            width: 8px;
            height: 8px;
            background-color: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            animation: pulse-animation 2s infinite;
        }

        @keyframes pulse-animation {
            0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
            100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .dashboard {
            width: 100%;
            max-width: 900px;
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 1.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            animation: fadeIn 1s ease-out 0.2s backwards;
        }

        .video-container {
            width: 100%;
            border-radius: 16px;
            overflow: hidden;
            position: relative;
            background: #000;
            aspect-ratio: 16/9;
            box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
        }

        .video-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }
        
        .overlay {
            position: absolute;
            bottom: 1rem;
            left: 1rem;
            right: 1rem;
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .metric {
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(8px);
            padding: 0.5rem 1rem;
            border-radius: 12px;
            font-size: 0.875rem;
            font-weight: 600;
            border: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            align-items: center;
        }

        .metric-label {
            color: #94a3b8;
            margin-right: 0.5rem;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Decorative scanline effect */
        .scanline {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                to bottom,
                transparent 50%,
                rgba(0, 0, 0, 0.1) 51%
            );
            background-size: 100% 4px;
            pointer-events: none;
            z-index: 10;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1 class="title">AGX Thor | G1 Vision Dashboard</h1>
        <div class="status-badge">
            <div class="pulse"></div>
            ROS2 Network Stream Active
        </div>
    </div>

    <div class="dashboard">
        <div class="video-container">
            <div class="scanline"></div>
            <img src="{{ url_for('video_feed') }}" alt="ROS2 Camera Feed" />
            
            <div class="overlay">
                <div class="metric">
                    <span class="metric-label">Source Node</span>
                    192.168.123.164 (G1)
                </div>
                <div class="metric">
                    <span class="metric-label">Compute Node</span>
                    192.168.123.166 (Thor)
                </div>
                <div class="metric">
                    <span class="metric-label">Protocol</span>
                    ROS2 (rclpy)
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

# Global variables for cross-thread frame sharing
latest_frame = None
frame_lock = threading.Lock()

class CameraSubscriber(Node):
    """ROS2 Node that subscribes to the RealSense image stream published by the G1"""
    def __init__(self):
        super().__init__('agx_thor_vision_dashboard')
        self.bridge = CvBridge()
        
        # Subscribe to the standard RealSense ROS2 topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.listener_callback,
            10
        )
        self.get_logger().info("Subscribed to G1 Camera ROS2 topic '/camera/color/image_raw'...")

    def listener_callback(self, msg):
        global latest_frame
        try:
            # Convert ROS2 Image message to an OpenCV image (numpy array)
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Safely copy it to our global variable for the Flask thread to consume
            with frame_lock:
                latest_frame = cv_image.copy()
        except Exception as e:
            self.get_logger().error(f'Error converting ROS2 image: {e}')

def ros2_thread_func():
    """Runs the ROS2 event loop in a separate thread."""
    rclpy.init(args=None)
    node = CameraSubscriber()
    # rclpy.spin blocks until interrupted
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

def generate_frames():
    """Generator function that yields JPEG frames from the ROS2 subscriber."""
    global latest_frame
    
    while True:
        with frame_lock:
            frame = latest_frame
            
        if frame is None:
            # Yield a placeholder blank frame if no data from ROS2 yet
            blank_image = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank_image, "WAITING FOR G1 ROS2 STREAM...", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', blank_image)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1.0)
            continue
            
        # Encode the OpenCV frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        # Yield the frame over the MJPEG stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
               
        # Limit to ~30 FPS to save bandwidth on the dashboard
        time.sleep(1.0 / 30.0)

@app.route('/')
def index():
    """Serve the dashboard UI."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    """Route that returns the MJPEG stream response."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("="*60)
    print("Starting AGX Thor Vision Dashboard")
    print("Ensure the G1 robot (192.168.123.164) is running: ")
    print("  ros2 launch realsense2_camera rs_launch.py")
    print("\nYou can access this dashboard from your desktop at:")
    print("  http://192.168.123.166:5000")
    print("="*60)
    
    # Start ROS2 Node in a background thread
    ros_thread = threading.Thread(target=ros2_thread_func, daemon=True)
    ros_thread.start()
    
    # Run the Flask app on all interfaces
    app.run(host='0.0.0.0', port=5000, threaded=True)
