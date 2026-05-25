import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response, render_template_string
import time

app = Flask(__name__)

# --- HTML/CSS Frontend Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>G1 Robot Vision Dashboard</title>
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
            background: linear-gradient(to right, #38bdf8, #818cf8);
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
        <h1 class="title">G1 Robot Vision</h1>
        <div class="status-badge">
            <div class="pulse"></div>
            Camera Active
        </div>
    </div>

    <div class="dashboard">
        <div class="video-container">
            <div class="scanline"></div>
            <img src="{{ url_for('video_feed') }}" alt="RealSense Camera Feed" />
            
            <div class="overlay">
                <div class="metric">
                    <span class="metric-label">Sensor</span>
                    RealSense D435i
                </div>
                <div class="metric">
                    <span class="metric-label">Format</span>
                    RGB &bull; 640x480
                </div>
                <div class="metric">
                    <span class="metric-label">Stream</span>
                    <span id="fps">30</span> FPS
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

# Global pipeline variables
pipeline = None
camera_started = False

def init_camera():
    global pipeline, camera_started
    if camera_started:
        return
    
    try:
        pipeline = rs.pipeline()
        config = rs.config()

        # Enable color stream
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        pipeline.start(config)
        camera_started = True
        print("[INFO] Camera pipeline started successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to start camera: {e}")
        # Note: In a real deployment, you might want to mock the camera if disconnected
        pass

def generate_frames():
    """Generator function that yields JPEG frames from RealSense pipeline."""
    global pipeline, camera_started
    
    # Attempt to init if not already
    init_camera()

    while True:
        if not camera_started or pipeline is None:
            # Yield a placeholder blank frame if camera isn't working
            blank_image = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank_image, "CAMERA DISCONNECTED", (100, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', blank_image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1.0)
            continue
            
        try:
            # Wait for a coherent pair of frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue

            # Convert image to numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', color_image)
            frame = buffer.tobytes()

            # Yield the frame in byte format for HTTP streaming (MJPEG)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print(f"[ERROR] Stream interrupted: {e}")
            time.sleep(0.5)

@app.route('/')
def index():
    """Serve the dashboard UI."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    """Route that returns the MJPEG stream response."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("="*50)
    print("Starting G1 Vision Dashboard")
    print("Ensure the robot's RealSense camera is connected.")
    print("You can access this dashboard from any browser at:")
    print("http://<ROBOT_IP>:5000")
    print("="*50)
    
    # Run the Flask app on all interfaces so it can be accessed over the network
    app.run(host='0.0.0.0', port=5000, threaded=True)
