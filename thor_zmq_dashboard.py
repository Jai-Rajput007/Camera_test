import cv2
import zmq
import numpy as np
import time
import threading
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
            ZMQ Network Stream Active
        </div>
    </div>

    <div class="dashboard">
        <div class="video-container">
            <div class="scanline"></div>
            <img src="{{ url_for('video_feed') }}" alt="ZMQ Camera Feed" />
            
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
                    ZeroMQ (TCP)
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

latest_frame_bytes = None
frame_lock = threading.Lock()

def zmq_receiver_thread():
    """Background thread that continuously receives frames from the G1 over ZMQ"""
    global latest_frame_bytes
    
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    # Connect to the G1 Robot IP on port 5555
    socket.connect("tcp://192.168.123.164:5555")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    
    print("[INFO] Connected to G1 ZMQ Stream at 192.168.123.164:5555")
    
    while True:
        try:
            # Receive byte frame (non-blocking)
            frame_bytes = socket.recv(flags=zmq.NOBLOCK)
            with frame_lock:
                latest_frame_bytes = frame_bytes
        except zmq.Again:
            # No frame received yet, wait a tiny bit
            time.sleep(0.005)
        except Exception as e:
            print(f"[ERROR] ZMQ Error: {e}")
            time.sleep(1)

def generate_frames():
    """Generator function that yields JPEG frames from ZMQ."""
    global latest_frame_bytes
    
    while True:
        with frame_lock:
            frame_data = latest_frame_bytes
            
        if frame_data is None:
            # Placeholder if no data from G1 yet
            blank_image = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank_image, "WAITING FOR G1 STREAM...", (100, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', blank_image)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.5)
            continue
            
        # Yield the JPEG frame received from ZMQ
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        
        # Moderate loop to ~30 FPS
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
    print("Starting AGX Thor Vision Dashboard (ZMQ Version)")
    print("Ensure the G1 robot (192.168.123.164) is running 'g1_zmq_sender.py'")
    print("\nYou can access this dashboard from your desktop at:")
    print("  http://192.168.123.166:5000")
    print("="*60)
    
    # Start ZMQ receiver in a background thread
    zmq_thread = threading.Thread(target=zmq_receiver_thread, daemon=True)
    zmq_thread.start()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, threaded=True)
