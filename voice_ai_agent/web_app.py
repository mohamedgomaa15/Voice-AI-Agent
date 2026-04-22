from flask import Flask, request, jsonify, render_template_string
import os
import tempfile
import sys
from pathlib import Path
from flask_cors import CORS

# Add parent directory to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voice_ai_agent.pipeline import agent_system_setclass_appmatch
from voice_ai_agent.command_executer import CommandExecutor
from voice_ai_agent.english_stt_optimized2 import EnglishSTT
from voice_ai_agent.api import api_bp

import subprocess

def convert_to_wav(input_path):
    output_path = input_path.replace(".webm", ".wav")
    
    subprocess.run([
        "ffmpeg",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output_path

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Register API blueprint
app.register_blueprint(api_bp)

# Initialize components
stt = EnglishSTT(device="cpu")
executor = CommandExecutor()

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice AI Agent Demo</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .section {
            background: white;
            margin-bottom: 25px;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }
        
        .section:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        }
        
        .section h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.5em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        
        input[type="text"] {
            width: 100%;
            padding: 12px;
            margin: 10px 0;
            border: 2px solid #e0e0e0;
            border-radius: 5px;
            font-size: 1em;
            transition: border-color 0.3s ease;
        }
        
        input[type="text"]:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 10px rgba(102,126,234,0.2);
        }
        
        button {
            padding: 12px 25px;
            margin: 8px 5px;
            cursor: pointer;
            border: none;
            border-radius: 5px;
            font-size: 1em;
            font-weight: 600;
            transition: all 0.3s ease;
            text-transform: uppercase;
        }
        
        button:not(:disabled) {
            box-shadow: 0 4px 15px rgba(102,126,234,0.4);
        }
        
        #startRecording {
            background: #4CAF50;
            color: white;
        }
        
        #startRecording:hover:not(:disabled) {
            background: #45a049;
            box-shadow: 0 6px 20px rgba(76,175,80,0.4);
        }
        
        #stopRecording {
            background: #f44336;
            color: white;
        }
        
        #stopRecording:hover:not(:disabled) {
            background: #da190b;
            box-shadow: 0 6px 20px rgba(244,67,54,0.4);
        }
        
        button[onclick="processManual()"] {
            background: #667eea;
            color: white;
        }
        
        button[onclick="processManual()"]:hover {
            background: #5568d3;
            box-shadow: 0 6px 20px rgba(102,126,234,0.4);
        }
        
        button:disabled {
            background: #cccccc;
            color: #999999;
            cursor: not-allowed;
            opacity: 0.6;
        }
        
        #audioControls {
            margin: 20px 0;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: flex-start;
        }
        
        #recordingStatus {
            color: #764ba2;
            font-weight: 600;
            margin-top: 15px;
            font-size: 1.1em;
            padding: 10px;
            background: #f5f5f5;
            border-left: 4px solid #667eea;
            border-radius: 3px;
        }
        
        .status-recording {
            background: #fff3cd !important;
            border-left-color: #ffc107 !important;
            color: #856404 !important;
        }
        
        .status-processing {
            background: #d1ecf1 !important;
            border-left-color: #17a2b8 !important;
            color: #0c5460 !important;
        }
        
        #result {
            margin-top: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .result-content {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-top: 10px;
        }
        
        #result h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        #result p {
            margin: 10px 0;
            line-height: 1.6;
            color: #333;
        }
        
        #result strong {
            color: #764ba2;
        }
        
        .success {
            color: #4CAF50;
            font-weight: 600;
        }
        
        .failed {
            color: #f44336;
            font-weight: 600;
        }
        
        .transcription-box {
            background: #f0f4ff;
            border-left: 4px solid #667eea;
            padding: 10px;
            border-radius: 3px;
            margin: 10px 0;
        }
        
        .intent-badge {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 5px 10px;
            border-radius: 20px;
            font-weight: 600;
            margin: 5px 0;
        }
        
        .entities-box {
            background: #f9f9f9;
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-family: 'Courier New', monospace;
            max-height: 200px;
            overflow-y: auto;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .recording {
            animation: pulse 0.7s infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Voice AI Agent Demo</h1>

        <div class="section">
            <h2>📝 Manual Command Input</h2>
            <input type="text" id="manualCommand" placeholder="Enter your command here...">
            <button onclick="processManual()">Process Command</button>
        </div>

        <div class="section">
            <h2>🎙️ Voice Command Input</h2>
            <div id="audioControls">
                <button id="startRecording">🔴 Start Recording</button>
                <button id="stopRecording" disabled>⏹️ Stop Recording</button>
            </div>
            <p id="recordingStatus">Click "Start Recording" to begin</p>
        </div>

        <div id="result"></div>
    </div>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        let audioBlob;

        // Check if getUserMedia is available
        function checkMediaDevicesSupport() {
            if (!navigator.mediaDevices) {
                console.error('mediaDevices API not available');
                return false;
            }
            if (!navigator.mediaDevices.getUserMedia) {
                console.error('getUserMedia not available');
                return false;
            }
            return true;
        }

        document.getElementById('startRecording').onclick = async () => {
            try {
                // Check browser support
                if (!checkMediaDevicesSupport()) {
                    throw new Error('Microphone access not supported. Please use HTTPS or ensure your browser has microphone support enabled.');
                }

                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = () => {
                    audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    document.getElementById('startRecording').disabled = false;
                    document.getElementById('recordingStatus').textContent = '⏳ Processing audio...';
                    document.getElementById('recordingStatus').classList.add('status-processing');
                    processAudio();
                };

                mediaRecorder.start();
                document.getElementById('startRecording').disabled = true;
                document.getElementById('stopRecording').disabled = false;
                document.getElementById('recordingStatus').textContent = '🔴 Recording...';
                document.getElementById('recordingStatus').classList.add('status-recording');
                document.getElementById('recordingStatus').classList.remove('status-processing');
            } catch (err) {
                let errorMsg = err.message;
                if (err.name === 'NotAllowedError') {
                    errorMsg = 'Microphone access denied. Please grant permission in browser settings.';
                } else if (err.name === 'NotFoundError') {
                    errorMsg = 'No microphone found. Please check your audio devices.';
                } else if (err.name === 'NotSupportedError') {
                    errorMsg = 'getUserMedia not supported on this browser.';
                } else if (err.name === 'SecurityError') {
                    errorMsg = 'SecurityError: Page must be served over HTTPS (or localhost).';
                }
                alert('Error accessing microphone: ' + errorMsg);
                console.error('Microphone error:', err);
            }
        };

        document.getElementById('stopRecording').onclick = () => {
            mediaRecorder.stop();
            document.getElementById('stopRecording').disabled = true;
        };

        async function processManual() {
            const command = document.getElementById('manualCommand').value;
            if (!command.trim()) {
                alert('Please enter a command');
                return;
            }

            document.getElementById('result').innerHTML = '<div class="result-content"><p>⏳ Processing...</p></div>';

            try {
                console.log('Sending request to /process_manual...');
                const response = await fetch('/process_manual', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command: command })
                });

                console.log('Response status:', response.status);
                console.log('Response headers:', response.headers);

                if (!response.ok) {
                    const errorText = await response.text();
                    console.error('Response error:', errorText);
                    throw new Error(`HTTP ${response.status}: ${errorText}`);
                }

                const data = await response.json();
                console.log('Response data:', data);
                displayResult(data);
            } catch (err) {
                console.error('Fetch error:', err);
                document.getElementById('result').innerHTML = '<div class="result-content"><p style="color: #f44336;">❌ Error: ' + err.message + '</p></div>';
            }
        }

        async function processAudio() {
            if (!audioBlob) {
                alert('No audio recorded');
                return;
            }

            document.getElementById('result').innerHTML = '<div class="result-content"><p>⏳ Processing audio...</p></div>';

            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.wav');

            try {
                console.log('Sending audio to /process_audio...');
                console.log('Audio blob size:', audioBlob.size);

                const controller = new AbortController();
                const timeout = setTimeout(() => controller.abort(), 30000);

                const response = await fetch('/process_audio', {
                    method: 'POST',
                    body: formData,
                });
                clearTimeout(timeout);

                console.log('Response status:', response.status);
                console.log('Response headers:', response.headers);

                const text = await response.text();
                console.log('Response text:', text);

                let data;

                if (response.headers.get('content-type')?.includes('application/json')) {
                    data = JSON.parse(text);
                } else {
                    throw new Error('Server returned non-JSON response: ' + text.slice(0, 200));
                }

                if (!response.ok) {
                    throw new Error(data.error || JSON.stringify(data));
                }

                console.log('Parsed data:', data);
                displayResult(data);
            } catch (err) {
                console.error('Audio processing error:', err);
                document.getElementById('result').innerHTML = '<div class="result-content"><p style="color: #f44336;">❌ Error: ' + err.message + '</p></div>';
            }
        }

        function displayResult(data) {
            let html = '<div class="result-content"><h3>📊 Result:</h3>';
            
            if (data.transcription) {
                html += '<div class="transcription-box"><strong>📝 Transcription:</strong> ' + escapeHtml(data.transcription) + '</div>';
            }
            
            html += '<p><strong>🎯 Intent:</strong> <span class="intent-badge">' + escapeHtml(data.intent) + '</span></p>';
            
            html += '<p><strong>🏷️ Entities:</strong></p>';
            html += '<div class="entities-box">' + JSON.stringify(data.entities, null, 2) + '</div>';
            
            if (data.execution_success !== undefined) {
                const statusClass = data.execution_success ? 'success' : 'failed';
                const statusIcon = data.execution_success ? '✅' : '❌';
                html += '<p><strong>⚙️ Execution:</strong> <span class="' + statusClass + '">' + statusIcon + ' ' + (data.execution_success ? 'Success' : 'Failed') + '</span></p>';
                html += '<p><strong>💬 Message:</strong> ' + escapeHtml(data.execution_message) + '</p>';
            }
            
            html += '</div>';
            document.getElementById('result').innerHTML = html;
            document.getElementById('recordingStatus').classList.remove('status-recording', 'status-processing');
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/process_manual', methods=['POST'])
def process_manual():
    try:
        data = request.get_json()
        if not data or 'command' not in data:
            return jsonify({'error': 'No command provided'}), 400

        command = data.get('command', '')

        # Process with agent
        result = agent_system_setclass_appmatch(command)

        # Execute command
        success, message = executor.execute_command(result['intent'], result['entity'])

        return jsonify({
            'intent': result['intent'],
            'entities': result['entity'],
            'execution_success': success,
            'execution_message': message
        })
    except Exception as e:
        print(f"Error in process_manual: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save to temporary file safely on Windows
        fd, temp_path = tempfile.mkstemp(suffix='.webm')
        os.close(fd)
        audio_file.save(temp_path)

        try:
            # Transcribe
            # Convert webm → wav first
            wav_path = convert_to_wav(temp_path)

            # Transcribe
            transcription_result = stt.transcribe_file(wav_path)
            command = transcription_result['text'].strip()

            # Process with agent
            result = agent_system_setclass_appmatch(command)

            # Execute command
            success, message = executor.execute_command(result['intent'], result['entity'])

            return jsonify({
                'transcription': command,
                'intent': result['intent'],
                'entities': result['entity'],
                'execution_success': success,
                'execution_message': message
            })
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            if 'wav_path' in locals() and os.path.exists(wav_path):
                os.unlink(wav_path)
    except Exception as e:
        print(f"Error in process_audio: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import ssl

    print("🚀 Starting Voice AI Agent Web App...")
    print("📡 Server will be available at:")
    print("   - Local: http://localhost:5000")
    print("   - Network: http://0.0.0.0:5000")
    print("🔒 For microphone access, use: http://localhost:5000")

    # Try to use HTTPS with self-signed certificate
    # If certificates don't exist, fall back to HTTP
    try:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain('cert.pem', 'key.pem')
        print("🔐 SSL certificates found - using HTTPS")
        app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=context)
    except FileNotFoundError:
        print("⚠️  SSL certificates not found - using HTTP")
        print("💡 For microphone access, consider generating SSL certificates")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"⚠️  SSL setup failed: {e} - using HTTP")
        app.run(debug=True, host='0.0.0.0', port=5000)