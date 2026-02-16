from flask import Flask, request, render_template_string, redirect, url_for, jsonify
import os
import tempfile
import base64
from pathlib import Path
import re

try:
    # Prefer multilingual system if available
    from extract_entities import MultilingualHybridSystem as SystemClass
except Exception:
    from extract_entities import HybridIntentSystem as SystemClass

try:
    from arabic_stt import ArabicSTT
    STT_AVAILABLE = True
    print("✓ STT module loaded successfully")
except Exception as e:
    STT_AVAILABLE = False
    print(f"✗ Warning: STT not available: {e}")
    print("  To enable voice features:")
    print("  1. pip install openai-whisper torch")
    print("  2. Ensure arabic_stt.py is in the same directory")

try:
    from arabic_text_corrector import correct_arabic_text
    CORRECTOR_AVAILABLE = True
    print("✓ Arabic text corrector loaded")
except Exception as e:
    CORRECTOR_AVAILABLE = False
    print(f"✗ Arabic corrector not available: {e}")
    print("  Transcription errors may affect accuracy")
    # Fallback corrector
    def correct_arabic_text(text, debug=False):
        return text

try:
    from command_executer import CommandExecutor
    EXECUTOR_AVAILABLE = True
    executor = CommandExecutor()
    print("✓ Command executor loaded")
except Exception as e:
    EXECUTOR_AVAILABLE = False
    executor = None
    print(f"✗ Command executor not available: {e}")
    print("  Commands will not be executed")

app = Flask(__name__)
system = SystemClass()
stt = None

# Execution settings
AUTO_EXECUTE_THRESHOLD = 0.7  # Only auto-execute if confidence >= 70%
AUTO_EXECUTE_VOICE = True     # Enable/disable auto-execution for voice
AUTO_EXECUTE_TEXT = False     # Text requires manual checkbox (default)

if STT_AVAILABLE:
    try:
        # Use "base" model - MUCH faster, still good accuracy
        stt = ArabicSTT(model_size="base")
        print("✓ Whisper model loaded successfully")
        print("  Using 'base' model for optimal speed/accuracy balance")
    except Exception as e:
        STT_AVAILABLE = False
        print(f"✗ Error loading Whisper model: {e}")
        print("  Voice features disabled")

history = []

def detect_language_from_audio_text(text):
    """
    Improved language detection - checks Arabic character density
    Returns 'ar' or 'en'
    """
    if not text or len(text.strip()) == 0:
        return 'en'
    
    # Arabic Unicode ranges
    arabic_chars = set('ابتثجحخدذرزسشصضطظعغفقكلمنهويىءآأإؤئةـَُِّْ')
    
    # Count Arabic characters
    text_cleaned = text.replace(' ', '')  # Remove spaces
    if len(text_cleaned) == 0:
        return 'en'
    
    arabic_count = sum(1 for char in text_cleaned if char in arabic_chars)
    arabic_ratio = arabic_count / len(text_cleaned)
    
    # If >30% Arabic characters, it's Arabic
    if arabic_ratio > 0.3:
        print(f"  Arabic detection: {arabic_ratio:.1%} Arabic chars -> 'ar'")
        return 'ar'
    else:
        print(f"  Arabic detection: {arabic_ratio:.1%} Arabic chars -> 'en'")
        return 'en'

def improve_arabic_text(text):
    """
    Post-process Arabic transcription using advanced corrector
    """
    # Use the advanced corrector
    corrected = correct_arabic_text(text, debug=True)
    return corrected

TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hybrid Intent Extractor with STT</title>
  <style>
    :root{--bg:#f4f7fb;--card:#ffffff;--muted:#6b7280;--accent:#2563eb;--success:#10b981;--error:#ef4444}
    body{font-family:Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; background:var(--bg); color:#111; margin:0; padding:2rem}
    .wrap{max-width:900px;margin:0 auto}
    header{display:flex;align-items:center;gap:1rem;margin-bottom:1rem;flex-wrap:wrap}
    h1{font-size:1.25rem;margin:0}
    .status{font-size:.85rem;color:var(--muted);padding:.25rem .6rem;background:#e6edf3;border-radius:6px}
    .status.active{background:#d1fae5;color:#065f46}
    .tips{background:#fffbeb;border-left:3px solid #f59e0b;padding:1rem;border-radius:8px;margin-bottom:1rem}
    .tips-title{font-weight:600;margin-bottom:.5rem;color:#92400e}
    .tips ul{margin:.5rem 0;padding-left:1.5rem}
    .tips li{margin:.25rem 0;font-size:.9rem;color:#78350f}
    .input-section{background:var(--card);padding:1.5rem;border-radius:12px;box-shadow:0 6px 18px rgba(16,24,40,0.06);margin-bottom:1rem}
    .input-tabs{display:flex;gap:.5rem;margin-bottom:1rem;border-bottom:2px solid #e6edf3}
    .tab{padding:.5rem 1rem;border:none;background:none;cursor:pointer;font-size:.95rem;color:var(--muted);border-bottom:2px solid transparent;margin-bottom:-2px;transition:all .2s}
    .tab.active{color:var(--accent);border-bottom-color:var(--accent);font-weight:500}
    .tab:hover{color:#111}
    .tab-content{display:none}
    .tab-content.active{display:block}
    form{display:flex;gap:.5rem}
    input[type=text]{flex:1;padding:.6rem .8rem;border:1px solid #e6edf3;border-radius:8px;font-size:1rem}
    input[type=submit], button{background:var(--accent);color:#fff;border:none;padding:.6rem 1rem;border-radius:8px;cursor:pointer;font-size:.95rem;transition:background .2s}
    input[type=submit]:hover, button:hover{background:#1d4ed8}
    button:disabled{background:var(--muted);cursor:not-allowed}
    .mic-controls{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}
    .mic-button{background:var(--accent);color:#fff;border:none;padding:.8rem 1.2rem;border-radius:8px;cursor:pointer;font-size:1rem;display:flex;align-items:center;gap:.5rem;transition:all .2s}
    .mic-button:hover{background:#1d4ed8;transform:scale(1.02)}
    .mic-button.recording{background:var(--error);animation:pulse 1.5s infinite}
    .mic-button:disabled{background:var(--muted);cursor:not-allowed;transform:none}
    @keyframes pulse{0%,100%{opacity:1}50%{opacity:.7}}
    .upload-area{border:2px dashed #e6edf3;border-radius:8px;padding:2rem;text-align:center;cursor:pointer;transition:all .2s}
    .upload-area:hover{border-color:var(--accent);background:#f9fafb}
    .upload-area.dragover{border-color:var(--accent);background:#eff6ff}
    .file-input{display:none}
    .processing{color:var(--muted);font-size:.9rem;padding:.5rem;text-align:center;display:none}
    .processing.show{display:block}
    a.clear{align-self:center;color:var(--muted);text-decoration:none;font-size:.9rem;padding:.5rem .8rem}
    a.clear:hover{color:#111}
    .card{background:var(--card);padding:1rem;border-radius:10px;box-shadow:0 6px 18px rgba(16,24,40,0.04);margin-top:1rem}
    pre{white-space:pre-wrap;margin:0;font-family:inherit;background:#f9fafb;padding:.5rem;border-radius:6px;font-size:.9rem}
    .history{display:flex;flex-direction:column;gap:.6rem}
    .item{display:flex;justify-content:space-between;gap:1rem;padding:.75rem;border-radius:8px;background:#fbfdff;border:1px solid #eef5ff}
    .meta{color:var(--muted);font-size:.9rem;display:flex;gap:.5rem;flex-wrap:wrap;align-items:center}
    .badge{padding:.15rem .4rem;background:#e6edf3;border-radius:4px;font-size:.8rem;font-weight:500}
    .badge.ar{background:#dbeafe;color:#1e40af}
    .badge.en{background:#dfe0e2;color:#374151}
    .badge.low-conf{background:#fef3c7;color:#92400e}
    .small{font-size:.85rem;color:var(--muted)}
    .transcription{background:#fffbeb;padding:.75rem;border-radius:6px;margin-top:.5rem;border-left:3px solid #f59e0b}
    .transcription-text{color:#92400e;font-style:italic}
    .error{background:#fef2f2;color:#991b1b;padding:.75rem;border-radius:6px;margin-top:.5rem;border-left:3px solid #dc2626}
    .result-row{display:flex;justify-content:space-between;align-items:start;margin:.5rem 0}
    .result-label{font-weight:600;min-width:100px}
    .confidence-bar{height:8px;background:#e5e7eb;border-radius:4px;overflow:hidden;width:150px}
    .confidence-fill{height:100%;background:var(--success);transition:width .3s}
    .confidence-fill.low{background:#f59e0b}
    .confidence-fill.very-low{background:#ef4444}
    .execution-pending{background:#fffbeb;border-left:3px solid #f59e0b;padding:.75rem;border-radius:8px;margin-top:.75rem}
    .execution-pending .title{font-weight:600;margin-bottom:.5rem;color:#92400e}
    .execution-actions{display:flex;gap:.5rem;margin-top:.5rem}
    .execution-actions button{padding:.5rem 1rem;border-radius:6px;border:none;cursor:pointer;font-size:.9rem}
    .btn-execute{background:#10b981;color:white}
    .btn-execute:hover{background:#059669}
    .btn-skip{background:#6b7280;color:white}
    .btn-skip:hover{background:#4b5563}
    .settings-panel{background:var(--card);padding:1rem;border-radius:10px;box-shadow:0 6px 18px rgba(16,24,40,0.04);margin-bottom:1rem}
    .settings-title{font-weight:600;margin-bottom:.75rem}
    .setting-row{display:flex;align-items:center;gap:.75rem;margin:.5rem 0;padding:.5rem;border-radius:6px}
    .setting-row:hover{background:#f9fafb}
    .setting-row label{display:flex;align-items:center;gap:.5rem;cursor:pointer;flex:1}
    .setting-row input[type=checkbox]{cursor:pointer}
    .setting-row input[type=number]{width:60px;padding:.25rem .5rem;border:1px solid #e5e7eb;border-radius:4px}
    .setting-description{font-size:.85rem;color:var(--muted);margin-left:1.5rem}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>🎤 Hybrid Intent Extractor</h1>
      <div class="status {{ 'active' if stt_available else '' }}">
        {{ 'STT Active' if stt_available else 'Text Only' }}
      </div>
    </header>

    {% if executor_available %}
    <div class="settings-panel">
      <div class="settings-title">⚙️ Execution Settings</div>
      
      <div class="setting-row">
        <label>
          <input type="checkbox" id="autoExecuteVoice" checked onchange="updateSettings()">
          <span>Auto-execute voice commands</span>
        </label>
      </div>
      <div class="setting-description">
        Voice commands will execute automatically when confidence is high
      </div>
      
      <div class="setting-row">
        <label>
          <span>Confidence threshold:</span>
          <input type="number" id="confidenceThreshold" value="70" min="0" max="100" step="5" onchange="updateSettings()">
          <span>%</span>
        </label>
      </div>
      <div class="setting-description">
        Only auto-execute if confidence is above this threshold
      </div>
      
      <div class="setting-row">
        <label>
          <input type="checkbox" id="autoExecuteText" onchange="updateSettings()">
          <span>Auto-execute text commands</span>
        </label>
      </div>
      <div class="setting-description">
        Text commands will execute automatically (bypasses checkbox)
      </div>
    </div>
    {% endif %}

    {% if stt_available %}
    <div class="tips">
      <div class="tips-title">💡 Tips for Better Accuracy</div>
      <ul>
        <li><strong>Speak clearly</strong> - Enunciate words, avoid mumbling</li>
        <li><strong>Reduce background noise</strong> - Find a quiet environment</li>
        <li><strong>Use complete phrases</strong> - "افتح يوتيوب" or "open youtube"</li>
        <li><strong>Wait for processing</strong> - Takes 2-4 seconds per recording</li>
        <li><strong>Try again if wrong</strong> - Voice recognition isn't perfect!</li>
      </ul>
    </div>
    {% endif %}

    <div class="input-section">
      <div class="input-tabs">
        <button class="tab active" onclick="switchTab('text')">✏️ Text Input</button>
        {% if stt_available %}
        <button class="tab" onclick="switchTab('voice')">🎤 Voice Input</button>
        <button class="tab" onclick="switchTab('file')">📁 Audio File</button>
        {% endif %}
      </div>

      <!-- Text Input Tab -->
      <div id="text-tab" class="tab-content active">
        <form method="post" action="/">
          <input type="text" name="query" placeholder="Enter query (English or Arabic)" autofocus required>
          {% if executor_available %}
          <label style="display:flex;align-items:center;gap:.5rem;font-size:.9rem;white-space:nowrap">
            <input type="checkbox" name="execute" value="true" checked>
            <span>Execute</span>
          </label>
          {% endif %}
          <input type="submit" value="Process">
          <a class="clear" href="/clear">Clear</a>
        </form>
      </div>

      {% if stt_available %}
      <!-- Voice Input Tab -->
      <div id="voice-tab" class="tab-content">
        <div class="mic-controls">
          <button id="micButton" class="mic-button" onclick="toggleRecording()">
            <span id="micIcon">🎤</span>
            <span id="micText">Start Recording</span>
          </button>
          <span id="recordingTime" class="small" style="display:none">0:00</span>
        </div>
        <div id="processingMsg" class="processing">
          <span>🔄 Processing audio... (2-4 seconds)</span>
        </div>
      </div>

      <!-- File Upload Tab -->
      <div id="file-tab" class="tab-content">
        <form id="uploadForm" enctype="multipart/form-data">
          <div class="upload-area" id="uploadArea">
            <p>📁 Drop audio file here or click to browse</p>
            <p class="small">Supports: MP3, WAV, M4A, OGG, WEBM</p>
            <input type="file" id="fileInput" name="audio" accept="audio/*" class="file-input">
          </div>
          <div id="fileProcessing" class="processing">
            <span>🔄 Processing file...</span>
          </div>
        </form>
      </div>
      {% endif %}
    </div>

    {% if last %}
      <div class="card">
        <h3 style="margin-top:0">Last Result</h3>
        {% if last.get('transcription') %}
        <div class="transcription">
          <div class="small" style="margin-bottom:.3rem">🎤 Transcription:</div>
          <div class="transcription-text">"{{ last['transcription'] }}"</div>
        </div>
        {% endif %}
        
        <div style="margin-top:1rem">
          <div class="result-row">
            <span class="result-label">Query:</span>
            <strong>{{ last['query'] }}</strong>
          </div>
          
          <div class="result-row">
            <span class="result-label">Language:</span>
            <span class="badge {{ last['language'] }}">{{ last['language'].upper() }}</span>
          </div>
          
          <div class="result-row">
            <span class="result-label">Intent:</span>
            <strong>{{ last['intent'] }}</strong>
          </div>
          
          <div class="result-row">
            <span class="result-label">Confidence:</span>
            <div>
              <div class="confidence-bar">
                <div class="confidence-fill {{ 'low' if last['confidence'] < 0.7 else '' }} {{ 'very-low' if last['confidence'] < 0.5 else '' }}" 
                     style="width: {{ last['confidence'] * 100 }}%"></div>
              </div>
              <span class="small">{{ '%.1f'|format(last['confidence']*100) }}%</span>
              {% if last['confidence'] < 0.7 %}
              <span class="badge low-conf">Low confidence</span>
              {% endif %}
            </div>
          </div>
          
          <div style="margin-top:.75rem">
            <div class="result-label">Entities:</div>
            <pre>{{ last['entities'] if last['entities'] else 'None' }}</pre>
          </div>
          
          {% if last.get('execution') %}
          <div style="margin-top:.75rem;padding:.75rem;border-radius:8px;{{ 'background:#d1fae5;border-left:3px solid #10b981' if last['execution']['success'] else 'background:#fef2f2;border-left:3px solid #ef4444' }}">
            <div style="font-weight:600;margin-bottom:.25rem">
              {{ '✅' if last['execution']['success'] else '❌' }} Execution
            </div>
            <div style="color:#111">{{ last['execution']['message'] }}</div>
          </div>
          {% elif last.get('execution_skipped') %}
          <div class="execution-pending">
            <div class="title">⚠️ Execution Skipped</div>
            <div>{{ last['execution_skipped']['reason'] }}</div>
            {% if last['execution_skipped']['can_retry'] %}
            <div class="execution-actions">
              <button class="btn-execute" onclick="executeCommand('{{ last['intent'] }}', {{ last['entities']|tojson }}, '{{ last['language'] }}')">
                Execute Now
              </button>
            </div>
            {% endif %}
          </div>
          {% endif %}
        </div>
        
        <div class="small" style="margin-top:.75rem;color:var(--muted)">
          ⏱️ Processing: {{ '%.0f'|format(last['timing_ms']) }} ms
        </div>
      </div>
    {% endif %}

    {% if error %}
      <div class="error">❌ {{ error }}</div>
    {% endif %}

    {% if history %}
      <div class="card" style="margin-top:1rem">
        <h3 style="margin-top:0">History ({{ history|length }})</h3>
        <div class="history">
        {% for item in history %}
          <div class="item">
            <div style="flex:1">
              <div><strong>{{ item['query'] }}</strong></div>
              <div class="meta">
                <span class="badge {{ item.get('language', 'en') }}">{{ item.get('language', 'EN').upper() }}</span>
                <span>{{ item['intent'] }}</span>
                <span>{{ '%.0f'|format(item['confidence']*100) }}%</span>
                {% if item.get('transcription') %}
                <span>🎤</span>
                {% endif %}
              </div>
            </div>
            <div class="small">{{ '%.0f'|format(item['timing_ms']) }}ms</div>
          </div>
        {% endfor %}
        </div>
      </div>
    {% endif %}
  </div>

  {% if stt_available %}
  <script>
    let mediaRecorder;
    let audioChunks = [];
    let recordingInterval;
    let recordingSeconds = 0;
    
    // Settings stored in localStorage
    let settings = {
      autoExecuteVoice: true,
      autoExecuteText: false,
      confidenceThreshold: 70
    };
    
    // Load settings on page load
    function loadSettings() {
      const saved = localStorage.getItem('executionSettings');
      if (saved) {
        settings = JSON.parse(saved);
        document.getElementById('autoExecuteVoice').checked = settings.autoExecuteVoice;
        document.getElementById('autoExecuteText').checked = settings.autoExecuteText;
        document.getElementById('confidenceThreshold').value = settings.confidenceThreshold;
      }
    }
    
    // Update settings
    function updateSettings() {
      settings.autoExecuteVoice = document.getElementById('autoExecuteVoice').checked;
      settings.autoExecuteText = document.getElementById('autoExecuteText').checked;
      settings.confidenceThreshold = parseInt(document.getElementById('confidenceThreshold').value);
      localStorage.setItem('executionSettings', JSON.stringify(settings));
      console.log('Settings updated:', settings);
    }
    
    // Execute a command via API
    async function executeCommand(intent, entities, language) {
      try {
        const response = await fetch('/api/execute_only', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({intent, entities, language})
        });
        
        if (response.ok) {
          window.location.reload();
        } else {
          alert('Execution failed');
        }
      } catch (err) {
        alert('Error: ' + err.message);
      }
    }
    
    // Load settings when page loads
    window.addEventListener('load', loadSettings);

    function switchTab(tab) {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      
      event.target.classList.add('active');
      document.getElementById(tab + '-tab').classList.add('active');
    }

    async function toggleRecording() {
      const button = document.getElementById('micButton');
      const icon = document.getElementById('micIcon');
      const text = document.getElementById('micText');
      const timeDisplay = document.getElementById('recordingTime');

      if (!mediaRecorder || mediaRecorder.state === 'inactive') {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          mediaRecorder = new MediaRecorder(stream);
          audioChunks = [];
          recordingSeconds = 0;

          mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
          };

          mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            await sendAudioToServer(audioBlob);
            stream.getTracks().forEach(track => track.stop());
            clearInterval(recordingInterval);
            timeDisplay.style.display = 'none';
          };

          mediaRecorder.start();
          button.classList.add('recording');
          icon.textContent = '⏹️';
          text.textContent = 'Stop Recording';
          timeDisplay.style.display = 'inline';

          recordingInterval = setInterval(() => {
            recordingSeconds++;
            const mins = Math.floor(recordingSeconds / 60);
            const secs = recordingSeconds % 60;
            timeDisplay.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
          }, 1000);

        } catch (err) {
          alert('Could not access microphone: ' + err.message);
        }
      } else {
        mediaRecorder.stop();
        button.classList.remove('recording');
        icon.textContent = '🎤';
        text.textContent = 'Start Recording';
      }
    }

    async function sendAudioToServer(audioBlob) {
      const processingMsg = document.getElementById('processingMsg');
      processingMsg.classList.add('show');

      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');
      formData.append('auto_execute', settings.autoExecuteVoice);
      formData.append('confidence_threshold', settings.confidenceThreshold / 100);

      try {
        const response = await fetch('/process_audio', {
          method: 'POST',
          body: formData
        });

        if (response.ok) {
          window.location.reload();
        } else {
          const error = await response.json();
          alert('Error: ' + (error.error || 'Unknown error'));
        }
      } catch (err) {
        alert('Error processing audio: ' + err.message);
      } finally {
        processingMsg.classList.remove('show');
      }
    }

    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const fileProcessing = document.getElementById('fileProcessing');

    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
      uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadArea.classList.remove('dragover');
      
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        fileInput.files = files;
        uploadFile();
      }
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        uploadFile();
      }
    });

    async function uploadFile() {
      fileProcessing.classList.add('show');
      
      const formData = new FormData();
      formData.append('audio', fileInput.files[0]);
      formData.append('auto_execute', settings.autoExecuteVoice);
      formData.append('confidence_threshold', settings.confidenceThreshold / 100);

      try {
        const response = await fetch('/process_audio', {
          method: 'POST',
          body: formData
        });

        if (response.ok) {
          window.location.reload();
        } else {
          const error = await response.json();
          alert('Error: ' + (error.error || 'Unknown error'));
        }
      } catch (err) {
        alert('Error processing file: ' + err.message);
      } finally {
        fileProcessing.classList.remove('show');
      }
    }
  </script>
  {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    last = None
    error = None
    execution_result = None
    
    if request.method == 'POST':
        q = request.form.get('query', '').strip()
        execute = request.form.get('execute', 'false') == 'true'
        
        if q:
            try:
                # Support different system APIs
                if hasattr(system, 'process_query'):
                    res = system.process_query(q)
                elif hasattr(system, 'process_query_optimized'):
                    res = system.process_query_optimized(q)
                else:
                    fn = getattr(system, 'process', None)
                    if callable(fn):
                        res = fn(q)
                    else:
                        raise RuntimeError('No supported processing method found on system')

                entry = {
                    'query': q,
                    'intent': res.get('intent'),
                    'confidence': res.get('confidence', 0.0),
                    'entities': res.get('entities', {}),
                    'language': res.get('language', 'en'),
                    'timing_ms': res.get('timing', {}).get('total_ms', 0.0)
                }
                
                # Execute command if requested
                if execute and EXECUTOR_AVAILABLE:
                    success, message = executor.execute_command(
                        entry['intent'],
                        entry['entities'],
                        entry['language']
                    )
                    execution_result = {
                        'success': success,
                        'message': message
                    }
                    entry['execution'] = execution_result
                
                history.insert(0, entry)
                last = entry
            except Exception as e:
                error = f"Error processing query: {str(e)}"

    return render_template_string(
        TEMPLATE, 
        last=last, 
        history=history, 
        error=error,
        execution_result=execution_result,
        stt_available=STT_AVAILABLE,
        executor_available=EXECUTOR_AVAILABLE
    )

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if not STT_AVAILABLE:
        return jsonify({'error': 'STT not available'}), 400
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    auto_execute = request.form.get('auto_execute', 'true').lower() == 'true'
    confidence_threshold = float(request.form.get('confidence_threshold', AUTO_EXECUTE_THRESHOLD))
    
    try:
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
            audio_file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        print("\n" + "="*50)
        print("PROCESSING AUDIO")
        print("="*50)
        
        # Step 1: First pass - quick detection to determine language
        print("Step 1: Quick language detection...")
        
        # Very fast transcription just to detect language
        quick_result = stt.transcribe_file(
            tmp_path,
            language=None,
            beam_size=1,  # Fast
            best_of=1,
            temperature=0.0
        )
        
        quick_text = quick_result['text'].strip()
        whisper_lang = quick_result.get('language', 'unknown')
        
        print(f"  Quick transcription: {quick_text[:50]}...")
        print(f"  Whisper language: {whisper_lang}")
        
        # Detect language from the quick transcription
        detected_lang = detect_language_from_audio_text(quick_text)
        print(f"  Our detection: {detected_lang}")
        
        # Step 2: High-quality transcription with correct language FORCED
        # This prevents Whisper from translating
        print(f"\nStep 2: High-quality transcription with language={detected_lang}...")
        
        result = stt.transcribe_file(
            tmp_path,
            language=detected_lang,  # FORCE the language - prevents translation
            task="transcribe",  # IMPORTANT: transcribe, NOT translate
            beam_size=5,
            best_of=5,
            temperature=0.0
        )
        
        transcription = result['text'].strip()
        print(f"  Transcribed: {transcription}")
        
        # Step 3: Correct Arabic transcription errors
        if detected_lang == 'ar':
            print("\nStep 3: Correcting Arabic transcription errors...")
            print(f"  Before correction: {transcription}")
            transcription = improve_arabic_text(transcription)
            print(f"  After correction: {transcription}")
        
        print(f"\n✓ Final transcription: {transcription}")
        print(f"✓ Final language: {detected_lang}")
        print("="*50 + "\n")
        
        # Remove temporary file
        os.unlink(tmp_path)
        
        if not transcription:
            return jsonify({'error': 'Could not transcribe audio'}), 400
        
        # Process the transcribed text through entity extraction
        if hasattr(system, 'process_query'):
            res = system.process_query(transcription)
        elif hasattr(system, 'process_query_optimized'):
            res = system.process_query_optimized(transcription)
        else:
            fn = getattr(system, 'process', None)
            if callable(fn):
                res = fn(transcription)
            else:
                return jsonify({'error': 'No processing method available'}), 500
        
        # Prepare entry
        entry = {
            'query': transcription,
            'intent': res.get('intent'),
            'confidence': res.get('confidence', 0.0),
            'entities': res.get('entities', {}),
            'language': detected_lang,
            'timing_ms': res.get('timing', {}).get('total_ms', 0.0),
            'transcription': transcription
        }
        
        # Decide whether to execute based on settings and confidence
        execution_result = None
        execution_skipped = None
        
        if EXECUTOR_AVAILABLE and auto_execute:
            confidence = entry['confidence']
            
            if confidence >= confidence_threshold:
                # High confidence - execute
                print(f"✓ Confidence {confidence:.1%} >= {confidence_threshold:.1%} - Executing command...")
                success, message = executor.execute_command(
                    entry['intent'],
                    entry['entities'],
                    detected_lang
                )
                execution_result = {
                    'success': success,
                    'message': message
                }
                print(f"Execution: {message}")
            else:
                # Low confidence - skip execution
                print(f"⚠️ Confidence {confidence:.1%} < {confidence_threshold:.1%} - Skipping execution")
                execution_skipped = {
                    'reason': f"Confidence too low ({confidence:.1%} < {confidence_threshold:.1%})",
                    'can_retry': True
                }
        elif not auto_execute:
            execution_skipped = {
                'reason': "Auto-execution disabled in settings",
                'can_retry': True
            }
        
        entry['execution'] = execution_result
        entry['execution_skipped'] = execution_skipped
        
        history.insert(0, entry)
        
        return jsonify({'success': True, 'result': entry})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        return jsonify({'error': str(e)}), 500

@app.route('/api/execute_only', methods=['POST'])
def api_execute_only():
    """Execute a command that was previously classified"""
    if not EXECUTOR_AVAILABLE:
        return jsonify({'error': 'Executor not available'}), 400
    
    data = request.get_json(force=True)
    intent = data.get('intent')
    entities = data.get('entities', {})
    language = data.get('language', 'en')
    
    if not intent:
        return jsonify({'error': 'No intent provided'}), 400
    
    try:
        success, message = executor.execute_command(intent, entities, language)
        
        # Update the last history entry
        if history:
            history[0]['execution'] = {
                'success': success,
                'message': message
            }
            if 'execution_skipped' in history[0]:
                del history[0]['execution_skipped']
        
        return jsonify({
            'success': success,
            'message': message
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/execute', methods=['POST'])
def api_execute():
    """API endpoint for executing commands"""
    data = request.get_json(force=True)
    q = data.get('query') if isinstance(data, dict) else None
    execute = data.get('execute', True)
    
    if not q:
        return jsonify({'error': 'no query provided'}), 400
    
    try:
        # Process query
        if hasattr(system, 'process_query'):
            res = system.process_query(q)
        elif hasattr(system, 'process_query_optimized'):
            res = system.process_query_optimized(q)
        else:
            fn = getattr(system, 'process', None)
            if callable(fn):
                res = fn(q)
            else:
                return jsonify({'error': 'no processing method available'}), 500

        result = {
            'query': q,
            'intent': res.get('intent'),
            'confidence': res.get('confidence', 0.0),
            'entities': res.get('entities', {}),
            'language': res.get('language', 'en'),
        }
        
        # Execute if requested
        if execute and EXECUTOR_AVAILABLE:
            success, message = executor.execute_command(
                result['intent'],
                result['entities'],
                result['language']
            )
            result['execution'] = {
                'success': success,
                'message': message
            }
        elif execute and not EXECUTOR_AVAILABLE:
            result['execution'] = {
                'success': False,
                'message': 'Command executor not available'
            }

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    """API endpoint for audio transcription only"""
    if not STT_AVAILABLE:
        return jsonify({'error': 'STT not available'}), 400
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
            audio_file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        result = stt.transcribe_file(tmp_path, language=None, beam_size=3, best_of=3)
        transcription = result['text'].strip()
        detected_lang = detect_language_from_audio_text(transcription)
        
        os.unlink(tmp_path)
        
        return jsonify({
            'transcription': transcription,
            'language': detected_lang
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear')
def clear_history():
    history.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run Flask web UI with STT for Intent Extraction')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=5000, type=int)
    parser.add_argument('--model', default='base', choices=['tiny', 'base', 'small', 'medium'],
                       help='Whisper model size (base=recommended for speed/accuracy)')
    args = parser.parse_args()
    
    print(f"Starting web UI on http://{args.host}:{args.port}")
    print(f"STT Available: {STT_AVAILABLE}")
    if STT_AVAILABLE:
        print(f"Model: {args.model}")
        print("Voice input and audio file upload enabled")
        print("\nTips for best results:")
        print("  • Speak clearly in a quiet environment")
        print("  • Use complete phrases (e.g., 'افتح يوتيوب' or 'open youtube')")
        print("  • Wait 2-4 seconds for processing")
        print("  • Check transcription if intent seems wrong")
    else:
        print("Text input only - install dependencies for voice features:")
        print("  pip install openai-whisper torch torchaudio")
    
    print("\n" + "="*60)
    app.run(host=args.host, port=args.port, debug=True)