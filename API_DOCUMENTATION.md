# Voice AI Agent API Documentation

Your Voice AI Agent now provides a REST API that other developers can integrate into their projects.

## Base URL
```
http://localhost:5000/api/v1
```

## Authentication
Currently no authentication required (can be added later with API keys/tokens).

---

## Endpoints

### 1. Health Check
**GET** `/health`

Check if the API is running.

**Response:**
```json
{
  "status": "ok",
  "service": "Voice AI Agent API",
  "version": "1.0"
}
```

---

### 2. Process Text Command
**POST** `/process-text`

Process a text command with optional execution.

**Request:**
```json
{
  "command": "open youtube",
  "execute": true
}
```

**Parameters:**
- `command` (required): The voice command as text
- `execute` (optional, default: true): Whether to execute the command or just classify it

**Response:**
```json
{
  "intent": "open_app",
  "entities": {
    "app_name": "YouTube"
  },
  "execution_success": true,
  "execution_message": "Opened YouTube"
}
```

**Example cURL:**
```bash
curl -X POST http://localhost:5000/api/v1/process-text \
  -H "Content-Type: application/json" \
  -d '{"command": "open youtube", "execute": true}'
```

**Example Python:**
```python
import requests

response = requests.post(
    'http://localhost:5000/api/v1/process-text',
    json={'command': 'open youtube', 'execute': True}
)
print(response.json())
```

---

### 3. Process Audio File
**POST** `/process-audio`

Process an audio file: transcribe → classify → execute.

**Request:**
- Content-Type: `multipart/form-data`
- `audio`: Audio file (WAV, MP3, etc.)
- `execute` (optional, default: true): Whether to execute after classification

**Response:**
```json
{
  "transcription": "open youtube",
  "intent": "open_app",
  "entities": {
    "app_name": "YouTube"
  },
  "execution_success": true,
  "execution_message": "Opened YouTube"
}
```

**Example cURL:**
```bash
curl -X POST http://localhost:5000/api/v1/process-audio \
  -F "audio=@recording.wav" \
  -F "execute=true"
```

**Example Python:**
```python
import requests

with open('recording.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/v1/process-audio',
        files={'audio': f},
        data={'execute': 'true'}
    )
print(response.json())
```

---

### 4. Classify Command (No Execution)
**POST** `/classify`

Only classify the intent, don't execute.

**Request:**
```json
{
  "command": "open youtube"
}
```

**Response:**
```json
{
  "intent": "open_app",
  "entities": {
    "app_name": "YouTube"
  }
}
```

**Example cURL:**
```bash
curl -X POST http://localhost:5000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"command": "open youtube"}'
```

---

### 5. Transcribe Audio Only
**POST** `/transcribe`

Only transcribe audio to text, no classification.

**Request:**
- Content-Type: `multipart/form-data`
- `audio`: Audio file

**Response:**
```json
{
  "transcription": "open youtube"
}
```

**Example cURL:**
```bash
curl -X POST http://localhost:5000/api/v1/transcribe \
  -F "audio=@recording.wav"
```

---

### 6. Execute Command
**POST** `/execute`

Execute a command given intent and entities (useful if you handle classification separately).

**Request:**
```json
{
  "intent": "open_app",
  "entities": {
    "app_name": "YouTube"
  }
}
```

**Response:**
```json
{
  "execution_success": true,
  "execution_message": "Opened YouTube"
}
```

**Example cURL:**
```bash
curl -X POST http://localhost:5000/api/v1/execute \
  -H "Content-Type: application/json" \
  -d '{
    "intent": "open_app",
    "entities": {"app_name": "YouTube"}
  }'
```

---

### 7. Get API Info
**GET** `/info`

Get documentation about supported intents and all available endpoints.

**Response:**
```json
{
  "service": "Voice AI Agent API",
  "version": "1.0",
  "supported_intents": [
    "open_app",
    "search",
    "open_app_and_search",
    "settings",
    "out_of_scope"
  ],
  "endpoints": { ... }
}
```

---

## Supported Intents

- **`open_app`**: Open an application (e.g., YouTube, Spotify, Chrome)
- **`search`**: Search for content on Google, YouTube, Netflix, etc.
- **`open_app_and_search`**: Open app and search (e.g., "open YouTube and search for AI")
- **`settings`**: Control system settings (volume, brightness, etc.)
- **`out_of_scope`**: Command not recognized

---

## Error Handling

All endpoints return errors in the following format:

```json
{
  "error": "Error message describing what went wrong"
}
```

**HTTP Status Codes:**
- `200`: Success
- `400`: Bad request (missing or invalid parameters)
- `500`: Server error

---

## Integration Examples

### Python Client
```python
import requests

class VoiceAIClient:
    BASE_URL = 'http://localhost:5000/api/v1'
    
    def __init__(self):
        self.session = requests.Session()
    
    def process_text(self, command, execute=True):
        """Process text command"""
        response = self.session.post(
            f'{self.BASE_URL}/process-text',
            json={'command': command, 'execute': execute}
        )
        return response.json()
    
    def process_audio(self, audio_path, execute=True):
        """Process audio file"""
        with open(audio_path, 'rb') as f:
            response = self.session.post(
                f'{self.BASE_URL}/process-audio',
                files={'audio': f},
                data={'execute': execute}
            )
        return response.json()
    
    def classify(self, command):
        """Classify only"""
        response = self.session.post(
            f'{self.BASE_URL}/classify',
            json={'command': command}
        )
        return response.json()

# Usage
client = VoiceAIClient()
result = client.process_text('open youtube')
print(result)
```

### JavaScript/Node.js
```javascript
const API_BASE = 'http://localhost:5000/api/v1';

async function processText(command, execute = true) {
  const response = await fetch(`${API_BASE}/process-text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ command, execute })
  });
  return response.json();
}

async function processAudio(audioFile, execute = true) {
  const formData = new FormData();
  formData.append('audio', audioFile);
  formData.append('execute', execute);
  
  const response = await fetch(`${API_BASE}/process-audio`, {
    method: 'POST',
    body: formData
  });
  return response.json();
}

// Usage
processText('open youtube').then(result => console.log(result));
```

### cURL Examples

**Process text:**
```bash
curl -X POST http://localhost:5000/api/v1/process-text \
  -H "Content-Type: application/json" \
  -d '{"command": "search for machine learning", "execute": false}'
```

**Process audio:**
```bash
curl -X POST http://localhost:5000/api/v1/process-audio \
  -F "audio=@audio.wav"
```

**Get API info:**
```bash
curl http://localhost:5000/api/v1/info
```

---

## Running the API Server

```bash
# From project root
python voice_ai_agent/web_app.py

# Or
python web_app.py
```

The API will be available at `http://localhost:5000/api/v1`

---

## Future Enhancements

- [ ] API authentication (API keys)
- [ ] Rate limiting
- [ ] Response caching
- [ ] WebSocket support for real-time streaming
- [ ] Batch processing endpoint
- [ ] Webhook support for async operations
- [ ] Detailed logging and monitoring
