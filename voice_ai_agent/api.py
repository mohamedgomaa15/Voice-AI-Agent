"""
REST API for Voice AI Agent
Provides endpoints for other developers to integrate voice/text command processing
"""

from flask import Blueprint, request, jsonify
from voice_ai_agent.pipeline import agent_system_setclass_appmatch
from voice_ai_agent.command_executer import CommandExecutor
from voice_ai_agent.english_stt_optimized2 import EnglishSTT
import tempfile
import os

api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# Initialize components
stt = EnglishSTT(device="cpu")

# Get the correct path to clean_apps.json
def get_apps_json_path():
    possible_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "clean_apps.json"),
        os.path.join(os.path.dirname(__file__), "data", "clean_apps.json"),
        os.path.join(os.getcwd(), "data", "clean_apps.json"),
        "./data/clean_apps.json",
        "../data/clean_apps.json",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return "./data/clean_apps.json"

executor = CommandExecutor(json_path=get_apps_json_path())


# ==================== MAIN API ENDPOINT ====================
@api_bp.route('/process', methods=['POST'])
def process_command():
    """
    Process a command (text or audio) and return structured response.
    
    For text: 
    {
        "command": "open visual studio code",
        "type": "text"
    }
    
    For audio (multipart/form-data):
    - audio: audio file
    - type: "audio"
    
    Response:
    {
        "command": "open_app",
        "app_name": "Visual Studio Code",  # if open_app
        "search_query": "cat videos",      # if open_app_and_search
        "settings_action": "volume_up",    # if settings
        "message": "Out of scope message"  # if out_of_scope
    }
    """
    try:
        # Check if it's text or audio
        if request.is_json:
            data = request.get_json()
            command_type = data.get('type', 'text')
            
            if command_type == 'text':
                command = data.get('command', '').strip()
                if not command:
                    return jsonify({'error': 'No command provided'}), 400
                
                return _process_text_command(command)
            else:
                return jsonify({'error': 'Invalid type. Use "text" or send audio file'}), 400
        
        # Audio processing
        elif 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            return _process_audio_command(audio_file)
        
        else:
            return jsonify({'error': 'No command or audio file provided'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _process_text_command(command):
    """Process text command and return structured response"""
    # Process with agent
    result = agent_system_setclass_appmatch(command)
    
    # Build response based on intent
    response = _build_response(result['intent'], result['entity'])
    
    return jsonify(response), 200


def _process_audio_command(audio_file):
    """Process audio file and return structured response"""
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_path = temp_file.name
        audio_file.save(temp_path)
    
    try:
        # Transcribe
        transcription_result = stt.transcribe_file(temp_path)
        command = transcription_result['text'].strip()
        
        if not command:
            return jsonify({'error': 'No speech detected'}), 400
        
        # Process with agent
        result = agent_system_setclass_appmatch(command)
        
        # Build response
        response = _build_response(result['intent'], result['entity'])
        response['transcription'] = command  # Add transcription for reference
        
        return jsonify(response), 200
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _build_response(intent: str, entities: dict) -> dict:
    """Build structured response based on intent and entities"""
    
    if intent == 'open_app':
        app_name = entities.get('app_name', 'unknown')
        return {
            'command': 'open_app',
            'app_name': app_name
        }
    
    elif intent == 'open_app_and_search':
        app_name = entities.get('app_name', 'unknown')
        search_query = entities.get('search_query', '')
        return {
            'command': 'open_app_and_search',
            'app_name': app_name,
            'search_query': search_query
        }
    
    elif intent == 'search':
        search_query = entities.get('search_query', '')
        return {
            'command': 'search',
            'search_query': search_query
        }
    
    elif intent == 'settings':
        settings_action = entities.get('settings_action', 'unknown')
        return {
            'command': 'settings',
            'settings_action': settings_action
        }
    
    elif intent == 'out_of_scope':
        message = entities.get('message', 'I can help with searching for content, opening applications, and control settings.')
        return {
            'command': 'out_of_scope',
            'message': message
        }
    
    else:
        return {
            'command': 'unknown',
            'message': f'Unknown intent: {intent}'
        }


# ==================== LEGACY ENDPOINTS (for backward compatibility) ====================

@api_bp.route('/process-text', methods=['POST'])
def process_text():
    """Legacy endpoint - use /process instead"""
    try:
        data = request.get_json()
        if not data or 'command' not in data:
            return jsonify({'error': 'Missing required field: command'}), 400
        
        command = data.get('command', '').strip()
        should_execute = data.get('execute', False)
        
        if not command:
            return jsonify({'error': 'Command cannot be empty'}), 400
        
        # Process with pipeline
        result = agent_system_setclass_appmatch(command)
        
        response = _build_response(result['intent'], result['entity'])
        
        # Execute if requested
        if should_execute:
            success, message = executor.execute_command(result['intent'], result['entity'])
            response['execution_success'] = success
            response['execution_message'] = message
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/classify', methods=['POST'])
def classify():
    """Classify intent without executing"""
    try:
        data = request.get_json()
        if not data or 'command' not in data:
            return jsonify({'error': 'Missing required field: command'}), 400
        
        command = data.get('command', '').strip()
        if not command:
            return jsonify({'error': 'Command cannot be empty'}), 400
        
        result = agent_system_setclass_appmatch(command)
        response = _build_response(result['intent'], result['entity'])
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'Voice AI Agent API',
        'version': '2.0'
    }), 200


@api_bp.route('/info', methods=['GET'])
def info():
    """Get API documentation"""
    return jsonify({
        'service': 'Voice AI Agent API',
        'version': '2.0',
        'endpoints': {
            'POST /api/v1/process': 'Main endpoint - accepts text or audio',
            'POST /api/v1/process-text': 'Legacy - text only',
            'POST /api/v1/classify': 'Classify only (no execution)',
            'GET /api/v1/health': 'Health check',
            'GET /api/v1/info': 'API documentation'
        },
        'request_examples': {
            'text': {
                'method': 'POST',
                'content_type': 'application/json',
                'body': {
                    'type': 'text',
                    'command': 'open visual studio code'
                }
            },
            'audio': {
                'method': 'POST',
                'content_type': 'multipart/form-data',
                'body': {
                    'audio': '<file>',
                    'type': 'audio'
                }
            }
        },
        'response_formats': {
            'open_app': {
                'command': 'open_app',
                'app_name': 'Visual Studio Code'
            },
            'open_app_and_search': {
                'command': 'open_app_and_search',
                'app_name': 'YouTube',
                'search_query': 'cat videos'
            },
            'search': {
                'command': 'search',
                'search_query': 'machine learning'
            },
            'settings': {
                'command': 'settings',
                'settings_action': 'volume_up'
            },
            'out_of_scope': {
                'command': 'out_of_scope',
                'message': 'I can help with...'
            }
        }
    }), 200