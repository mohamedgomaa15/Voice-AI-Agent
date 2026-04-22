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
executor = CommandExecutor()


# ==================== HEALTH CHECK ====================
@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'Voice AI Agent API',
        'version': '1.0'
    }), 200


# ==================== TEXT PROCESSING ====================
@api_bp.route('/process-text', methods=['POST'])
def process_text():
    """
    Process a text command
    
    Request JSON:
    {
        "command": "open youtube",
        "execute": true  # optional, default: true
    }
    
    Response:
    {
        "intent": "open_app",
        "entities": {"app_name": "YouTube"},
        "execution_success": true,
        "execution_message": "Opened YouTube"
    }
    """
    try:
        data = request.get_json()
        if not data or 'command' not in data:
            return jsonify({'error': 'Missing required field: command'}), 400
        
        command = data.get('command', '').strip()
        should_execute = data.get('execute', True)
        
        if not command:
            return jsonify({'error': 'Command cannot be empty'}), 400
        
        # Process with pipeline
        result = agent_system_setclass_appmatch(command)
        
        response = {
            'intent': result['intent'],
            'entities': result['entity'],
        }
        
        # Execute if requested
        if should_execute:
            success, message = executor.execute_command(result['intent'], result['entity'])
            response['execution_success'] = success
            response['execution_message'] = message
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== AUDIO PROCESSING ====================
@api_bp.route('/process-audio', methods=['POST'])
def process_audio():
    """
    Process audio file
    
    Request: multipart/form-data
    - audio: audio file (WAV, MP3)
    - execute: true/false (optional, default: true)
    
    Response:
    {
        "transcription": "open youtube",
        "intent": "open_app",
        "entities": {"app_name": "YouTube"},
        "execution_success": true,
        "execution_message": "Opened YouTube"
    }
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        should_execute = request.form.get('execute', 'true').lower() == 'true'
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_path = temp_file.name
            audio_file.save(temp_path)
        
        try:
            # Transcribe
            transcription_result = stt.transcribe_file(temp_path)
            command = transcription_result['text'].strip()
            
            # Process with pipeline
            result = agent_system_setclass_appmatch(command)
            
            response = {
                'transcription': command,
                'intent': result['intent'],
                'entities': result['entity'],
            }
            
            # Execute if requested
            if should_execute:
                success, message = executor.execute_command(result['intent'], result['entity'])
                response['execution_success'] = success
                response['execution_message'] = message
            
            return jsonify(response), 200
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== CLASSIFICATION ONLY ====================
@api_bp.route('/classify', methods=['POST'])
def classify():
    """
    Classify intent without executing
    
    Request JSON:
    {
        "command": "open youtube"
    }
    
    Response:
    {
        "intent": "open_app",
        "entities": {"app_name": "YouTube"}
    }
    """
    try:
        data = request.get_json()
        if not data or 'command' not in data:
            return jsonify({'error': 'Missing required field: command'}), 400
        
        command = data.get('command', '').strip()
        if not command:
            return jsonify({'error': 'Command cannot be empty'}), 400
        
        # Process with pipeline (no execution)
        result = agent_system_setclass_appmatch(command)
        
        return jsonify({
            'intent': result['intent'],
            'entities': result['entity']
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== TRANSCRIPTION ONLY ====================
@api_bp.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Transcribe audio to text only
    
    Request: multipart/form-data
    - audio: audio file (WAV, MP3)
    
    Response:
    {
        "transcription": "open youtube"
    }
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_path = temp_file.name
            audio_file.save(temp_path)
        
        try:
            # Transcribe
            transcription_result = stt.transcribe_file(temp_path)
            command = transcription_result['text'].strip()
            
            return jsonify({
                'transcription': command
            }), 200
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== COMMAND EXECUTION ====================
@api_bp.route('/execute', methods=['POST'])
def execute():
    """
    Execute a command with given intent and entities
    
    Request JSON:
    {
        "intent": "open_app",
        "entities": {"app_name": "YouTube"}
    }
    
    Response:
    {
        "execution_success": true,
        "execution_message": "Opened YouTube"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body cannot be empty'}), 400
        
        intent = data.get('intent')
        entities = data.get('entities')
        
        if not intent:
            return jsonify({'error': 'Missing required field: intent'}), 400
        if not isinstance(entities, dict):
            return jsonify({'error': 'Entities must be a dictionary'}), 400
        
        # Execute command
        success, message = executor.execute_command(intent, entities)
        
        return jsonify({
            'execution_success': success,
            'execution_message': message
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== INFO ENDPOINT ====================
@api_bp.route('/info', methods=['GET'])
def info():
    """Get API documentation and supported intents"""
    return jsonify({
        'service': 'Voice AI Agent API',
        'version': '1.0',
        'supported_intents': [
            'open_app',
            'search',
            'open_app_and_search',
            'settings',
            'out_of_scope'
        ],
        'endpoints': {
            'POST /api/v1/process-text': 'Process text command (classify + optional execute)',
            'POST /api/v1/process-audio': 'Process audio file (transcribe + classify + optional execute)',
            'POST /api/v1/classify': 'Classify text command intent only',
            'POST /api/v1/transcribe': 'Transcribe audio to text only',
            'POST /api/v1/execute': 'Execute command with intent and entities',
            'GET /api/v1/health': 'Health check',
            'GET /api/v1/info': 'API documentation'
        }
    }), 200
