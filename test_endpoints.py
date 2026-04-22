import requests

# Health check
# response = requests.get('http://localhost:5000/api/v1/health')
# print(response.json())

# Process text
# response = requests.post(
#     'http://localhost:5000/api/v1/process-text',
#     json={'command': 'open youtube', 'execute': True}
# )
# print(response.json())

# # Classify only (no execution)
response = requests.post(
    'http://localhost:5000/api/v1/classify',
    json={'command': 'turn up the volume'}
)
print(response.json())

# # Process audio
# with open('voice_command.wav', 'rb') as f:
#     response = requests.post(
#         'http://localhost:5000/api/v1/process-audio',
#         files={'audio': f},
#         data={'execute': True}
#     )
# print(response.json())

# # Transcribe only
# with open('voice_command.wav', 'rb') as f:
#     response = requests.post(
#         'http://localhost:5000/api/v1/transcribe',
#         files={'audio': f}
#     )
# print(response.json())