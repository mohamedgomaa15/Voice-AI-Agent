# """
# Command Executor - Executes actions based on intent classification
# Supports: Opening apps, searching, controlling system settings
# """

# import subprocess
# import webbrowser
# import time
# import platform
# import os
# from typing import Dict, Optional, Tuple
# import urllib.parse

# class CommandExecutor:
#     def __init__(self):
#         self.os_type = platform.system()  # 'Windows', 'Darwin' (Mac), 'Linux'
#         print(f"Command Executor initialized for {self.os_type}")
        
#         # App launch commands by platform
#         self.app_commands = {
#             'Windows': {
#                 'youtube': 'start chrome https://www.youtube.com',
#                 'netflix': 'start chrome https://www.netflix.com',
#                 'google': 'start chrome https://www.google.com',
#                 'spotify': 'start spotify:',
#                 'instagram': 'start chrome https://www.instagram.com',
#                 'facebook': 'start chrome https://www.facebook.com',
#                 'twitter': 'start chrome https://www.twitter.com',
#                 'whatsapp': 'start chrome https://web.whatsapp.com',
#                 'tiktok': 'start chrome https://www.tiktok.com',
#                 'chrome': 'start chrome',
#                 'edge': 'start msedge',
#                 'firefox': 'start firefox',
#                 'notepad': 'start notepad',
#                 'calculator': 'start calc',
#                 'settings': 'start ms-settings:',
#                 'control panel': 'start control',
#             },
#             'Darwin': {  # macOS
#                 'youtube': 'open https://www.youtube.com',
#                 'netflix': 'open https://www.netflix.com',
#                 'google': 'open https://www.google.com',
#                 'spotify': 'open -a Spotify',
#                 'instagram': 'open https://www.instagram.com',
#                 'facebook': 'open https://www.facebook.com',
#                 'safari': 'open -a Safari',
#                 'chrome': 'open -a "Google Chrome"',
#                 'settings': 'open -a "System Preferences"',
#             },
#             'Linux': {
#                 'youtube': 'xdg-open https://www.youtube.com',
#                 'netflix': 'xdg-open https://www.netflix.com',
#                 'google': 'xdg-open https://www.google.com',
#                 'spotify': 'spotify',
#                 'instagram': 'xdg-open https://www.instagram.com',
#                 'chrome': 'google-chrome',
#                 'firefox': 'firefox',
#                 'settings': 'gnome-control-center',
#             }
#         }
        
#         # URL templates for search
#         self.search_urls = {
#             'youtube': 'https://www.youtube.com/results?search_query={}',
#             'netflix': 'https://www.netflix.com/search?q={}',
#             'google': 'https://www.google.com/search?q={}',
#             'spotify': 'https://open.spotify.com/search/{}',
#             'instagram': 'https://www.instagram.com/explore/tags/{}/',
#             'twitter': 'https://twitter.com/search?q={}',
#         }
        
#         # App name normalization (Arabic and English variations)
#         self.app_name_mapping = {
#             # English
#             'youtube': 'youtube',
#             'you tube': 'youtube',
#             'yt': 'youtube',
            
#             'netflix': 'netflix',
#             'net flix': 'netflix',
            
#             'spotify': 'spotify',
#             'spot if eye': 'spotify',
            
#             'instagram': 'instagram',
#             'insta': 'instagram',
#             'ig': 'instagram',
            
#             'facebook': 'facebook',
#             'fb': 'facebook',
            
#             'google': 'google',
#             'google chrome': 'chrome',
            
#             # Arabic
#             'يوتيوب': 'youtube',
#             'يوتوب': 'youtube',
#             'نتفليكس': 'netflix',
#             'نتفلكس': 'netflix',
#             'سبوتيفاي': 'spotify',
#             'سبوتفاي': 'spotify',
#             'انستقرام': 'instagram',
#             'انستغرام': 'instagram',
#             'انستجرام': 'instagram',
#             'فيسبوك': 'facebook',
#             'فيس بوك': 'facebook',
#             'تويتر': 'twitter',
#             'واتساب': 'whatsapp',
#             'وتساب': 'whatsapp',
#             'جوجل': 'google',
#             'كروم': 'chrome',
#         }
    
#     def normalize_app_name(self, app_name: str) -> str:
#         """Normalize app name to standard format"""
#         if not app_name:
#             return ''
        
#         app_lower = app_name.lower().strip()
#         return self.app_name_mapping.get(app_lower, app_lower)
    
#     def execute_command(self, intent: str, entities: Dict, language: str = 'en') -> Tuple[bool, str]:
#         """
#         Execute command based on intent and entities
        
#         Returns:
#             (success: bool, message: str)
#         """
#         try:
#             if intent == 'open_app':
#                 return self._open_app(entities)
            
#             elif intent == 'search':
#                 return self._search(entities)
            
#             elif intent == 'open_app_and_search':
#                 return self._open_app_and_search(entities)
            
#             elif intent == 'play_media':
#                 return self._play_media(entities)
            
#             elif intent == 'settings':
#                 return self._execute_settings(entities)
            
#             elif intent == 'out_of_scope':
#                 return False, "Command not recognized. Please try again."
            
#             else:
#                 return False, f"Intent '{intent}' not supported yet"
                
#         except Exception as e:
#             return False, f"Error executing command: {str(e)}"
    
#     def _open_app(self, entities: Dict) -> Tuple[bool, str]:
#         """Open an application"""
#         app_name = entities.get('app_name', '')
        
#         if not app_name:
#             return False, "No app name specified"
        
#         # Normalize app name
#         app_normalized = self.normalize_app_name(app_name)
        
#         # Get command for this OS
#         commands = self.app_commands.get(self.os_type, {})
#         command = commands.get(app_normalized)
        
#         if not command:
#             # Try opening as URL in default browser
#             if app_normalized in self.search_urls:
#                 url = self.search_urls[app_normalized].format('')
#                 webbrowser.open(url)
#                 return True, f"Opened {app_name} in browser"
#             else:
#                 return False, f"Don't know how to open '{app_name}' on {self.os_type}"
        
#         # Execute command
#         try:
#             if self.os_type == 'Windows':
#                 subprocess.Popen(command, shell=True)
#             else:
#                 subprocess.Popen(command.split())
            
#             return True, f"Opened {app_name}"
        
#         except Exception as e:
#             return False, f"Failed to open {app_name}: {str(e)}"
    
#     def _search(self, entities: Dict) -> Tuple[bool, str]:
#         """Perform a search (defaults to Google if no app specified)"""
#         query = entities.get('search_query', '')
#         app_name = entities.get('app_name', 'google')
        
#         if not query:
#             return False, "No search query specified"
        
#         # Normalize app name
#         app_normalized = self.normalize_app_name(app_name)
        
#         # Get search URL
#         search_template = self.search_urls.get(app_normalized, self.search_urls['google'])
        
#         # Encode query for URL
#         encoded_query = urllib.parse.quote(query)
#         url = search_template.format(encoded_query)
        
#         # Open in browser
#         webbrowser.open(url)
        
#         return True, f"Searching for '{query}' on {app_name}"
    
#     def _open_app_and_search(self, entities: Dict) -> Tuple[bool, str]:
#         """Open app and perform search"""
#         app_name = entities.get('app_name', '')
#         query = entities.get('search_query', '')
        
#         if not app_name:
#             return False, "No app name specified"
        
#         if not query:
#             # Just open the app
#             return self._open_app(entities)
        
#         # Normalize app name
#         app_normalized = self.normalize_app_name(app_name)
        
#         # Get search URL
#         search_template = self.search_urls.get(app_normalized)
        
#         if search_template:
#             encoded_query = urllib.parse.quote(query)
#             url = search_template.format(encoded_query)
#             webbrowser.open(url)
#             return True, f"Opened {app_name} and searching for '{query}'"
#         else:
#             # Open app first, then search
#             success, msg = self._open_app(entities)
#             if success:
#                 time.sleep(2)  # Wait for app to open
#                 # Try to search (platform-specific)
#                 return True, f"Opened {app_name}. Please search for '{query}' manually."
#             return success, msg
    
#     def _play_media(self, entities: Dict) -> Tuple[bool, str]:
#         """Play media content"""
#         title = entities.get('title', '')
#         platform = entities.get('platform', 'youtube')
        
#         if not title:
#             return False, "No media title specified"
        
#         # Normalize platform
#         platform_normalized = self.normalize_app_name(platform)
        
#         # Search for the media
#         search_template = self.search_urls.get(platform_normalized, self.search_urls['youtube'])
#         encoded_query = urllib.parse.quote(title)
#         url = search_template.format(encoded_query)
        
#         webbrowser.open(url)
        
#         return True, f"Searching for '{title}' on {platform}"
    
#     def _execute_settings(self, entities: Dict) -> Tuple[bool, str]:
#         """Execute system settings commands"""
#         action = entities.get('settings_action', '')
#         parameter = entities.get('parameter', '')
        
#         if not action:
#             # Just open settings
#             return self._open_settings()
        
#         # Volume control
#         if 'volume' in action:
#             return self._control_volume(action, parameter)
        
#         # Brightness control
#         elif 'brightness' in action:
#             return self._control_brightness(action, parameter)
        
#         # Mute
#         elif action == 'mute':
#             return self._mute_audio()
        
#         # Channel change (for TV control - would need additional hardware)
#         elif 'channel' in action:
#             return False, "TV control not implemented. Requires smart TV integration."
        
#         else:
#             return self._open_settings()
    
#     def _open_settings(self) -> Tuple[bool, str]:
#         """Open system settings"""
#         try:
#             if self.os_type == 'Windows':
#                 subprocess.Popen('start ms-settings:', shell=True)
#             elif self.os_type == 'Darwin':
#                 subprocess.Popen(['open', '-a', 'System Preferences'])
#             else:
#                 subprocess.Popen(['gnome-control-center'])
            
#             return True, "Opened system settings"
#         except Exception as e:
#             return False, f"Failed to open settings: {str(e)}"
    
#     def _control_volume(self, action: str, parameter: str) -> Tuple[bool, str]:
#         """Control system volume"""
#         try:
#             if self.os_type == 'Windows':
#                 # Use nircmd for Windows (requires nircmd.exe)
#                 # Or use pycaw library
#                 try:
#                         from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
#                         from comtypes import CLSCTX_ALL
#                         from ctypes import cast, POINTER

#                         # Try several strategies to obtain a device that supports Activate()
#                         device = None
#                         try:
#                             # Preferred: GetSpeakers() (common pycaw helper)
#                             dev = AudioUtilities.GetSpeakers()
#                             if hasattr(dev, 'Activate'):
#                                 device = dev
#                         except Exception:
#                             device = None

#                         if device is None:
#                             # Try enumerating all devices and pick one that supports Activate
#                             try:
#                                 all_devices = AudioUtilities.GetAllDevices()
#                                 for d in all_devices:
#                                     if hasattr(d, 'Activate'):
#                                         device = d
#                                         break
#                             except Exception:
#                                 device = None

#                         if device is None:
#                             raise AttributeError("No audio device with Activate() found via pycaw")

#                         interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
#                         volume = cast(interface, POINTER(IAudioEndpointVolume))

#                         current = volume.GetMasterVolumeLevelScalar()

#                         # Determine direction from action string (supports 'volume_up' / 'up')
#                         if 'up' in action or 'increase' in action:
#                             new_volume = min(current + 0.1, 1.0)
#                             volume.SetMasterVolumeLevelScalar(new_volume, None)
#                             return True, f"Volume increased to {int(new_volume * 100)}%"

#                         elif 'down' in action or 'decrease' in action:
#                             new_volume = max(current - 0.1, 0.0)
#                             volume.SetMasterVolumeLevelScalar(new_volume, None)
#                             return True, f"Volume decreased to {int(new_volume * 100)}%"
                
#                 except ImportError:
#                     return False, "Volume control requires 'pycaw' library: pip install pycaw"
            
#             elif self.os_type == 'Darwin':
#                 # macOS
#                 if 'up' in action:
#                     subprocess.run(['osascript', '-e', 'set volume output volume (output volume of (get volume settings) + 10)'])
#                     return True, "Volume increased"
#                 elif 'down' in action:
#                     subprocess.run(['osascript', '-e', 'set volume output volume (output volume of (get volume settings) - 10)'])
#                     return True, "Volume decreased"
            
#             else:
#                 # Linux
#                 if 'up' in action:
#                     subprocess.run(['amixer', 'set', 'Master', '10%+'])
#                     return True, "Volume increased"
#                 elif 'down' in action:
#                     subprocess.run(['amixer', 'set', 'Master', '10%-'])
#                     return True, "Volume decreased"
            
#             return False, "Volume control not available"
        
#         except Exception as e:
#             return False, f"Volume control error: {str(e)}"
    
#     def _mute_audio(self) -> Tuple[bool, str]:
#         """Mute/unmute system audio"""
#         try:
#             if self.os_type == 'Windows':
#                 try:
#                     from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
#                     from comtypes import CLSCTX_ALL
#                     from ctypes import cast, POINTER
                    
#                     devices = AudioUtilities.GetSpeakers()
#                     interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
#                     volume = cast(interface, POINTER(IAudioEndpointVolume))
                    
#                     is_muted = volume.GetMute()
#                     volume.SetMute(not is_muted, None)
                    
#                     return True, "Unmuted" if is_muted else "Muted"
                
#                 except ImportError:
#                     return False, "Mute control requires 'pycaw' library: pip install pycaw"
            
#             elif self.os_type == 'Darwin':
#                 subprocess.run(['osascript', '-e', 'set volume with output muted'])
#                 return True, "Audio toggled"
            
#             else:
#                 subprocess.run(['amixer', 'set', 'Master', 'toggle'])
#                 return True, "Audio toggled"
        
#         except Exception as e:
#             return False, f"Mute error: {str(e)}"
    
#     def _control_brightness(self, action: str, parameter: str) -> Tuple[bool, str]:
#         """Control screen brightness"""
#         # Brightness control is platform and hardware specific
#         # This is a placeholder - full implementation would need screen control libraries
        
#         if self.os_type == 'Windows':
#             return False, "Brightness control requires WMI and hardware support"
#         elif self.os_type == 'Darwin':
#             try:
#                 if 'up' in action:
#                     subprocess.run(['brightness', '0.1'])  # Requires brightness utility
#                     return True, "Brightness increased"
#                 elif 'down' in action:
#                     subprocess.run(['brightness', '-0.1'])
#                     return True, "Brightness decreased"
#             except:
#                 return False, "Brightness control not available"
#         else:
#             return False, "Brightness control not implemented for Linux"


# # Test the executor
# if __name__ == "__main__":
#     executor = CommandExecutor()
    
#     # Test cases
#     test_cases = [
#         {
#             'intent': 'open_app',
#             'entities': {'app_name': 'YouTube'},
#             'description': 'Open YouTube'
#         },
#         {
#             'intent': 'search',
#             'entities': {'search_query': 'artificial intelligence', 'app_name': 'google'},
#             'description': 'Search Google for AI'
#         },
#         {
#             'intent': 'open_app_and_search',
#             'entities': {'app_name': 'YouTube', 'search_query': 'machine learning'},
#             'description': 'Open YouTube and search'
#         },
#         {
#             'intent': 'play_media',
#             'entities': {'title': 'Inception', 'platform': 'Netflix'},
#             'description': 'Play movie on Netflix'
#         },
#         {
#             'intent': 'settings',
#             'entities': {'settings_action': 'volume_up'},
#             'description': 'Increase volume'
#         },
#     ]
    
#     print("="*70)
#     print("COMMAND EXECUTOR TEST")
#     print("="*70)
    
#     for i, test in enumerate(test_cases, 1):
#         print(f"\nTest {i}: {test['description']}")
#         print(f"Intent: {test['intent']}")
#         print(f"Entities: {test['entities']}")
        
#         success, message = executor.execute_command(
#             test['intent'],
#             test['entities']
#         )
        
#         status = "✓" if success else "✗"
#         print(f"{status} Result: {message}")
        
#         if i < len(test_cases):
#             time.sleep(1)

"""
Command Executor - Executes actions based on intent classification
Now supports dynamic local apps from JSON file
"""

import subprocess
import webbrowser
import time
import platform
import json
import os
from typing import Dict, Tuple
import urllib.parse


class CommandExecutor:
    def __init__(self, json_path="./data/clean_apps.json"):
        self.os_type = platform.system()
        print(f"Command Executor initialized for {self.os_type}")

        # Load apps from JSON
        self.apps_data = self._load_apps(json_path)

        # Build alias → command mapping
        self.app_commands = self._build_app_mapping(self.apps_data)

        # Search URLs
        self.search_urls = {
            'google': 'https://www.google.com/search?q={}',
            'youtube': 'https://www.youtube.com/results?search_query={}'
        }

    # =========================
    # JSON HANDLING
    # =========================

    def _load_apps(self, file_path: str):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return []

    def _build_app_mapping(self, apps):
        mapping = {}

        for app in apps:
            path = app.get("path")
            name = app.get("name", "").lower()

            if not path:
                continue

            # Map main name
            if name:
                mapping[name] = path

            # Map aliases
            for alias in app.get("aliases", []):
                mapping[alias.lower()] = path

        return mapping

    # =========================
    # MAIN EXECUTION
    # =========================

    def execute_command(self, intent: str, entities: Dict) -> Tuple[bool, str]:
        try:
            if intent == 'open_app':
                return self._open_app(entities)

            elif intent == 'search':
                return self._search(entities)

            elif intent == 'open_app_and_search':
                return self._open_app_and_search(entities)

            else:
                return False, f"Intent '{intent}' not supported"

        except Exception as e:
            return False, f"Execution error: {str(e)}"

    # =========================
    # APP EXECUTION
    # =========================

    def _find_best_match(self, app_name: str):
        app_name = app_name.lower()
        
        # Exact match
        if app_name in self.app_commands:
            return self.app_commands[app_name]
        
        # Partial match (try to find any alias containing the user input)
        best_match = None
        best_score = 0
        
        for key, cmd in self.app_commands.items():
            # Check if user input is contained in alias OR alias is contained in user input
            if app_name in key:
                # Longer matches are better
                score = len(key)
                if score > best_score:
                    best_score = score
                    best_match = cmd
            elif key in app_name:
                score = len(key)
                if score > best_score:
                    best_score = score
                    best_match = cmd
        
        return best_match
    
    def _open_app(self, entities: Dict) -> Tuple[bool, str]:
        app_name = entities.get('app_name', '')

        if not app_name:
            return False, "No app name provided"

        command = self._find_best_match(app_name)

        if not command:
            return False, f"App '{app_name}' not found"

        try:
            if command.endswith('.desktop'):
                # Use gtk-launch or parse the desktop file
                desktop_name = os.path.basename(command)
                subprocess.Popen(['gtk-launch', desktop_name], shell=False)
            else:
                # Regular executable
                subprocess.Popen([command], shell=False)
                return True, f"Opened {app_name}"
        except Exception as e:
            return False, f"Failed to open {app_name}: {str(e)}"

    # =========================
    # SEARCH
    # =========================

    def _search(self, entities: Dict) -> Tuple[bool, str]:
        query = entities.get('search_query', '')
        app_name = entities.get('app_name', 'google')

        if not query:
            return False, "No search query"

        app_name = app_name.lower()
        template = self.search_urls.get(app_name, self.search_urls['google'])

        url = template.format(urllib.parse.quote(query))
        webbrowser.open(url)

        return True, f"Searching for '{query}'"

    def _open_app_and_search(self, entities: Dict) -> Tuple[bool, str]:
        app_name = entities.get('app_name', '')
        query = entities.get('search_query', '')

        if not app_name:
            return False, "No app name"

        # If it's a web app → direct search
        if app_name.lower() in self.search_urls:
            return self._search(entities)

        # Otherwise → open local app
        success, msg = self._open_app(entities)

        if success and query:
            time.sleep(2)
            return True, f"Opened {app_name}. Now search manually for '{query}'"

        return success, msg


# =========================
# TEST
# =========================

if __name__ == "__main__":
    executor = CommandExecutor()

    tests = [
        {"intent": "open_app", "entities": {"app_name": "vscode"}},
        # {"intent": "open_app", "entities": {"app_name": "telegram"}},
        # {"intent": "search", "entities": {"search_query": "machine learning"}},
        # {"intent": "open_app_and_search", "entities": {"app_name": "youtube", "search_query": "AI"}},
    ]

    for t in tests:
        print("\n---")
        print(t)
        success, msg = executor.execute_command(t["intent"], t["entities"])
        print("Result:", success, msg)