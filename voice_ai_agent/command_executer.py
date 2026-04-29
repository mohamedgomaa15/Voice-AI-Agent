

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