#!/opt/ai-env/bin/python3
import ollama
import subprocess
import shlex
import os
import requests
import json
from tool_registry import get_tool_for_command, load_awareness

OLLAMA_BASE_URL = "http://127.0.0.1:11434/api"

import shlex

def clean_ai_response(ai_response):
    """
    Extracts valid shell commands from the AI response.
    Ensures sequential execution and handles 'echo' safely.
    """

    # List of commands allowed for execution
    allowed_commands = [
        "ls", "cd", "pwd", "mkdir", "touch", "rm", "cp", "mv", 
        "cat", "echo", "git", "npm", "pip"
    ]

    # If the response does not start with '!', it's not a command
    if not ai_response.strip().startswith("!"):
        return []

    # Remove leading '!' and tool prefixes like '!file_manager'
    command_text = ai_response.lstrip("!").strip()
    tool_prefixes = ["file_manager", "web_search", "system_control"]
    for prefix in tool_prefixes:
        command_text = command_text.replace(prefix, "").strip()

    # Ensure commands are executed in sequence
    commands = command_text.split("&&")  # Split at '&&' for sequential execution
    cleaned_commands = []

    for cmd in commands:
        cmd = cmd.strip()
        
        # If there's 'echo' with redirection (">"), keep the whole command intact
        if "echo" in cmd and ">" in cmd:
            cleaned_commands.append(cmd)
            continue

        # Safely tokenize other commands
        try:
            parts = shlex.split(cmd)
        except ValueError:
            continue  # Skip invalid commands

        if not parts:
            continue

        # Allow only known commands
        if parts[0] in allowed_commands:
            cleaned_commands.append(cmd)

    return cleaned_commands



def execute_command(ai_response):
    """Executes valid commands (one by one) from the AI response."""
    commands = clean_ai_response(ai_response)

    if not commands:
        return "No valid commands found."

    results = []
    for cmd in commands:
        # Special handling for 'cd' because changing directory in a subprocess won't affect our main process.
        if cmd.startswith("cd "):
            try:
                target_dir = cmd.split(" ", 1)[1].strip()
                os.chdir(target_dir)
                results.append(f"Changed directory to: {os.getcwd()}")
            except Exception as e:
                results.append(f"Error changing directory: {str(e)}")
        else:
            try:
                print(f"\nExecuting: {cmd}")
                result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                if result.returncode == 0:
                    # Combine stdout + stderr for clarity
                    output = result.stdout.strip()
                    if result.stderr.strip():
                        output += ("\n" + result.stderr.strip())
                    results.append(output or f"Command '{cmd}' executed successfully.")
                else:
                    results.append(f"Error:\n{result.stderr.strip()}")
            except Exception as e:
                results.append(f"Execution error: {str(e)}")

    return "\n".join(filter(None, results))

def call_ollama(model_name: str, prompt: str, stream_response=True):
    """
    Sends the user prompt to Ollama for completion.
    The system prompt includes references to the Awareness data (tools).
    """
    url = f"{OLLAMA_BASE_URL}/chat"

    # Load tool awareness dynamically
    awareness_data = load_awareness()
    tool_info = "\n".join([
        f"{tool}: {data['description']}, Commands: {', '.join(data['commands'])}"
        for tool, data in awareness_data["tools"].items()
    ])

    system_prompt = (
        "You are an AI OS assistant with built-in tool awareness. "
        "Before responding, check the awareness system to see if the user command matches any registered tools.\n"
        "Awareness System Data:\n"
        f"{tool_info}\n\n"
        "If a command is found, **ONLY** respond in this format:\n"
        "'!command_here'\n\n"
        "If no matching tool exists, respond normally in text."
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "stream": stream_response
    }

    response = requests.post(url, json=payload, stream=stream_response)
    response.raise_for_status()

    full_text = []
    if stream_response:
        print("AI:", end=" ", flush=True)
        for line in response.iter_lines(decode_unicode=True):
            if line.strip():
                try:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        print(content, end="", flush=True)
                        full_text.append(content)
                except json.JSONDecodeError:
                    pass
        print()
    else:
        # If not streaming, we just read the entire response once
        full_text.append(response.json().get("message", {}).get("content", ""))

    # Return the entire text that was streamed or fetched
    return "".join(full_text).strip()

def main():
    print("AI Shell Active. Type your command (or 'exit'/'quit' to stop).")

    while True:
        user_input = input(">> ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Shutting down AI Shell...")
            break

        # Get the AI's response
        ai_response = call_ollama(
            model_name="llama3.2", 
            prompt=user_input, 
            stream_response=True
        )

        # If the AI responded with a command (starts with '!')
        if ai_response.startswith("!"):
            output = execute_command(ai_response)
            print(output)

if __name__ == "__main__":
    main()
