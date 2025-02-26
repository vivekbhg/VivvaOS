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
    Cleans AI-generated shell commands to remove tool prefixes and ensures proper execution.
    """

    # List of commands allowed for execution
    allowed_commands = [
        "ls", "cd", "pwd", "mkdir", "touch", "rm", "cp", "mv", 
        "cat", "echo", "git", "npm", "pip"
    ]

    # Remove leading '!'
    command_text = ai_response.lstrip("!").strip()

    # Remove known tool prefixes like 'file_manager'
    tool_prefixes = ["file_manager", "web_search", "system_control"]
    for prefix in tool_prefixes:
        command_text = command_text.replace(f"!{prefix}", "").strip()

    # Ensure commands are executed in sequence
    commands = command_text.split("&&")  # Split at '&&' to maintain order
    cleaned_commands = []

    for cmd in commands:
        cmd = cmd.strip()
        
        # Preserve 'echo' with redirection (">") as-is
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
    """Executes valid commands from AI response."""
    commands = clean_ai_response(ai_response)

    if not commands:
        return "No valid commands found."

    # Run commands sequentially in a single shell session
    final_command = " && ".join(commands)

    try:
        result = subprocess.run(final_command, capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            return result.stdout.strip() or f" ^z   ^o Command '{final_command}' executed successfully."
        else:
            return f" ^z   ^o Error:\n{result.stderr.strip()}"
    except Exception as e:
        return f"Execution error: {str(e)}"

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
        "You are an AI OS assistant with built-in tool awareness and command execution capabilities.\n"
        "Your role is to generate shell commands based on user requests, ensuring proper execution.\n"
        "You have access to the following tools:\n\n"
        f"{tool_info}\n\n"
        "### IMPORTANT RULES ###\n"
        "1. If the user's request matches a tool command, **respond ONLY with the exact shell command**, prefixed by `!`.\n"
        "   - Example: If the user says 'list files', respond with: `!ls`\n"
        "   - Example: If the user says 'create a directory called test', respond with: `!mkdir test`\n\n"
        "2. If the request involves multiple steps, combine them using `&&` and respond with **one single-line shell command**.\n"
        "   - Example: 'Create a folder called mydir and add a file named hello.txt inside it'  ^f^r\n"
        "     Response: `!mkdir mydir && touch mydir/hello.txt`\n\n"
        "3. **Do NOT** return structured tool references like `[file_manager: ls]` or explanations ^`^tjust the command.\n"
        "   - WRONG: `[file_manager: ls]`\n"
        "   - CORRECT: `!ls`\n\n"
        "4. If the user's request **cannot** be fulfilled with a shell command, respond in plain text instead.\n"
        "   - Example: 'What ^`^ys the meaning of life?'  ^f^r Response: `The meaning of life is subjective...`\n\n"
        "Always prioritize returning the correct command for execution."
    )   
    
# Basically, we're formatting the response if it's a command, and if it's not, we're just returning the response as is..


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
