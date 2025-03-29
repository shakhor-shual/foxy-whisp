#!/bin/bash

# Force immediate flush of stdout
export PYTHONUNBUFFERED=1
# Set debug level
export PYTHONDEBUGLEVEL=1

# Check if script name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <script_name.py> [additional args...]"
    exit 1
fi

# Get script name from first argument and remaining args
script_name="$1"
shift  # Remove first argument, leaving remaining args in $@

# Ensure the script resolves the correct path
SCRIPT_PATH=$(realpath "$script_name")

# Create logs directory if it doesn't exist
mkdir -p logs

# Fixed log file path
log_file="logs/debug_log.txt"

# Clear previous log file and start new logging
echo "=== New Session $(date) ===" > "${log_file}"

# Find the script
server_script=$(find "$(pwd)" -type f -name "$(basename "$SCRIPT_PATH")")

if [ -z "$server_script" ]; then
    echo "Error: Cannot find $script_name" | tee -a "${log_file}"
    exit 1
fi

echo "Found script at: $server_script" | tee -a "${log_file}"

# Run the command using absolute path with remaining arguments, using stdbuf to disable buffering
stdbuf -oL -eL python3 "$SCRIPT_PATH" "$@" 2>&1 | stdbuf -oL tee -a "${log_file}"
