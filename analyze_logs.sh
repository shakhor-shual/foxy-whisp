#!/bin/bash

LOG_FILE="logs/debug_log.txt"

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file not found at $LOG_FILE"
    exit 1
fi

echo "=== Log File Analysis ==="
echo "Reading from: $LOG_FILE"
echo "-------------------"

# Display the log content
echo "Log contents:"
cat "$LOG_FILE"
echo "-------------------"

# Check for common error patterns
echo "Error analysis:"
grep -i "error" "$LOG_FILE" || echo "No errors found"
grep -i "exception" "$LOG_FILE" || echo "No exceptions found"
grep -i "failed" "$LOG_FILE" || echo "No failures found"

echo "-------------------"
echo "End of analysis"
