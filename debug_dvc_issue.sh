#!/bin/bash
# Debug script for DVC pull issue
LOG_FILE="/home/user/ipml_boston_housing/.cursor/debug.log"

# Clear previous logs
true > "$LOG_FILE"

log_entry() {
    local hypothesis_id="$1"
    local location="$2"
    local message="$3"
    local data="$4"

    echo "{\"sessionId\":\"debug-session\",\"runId\":\"dvc-debug\",\"hypothesisId\":\"$hypothesis_id\",\"location\":\"$location\",\"message\":\"$message\",\"data\":$data,\"timestamp\":$(date +%s)000}" >> "$LOG_FILE"
}

echo "=== DVC Pull Issue Diagnostics ==="

# Hypothesis A: Check file hashes vs .dvc files
echo "Checking file hashes..."
MODEL_DIR="/home/user/ipml_boston_housing/data/models"
if [ -d "$MODEL_DIR" ]; then
    FILE_COUNT=$(ls -1 "$MODEL_DIR"/*.pkl 2>/dev/null | wc -l)
    FIRST_FILE=$(ls "$MODEL_DIR"/*.pkl 2>/dev/null | head -1)
    if [ -n "$FIRST_FILE" ]; then
        FIRST_HASH=$(md5sum "$FIRST_FILE" | awk '{print $1}')
        log_entry "A" "debug_dvc_issue.sh:25" "Local model files found" "{\"file_count\":$FILE_COUNT,\"first_file\":\"$FIRST_FILE\",\"first_hash\":\"$FIRST_HASH\"}"
    else
        log_entry "A" "debug_dvc_issue.sh:28" "No model files found" "{\"model_dir\":\"$MODEL_DIR\"}"
    fi
else
    log_entry "A" "debug_dvc_issue.sh:31" "Model directory does not exist" "{\"model_dir\":\"$MODEL_DIR\"}"
fi

# Hypothesis B: Check for .dvc tracking files
echo "Checking DVC tracking files..."
DVC_FILES=$(find /home/user/ipml_boston_housing -name "*.dvc" -type f | wc -l)
DVC_LIST=$(find /home/user/ipml_boston_housing -name "*.dvc" -type f | head -5 | tr '\n' ',' | sed 's/,$//')
log_entry "B" "debug_dvc_issue.sh:39" "DVC tracking files count" "{\"dvc_files_count\":$DVC_FILES,\"sample_files\":\"$DVC_LIST\"}"

# Hypothesis C: Check DVC remote configuration
echo "Checking DVC remote configuration..."
if [ -f "/home/user/ipml_boston_housing/.dvc/config" ]; then
    REMOTE_NAME=$(grep -A 1 "\[core\]" /home/user/ipml_boston_housing/.dvc/config | grep "remote" | awk -F'=' '{print $2}' | tr -d ' ')
    log_entry "C" "debug_dvc_issue.sh:47" "DVC remote configuration" "{\"remote_name\":\"$REMOTE_NAME\",\"config_exists\":true}"
else
    log_entry "C" "debug_dvc_issue.sh:49" "DVC config not found" "{\"config_path\":\"/home/user/ipml_boston_housing/.dvc/config\"}"
fi

# Hypothesis D: Check DVC status
echo "Checking DVC status..."
cd /home/user/ipml_boston_housing || exit
DVC_STATUS_OUTPUT=$(dvc status 2>&1 | head -20)
DVC_STATUS_EXIT=$?
log_entry "D" "debug_dvc_issue.sh:57" "DVC status command result" "{\"exit_code\":$DVC_STATUS_EXIT,\"output_preview\":\"$(echo "$DVC_STATUS_OUTPUT" | head -3 | tr '\n' ' ')\"}"

# Hypothesis E: Check file permissions
echo "Checking file permissions..."
if [ -n "$FIRST_FILE" ]; then
    PERMS=$(stat -c "%a %U:%G" "$FIRST_FILE" 2>/dev/null)
    log_entry "E" "debug_dvc_issue.sh:64" "File permissions check" "{\"file\":\"$FIRST_FILE\",\"permissions\":\"$PERMS\"}"
fi

# Additional: Check DVC cache
echo "Checking DVC cache..."
CACHE_DIR="/home/user/ipml_boston_housing/.dvc/cache"
if [ -d "$CACHE_DIR" ]; then
    CACHE_SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | awk '{print $1}')
    CACHE_FILES=$(find "$CACHE_DIR" -type f 2>/dev/null | wc -l)
    log_entry "C" "debug_dvc_issue.sh:74" "DVC cache status" "{\"cache_size\":\"$CACHE_SIZE\",\"cache_files\":$CACHE_FILES}"
else
    log_entry "C" "debug_dvc_issue.sh:76" "DVC cache directory not found" "{\"cache_dir\":\"$CACHE_DIR\"}"
fi

# Check what .dvc files expect
echo "Checking .dvc file contents..."
DVC_MODELS_FILE="/home/user/ipml_boston_housing/data/models.dvc"
if [ -f "$DVC_MODELS_FILE" ]; then
    DVC_HASH=$(grep "md5:" "$DVC_MODELS_FILE" | awk '{print $2}' | head -1)
    log_entry "A" "debug_dvc_issue.sh:85" "DVC tracking file hash" "{\"dvc_file\":\"$DVC_MODELS_FILE\",\"expected_hash\":\"$DVC_HASH\"}"
fi

echo "=== Diagnostics complete. Check log file: $LOG_FILE ==="
