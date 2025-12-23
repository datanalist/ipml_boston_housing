#!/bin/bash
# Fix script for DVC pull issue
LOG_FILE="/home/user/ipml_boston_housing/.cursor/debug.log"

log_entry() {
    local hypothesis_id="$1"
    local location="$2"
    local message="$3"
    local data="$4"

    echo "{\"sessionId\":\"debug-session\",\"runId\":\"fix-attempt\",\"hypothesisId\":\"$hypothesis_id\",\"location\":\"$location\",\"message\":\"$message\",\"data\":$data,\"timestamp\":$(date +%s)000}" >> "$LOG_FILE"
}

echo "=== Fixing DVC Pull Issue ==="

# #region agent log
# Log: Count files before cleanup
MODEL_COUNT_BEFORE=$(ls -1 /home/user/ipml_boston_housing/data/models/*.pkl 2>/dev/null | wc -l)
log_entry "FIX" "fix_dvc_pull.sh:18" "Files before cleanup" "{\"model_count\":$MODEL_COUNT_BEFORE}"
# #endregion

# Backup local models to a temporary location (just in case)
BACKUP_DIR="/tmp/boston_housing_models_backup_$(date +%s)"
mkdir -p "$BACKUP_DIR"

# #region agent log
# Log: Backup creation
log_entry "FIX" "fix_dvc_pull.sh:27" "Creating backup" "{\"backup_dir\":\"$BACKUP_DIR\"}"
# #endregion

cp -r /home/user/ipml_boston_housing/data/models/*.pkl "$BACKUP_DIR/" 2>/dev/null
BACKUP_COUNT=$(ls -1 "$BACKUP_DIR"/*.pkl 2>/dev/null | wc -l)

# #region agent log
# Log: Backup completed
log_entry "FIX" "fix_dvc_pull.sh:35" "Backup completed" "{\"backed_up_files\":$BACKUP_COUNT,\"backup_location\":\"$BACKUP_DIR\"}"
# #endregion

echo "Backed up $BACKUP_COUNT model files to $BACKUP_DIR"

# Now force DVC pull to overwrite local changes
cd /home/user/ipml_boston_housing || exit

# #region agent log
# Log: Before dvc pull
log_entry "FIX" "fix_dvc_pull.sh:46" "Starting dvc pull --force" "{\"cwd\":\"/home/user/ipml_boston_housing\"}"
# #endregion

uv run dvc pull --force 2>&1 | tee /tmp/dvc_pull_output.txt
DVC_EXIT_CODE=$?

# #region agent log
# Log: DVC pull result
MODEL_COUNT_AFTER=$(ls -1 /home/user/ipml_boston_housing/data/models/*.pkl 2>/dev/null | wc -l)
log_entry "FIX" "fix_dvc_pull.sh:56" "DVC pull completed" "{\"exit_code\":$DVC_EXIT_CODE,\"models_after\":$MODEL_COUNT_AFTER,\"models_before\":$MODEL_COUNT_BEFORE}"
# #endregion

if [ $DVC_EXIT_CODE -eq 0 ]; then
    echo "✅ DVC pull successful!"
    echo "📊 Models before: $MODEL_COUNT_BEFORE, after: $MODEL_COUNT_AFTER"
    echo "💾 Backup available at: $BACKUP_DIR"

    # #region agent log
    # Log: Success
    log_entry "FIX" "fix_dvc_pull.sh:66" "Fix successful" "{\"status\":\"success\",\"backup\":\"$BACKUP_DIR\"}"
    # #endregion
else
    echo "❌ DVC pull failed with exit code $DVC_EXIT_CODE"
    echo "📋 Output saved to /tmp/dvc_pull_output.txt"

    # #region agent log
    # Log: Failure
    ERROR_MSG=$(tail -5 /tmp/dvc_pull_output.txt | tr '\n' ' ')
    log_entry "FIX" "fix_dvc_pull.sh:75" "Fix failed" "{\"status\":\"failed\",\"exit_code\":$DVC_EXIT_CODE,\"error\":\"$ERROR_MSG\"}"
    # #endregion
fi

echo "=== Fix complete. Check log: $LOG_FILE ==="
