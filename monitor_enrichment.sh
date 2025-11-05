#!/bin/bash
# Detailed enrichment monitoring with file completion tracking

PID=5940
CHECK_INTERVAL=60  # Check every 60 seconds

# Initialize tracking
LAST_FILE=""
FILES_COMPLETED=0
CHECK_COUNT=0

echo "========================================"
echo "🔍 Starting Detailed Enrichment Monitor"
echo "========================================"
echo "PID: $PID"
echo "Check interval: ${CHECK_INTERVAL}s"
echo ""

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S UTC')

    # Check if process is running
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "[$TIMESTAMP] ⏹️  Process $PID has stopped"
        echo ""
        echo "Final Status:"
        ls -lh data/*.csv | awk '{printf "  %-25s %8s\n", $9, $5}'
        echo ""
        echo "Enrichment process completed or terminated."
        break
    fi

    # Get current file being processed
    CURRENT_FILE=$(tail -20 enrichment_output.log | grep "Processing:" | tail -1 | awk '{print $2}')

    # Check for file completion
    COMPLETED=$(tail -5 enrichment_output.log | grep "All locations in" | tail -1)

    if [ ! -z "$COMPLETED" ] && [ "$CURRENT_FILE" != "$LAST_FILE" ]; then
        COMPLETED_FILE=$(echo "$COMPLETED" | awk '{print $5}')
        FILES_COMPLETED=$((FILES_COMPLETED + 1))
        echo "[$TIMESTAMP] ✅ COMPLETED: $COMPLETED_FILE"
        echo "              Files done: $FILES_COMPLETED"
    fi

    # Show current status
    if [ ! -z "$CURRENT_FILE" ] && [ "$CURRENT_FILE" != "$LAST_FILE" ]; then
        echo "[$TIMESTAMP] 🔄 NOW PROCESSING: $CURRENT_FILE"
        LAST_FILE="$CURRENT_FILE"
    fi

    # Show recent progress
    RECENT_PROGRESS=$(tail -3 enrichment_output.log | grep "Progress:" | tail -1)
    if [ ! -z "$RECENT_PROGRESS" ]; then
        echo "[$TIMESTAMP]    $RECENT_PROGRESS"
    fi

    # Show file size changes every 5 checks
    CHECK_COUNT=$((CHECK_COUNT + 1))
    if [ $((CHECK_COUNT % 5)) -eq 0 ]; then
        echo "[$TIMESTAMP] 📊 File sizes:"
        ls -lh data/*.csv | grep -E "(kyoto|liestal|meteoswiss|south_korea|washingtondc|japan)" | awk '{printf "              %-20s %8s\n", $9, $5}'
    fi

    sleep $CHECK_INTERVAL
done
