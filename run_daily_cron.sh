#!/bin/bash
# run_daily_cron.sh
# This script is meant to be executed automatically by cron.

# 1. Navigate to your repository directory on the Linux machine
cd /mnt/disk/home/alexandregodoy/av2av || exit 1

# 2. Make Conda available to this script
# Adjust the path below if your miniconda is installed elsewhere
source ~/miniconda3/etc/profile.d/conda.sh

# 3. Activate the environment
conda activate av2av_env

# 4. Run the daily training pipeline
# We pipe both stdout and stderr (>> and 2>&1) into a log file so you can check it later
echo "--- Starting Chron Job: $(date) ---" >> cron_execution.log
python scripts/run_daily_cycle.py \
    --date "today" \
    --drive-folder "videos_mestrado" \
    --inference-repo "../model-stst-1" \
    >> cron_execution.log 2>&1
echo "--- Finished Chron Job: $(date) ---" >> cron_execution.log
