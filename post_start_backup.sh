#!/bin/bash
# RunPod post-start hook — runs after SSH is up on every pod boot.
# Launches the dualmirakl stack in background so it doesn't block pod startup.
#
# NOTE: This file lives on the ephemeral container filesystem.
# If the pod is RECREATED (not just restarted), restore it with:
#   bash /per.volume/dualmirakl/relink.sh

nohup bash /per.volume/dualmirakl/autostart.sh &
echo "[post_start] dualmirakl autostart launched (PID $!)"
echo "[post_start] Tail logs: tail -f /per.volume/dualmirakl/logs/autostart.log"
