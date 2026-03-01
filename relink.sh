#!/bin/bash
# Restores /post_start.sh after a pod RECREATION (template reset).
# Run once after logging into a fresh pod:
#   bash /per.volume/dualmirakl/relink.sh

cp /per.volume/dualmirakl/post_start_backup.sh /post_start.sh
chmod +x /post_start.sh
echo "[relink] /post_start.sh restored. It will auto-run on next restart."
echo "[relink] To trigger now: bash /per.volume/dualmirakl/autostart.sh"
