#!/bin/bash
# Helper script to upload files to RunPod using SFTP
# Usage: ./upload_files.sh <pod-username>@<pod-host>

if [ -z "$1" ]; then
    echo "Usage: $0 <pod-username>@<pod-host>"
    echo "Example: $0 1vtu46wv1fktma-64411e51@ssh.runpod.io"
    exit 1
fi

POD_HOST="$1"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

echo "Uploading files to RunPod..."
echo "Host: $POD_HOST"
echo ""

# Create SFTP batch script
SFTP_SCRIPT=$(mktemp)
cat > "$SFTP_SCRIPT" << SFTPEOF
mkdir -p /workspace/runpod
mkdir -p /workspace/data
put -r runpod/*.py /workspace/runpod/
put -r runpod/*.txt /workspace/runpod/
put -r runpod/*.sh /workspace/runpod/
put -r data/*.jsonl /workspace/data/
quit
SFTPEOF

# Upload using SFTP
if [ -f "$SSH_KEY" ]; then
    sftp -i "$SSH_KEY" -b "$SFTP_SCRIPT" "$POD_HOST"
else
    sftp -b "$SFTP_SCRIPT" "$POD_HOST"
fi

rm "$SFTP_SCRIPT"
echo ""
echo "Upload complete!"
