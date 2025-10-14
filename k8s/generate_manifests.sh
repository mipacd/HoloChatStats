#!/bin/bash
set -e

cd "$(dirname "$0")"

if [ ! -f "../.env" ]; then
  echo ".env file not found in project root!"
  exit 1
fi

echo "🧩 Generating Kubernetes manifests from .env..."

# Define secret keys
SECRET_KEYS=("POSTGRES_USER" "POSTGRES_PASSWORD" "SECRET_KEY" "OPENROUTER_API_KEY")

# Create clean temp file (strip CR, comments, and blank lines)
CLEAN_ENV=$(mktemp)
# Convert CRLF -> LF, remove comments/blank lines
tr -d '\r' < ../.env | grep -v '^[[:space:]]*#' | grep -v '^[[:space:]]*$' > "$CLEAN_ENV"

# Generate ConfigMap
echo "Creating configmap.yaml..."
cat > configmap.yaml <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: holochatstats-config
data:
EOF

while IFS='=' read -r key value; do
  [[ " ${SECRET_KEYS[*]} " == *" $key "* ]] && continue
  safe_value=$(echo "$value" | sed 's/"/\\"/g')
  echo "  $key: \"$safe_value\"" >> configmap.yaml
done < "$CLEAN_ENV"

# Generate Secret
echo "Creating secret.yaml..."
cat > secret.yaml <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: holochatstats-secrets
type: Opaque
stringData:
EOF

while IFS='=' read -r key value; do
  [[ " ${SECRET_KEYS[*]} " != *" $key "* ]] && continue
  safe_value=$(echo "$value" | sed 's/"/\\"/g')
  echo "  $key: \"$safe_value\"" >> secret.yaml
done < "$CLEAN_ENV"

rm "$CLEAN_ENV"

echo "ConfigMap and Secret generated successfully!"
