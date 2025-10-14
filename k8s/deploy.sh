#!/bin/bash
set -e

cd "$(dirname "$0")"

./generate_manifests.sh

echo "Applying manifests to Kubernetes..."
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
kubectl apply -k template/

echo "HoloChatStats deployed to Kubernetes"
kubectl get pods
kubectl get svc
