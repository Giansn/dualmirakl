#!/bin/bash
# =============================================================================
# Build and push the dualmirakl container image.
# Run from /per.volume/dualmirakl/:
#   bash build.sh [dockerhub-username]
# =============================================================================

REGISTRY="${1:-giansn}"
IMAGE="$REGISTRY/dualmirakl"
TAG="runpod-cu128"

echo "[build] Building $IMAGE:$TAG ..."
docker build \
    --platform linux/amd64 \
    -t "$IMAGE:$TAG" \
    -t "$IMAGE:latest" \
    -f Dockerfile \
    .

if [ $? -ne 0 ]; then
    echo "[build] Build FAILED."
    exit 1
fi

echo "[build] Pushing $IMAGE:$TAG ..."
docker push "$IMAGE:$TAG"
docker push "$IMAGE:latest"

echo ""
echo "Done. Use this image in RunPod:"
echo "  $IMAGE:$TAG"
echo ""
echo "RunPod pod settings:"
echo "  Container Image : $IMAGE:$TAG"
echo "  Volume Mount    : /per.volume"
echo "  HTTP Ports      : 8888, 8080, 8008"
