#!/usr/bin/env bash
set -euo pipefail

MUJOCO_DIR="${HOME}/.mujoco"
MUJOCO_ROOT="${MUJOCO_DIR}/mujoco210"
MUJOCO_ARCHIVE="${MUJOCO_DIR}/mujoco210-linux-x86_64.tar.gz"
MUJOCO_URL="https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz"

sudo apt update
sudo apt install -y build-essential python3-dev libosmesa6-dev libgl1 libglew-dev libglfw3 patchelf wget

mkdir -p "${MUJOCO_DIR}"
if [ ! -d "${MUJOCO_ROOT}" ]; then
    wget "${MUJOCO_URL}" -O "${MUJOCO_ARCHIVE}"
    tar -xzf "${MUJOCO_ARCHIVE}" -C "${MUJOCO_DIR}"
    rm -f "${MUJOCO_ARCHIVE}"
fi

echo "MuJoCo 2.1 is installed at ${MUJOCO_ROOT}"
