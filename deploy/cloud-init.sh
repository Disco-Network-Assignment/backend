#!/bin/bash
# EC2 user data for a fresh Ubuntu 24.04 instance: install Docker with the compose plugin and
# prepare the directory deploy.sh copies the code into. Runs once, as root, at first boot.
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y docker.io docker-compose-v2
systemctl enable --now docker
usermod -aG docker ubuntu

mkdir -p /opt/disco/backend
chown -R ubuntu:ubuntu /opt/disco

# small box: give the kernel some swap so a docker build never gets OOM-killed
if [ ! -f /swapfile ]; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
