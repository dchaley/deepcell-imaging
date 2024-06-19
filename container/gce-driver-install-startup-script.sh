#!/bin/bash

#########################################
# NOTE: WE DID NOT FINISH TESTING THIS! #
#########################################


# On GCE, we need to install the GPU driver.
# This is automated on Vertex AI and Batch.
# This is run as a startup script.
# It needs to be deployed to accessible cloud storage somewhere.

if test -f /opt/google/cuda-installer
then
  exit
fi

mkdir -p /opt/google/cuda-installer/
cd /opt/google/cuda-installer/ || exit

curl -fSsL -O https://github.com/GoogleCloudPlatform/compute-gpu-installation/releases/download/cuda-installer-v1.1.0/cuda_installer.pyz
python3 cuda_installer.pyz install_cuda
