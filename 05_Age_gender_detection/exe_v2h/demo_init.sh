#!/bin/bash

# Usage function to display help if arguments are missing
usage() {
    echo "Usage: $0 <frame_rate (1-30)> <flip (true/false)>"
    exit 1
}

# Check if there are exactly two arguments
if [ $# -ne 2 ]; then
    usage
fi

# Read frame rate and flip arguments
FR=$1
FLIP=$2

# Validate frame rate (must be an integer between 1 and 30)
if ! [[ "$FR" =~ ^[0-9]+$ ]] || [ "$FR" -lt 1 ] || [ "$FR" -gt 30 ]; then
    echo "Error: Frame rate must be a number between 1 and 30."
    exit 1
fi

# Run v4l2-init.sh
/home/root/v4l2-init.sh --device 1 --width 1920 --height 1080

# Configure flip based on the second argument
if [ "$FLIP" = "true" ]; then
    echo "Enabling flip..."
    i2cset -f -y 1 0x3d 0x10 0x0c 0x00 0x2 i
elif [ "$FLIP" = "false" ]; then
    echo "Disabling flip..."
    i2cset -f -y 1 0x3d 0x10 0x0c 0x00 0x0 i
else
    echo "Error: Flip value must be 'true' or 'false'."
    exit 1
fi

# Configure frame rate
echo "Setting frame rate to $FR fps..."
i2cset -f -y 1 0x3d 0x20 0x20 $FR 0x00 i
echo "Enabling ISP face based AE "
echo "0x5002" >  /sys/kernel/debug/ap1302.1-003d/isp_addr
echo "0x29C" > /sys/kernel/debug/ap1302.1-003d/isp_data
echo "Increasing AE brightness target"
echo "0x5018" >  /sys/kernel/debug/ap1302.1-003d/isp_addr
echo "0xFE00" > /sys/kernel/debug/ap1302.1-003d/isp_data
echo "Configuration complete."
echo "Starting Demo"
./age_gender_detection_app

