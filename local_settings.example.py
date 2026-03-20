# local_settings.example.py
#
# Machine-specific settings for crystal-3d-recon.
# These override the defaults in crystal_recon/config.py.
#
# HOW TO USE:
#   1. Copy this file to local_settings.py in the same directory
#   2. Fill in the values for your machine
#   3. local_settings.py is git-ignored — it will never be committed
#
# Copy command:
#   Windows:  copy local_settings.example.py local_settings.py
#   Mac/Linux: cp local_settings.example.py local_settings.py

# Path to the Allied Vision GenICam CTI file.
# Run `python capture.py --list-cameras` without this set to find the path,
# then paste it here so you don't need --cti every time.
#
# Windows examples:
#   CAMERA_CTI_PATH = r"C:\Program Files\Allied Vision\Vimba X\cti\VimbaUSBTL.cti"
#   CAMERA_CTI_PATH = r"C:\Program Files\Allied Vision\Vimba X\cti\VimbaGigETL.cti"
#
# macOS example:
#   CAMERA_CTI_PATH = "/Library/Frameworks/VimbaX.framework/Resources/cti/VimbaUSBTL.cti"
#
# Linux example:
#   CAMERA_CTI_PATH = "/opt/VimbaX/cti/VimbaUSBTL.cti"
#
CAMERA_CTI_PATH = None  # Replace None with your path

# Serial port for the Zaber rotation stage.
# Leave as None to auto-discover (scans all available ports).
# Set explicitly if auto-discovery is slow or picks the wrong port.
#
# Windows example:  ZABER_PORT = "COM3"
# macOS example:    ZABER_PORT = "/dev/tty.usbserial-A9XXXXXX"
# Linux example:    ZABER_PORT = "/dev/ttyUSB0"
#
ZABER_PORT = None  # Replace None with your port if needed
