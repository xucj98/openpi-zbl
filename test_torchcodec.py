import subprocess
import sys

def check_ffmpeg():
    try:
        # Try to get FFmpeg version
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        print("FFmpeg is installed:")
        print(result.stdout.split('\n')[0])  # Print first line which contains version
    except FileNotFoundError:
        print("FFmpeg is not installed or not in PATH")

def check_ffmpeg_libs():
    try:
        # Check for common FFmpeg library files
        libs = ['libavutil.so', 'libavcodec.so', 'libavformat.so']
        for lib in libs:
            result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True)
            if lib in result.stdout:
                print(f"Found {lib}")
            else:
                print(f"Missing {lib}")
    except Exception as e:
        print(f"Error checking FFmpeg libraries: {e}")

print("=== FFmpeg Check ===")
check_ffmpeg()
print("\n=== FFmpeg Libraries Check ===")
check_ffmpeg_libs()

print("\n=== TorchCodec Check ===")
try:
    from torchcodec.decoders import VideoDecoder
    print("Successfully imported VideoDecoder from torchcodec")
except ImportError as e:
    print(f"Failed to import VideoDecoder: {e}")

try:
    import torchcodec
    print(f"torchcodec version: {torchcodec.__version__}")
except ImportError as e:
    print(f"Failed to import torchcodec: {e}") 