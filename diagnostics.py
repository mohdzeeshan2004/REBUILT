"""Print environment checks to help diagnose installation errors."""
import importlib.util
import platform
import sys

print("Python:", sys.version.replace("\n", " "))
print("Executable:", sys.executable)
print("Platform:", platform.platform())
for module in ("streamlit", "cv2", "numpy", "PIL", "yaml", "ultralytics", "faster_whisper"):
    print(f"{module}:", "installed" if importlib.util.find_spec(module) else "NOT INSTALLED")
print("\nCore modules are streamlit, cv2, numpy, PIL, yaml. YOLO and faster-whisper are optional.")
