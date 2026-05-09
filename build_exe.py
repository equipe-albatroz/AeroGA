import os
import subprocess
import shutil

print("Building AeroGA GUI Executable...")

# Install requirements if not present
print("Ensuring dependencies are installed...")
subprocess.run(["pip", "install", "eel", "pyinstaller", "pillow"], check=True)

# Convert PNG to ICO for PyInstaller
# Skipping conversion since we are using aero_ico1.ico natively now


# Build command for PyInstaller
# We use eel's special command format to package the web folder
build_command = [
    "python", "-m", "eel", "gui/main.py", "gui/web", 
    "--onefile", 
    "--noconsole",
    "--name", "AeroGA",
    "--collect-all", "numpy",
    "--collect-all", "pandas",
    "--collect-submodules", "plotly",
    "--hidden-import", "plotly.io",
    "--hidden-import", "plotly.graph_objs",
    "--hidden-import", "plotly.express",
    "--collect-all", "matplotlib",
    "--collect-all", "matplotlib_inline",
    "--collect-all", "joblib",
    "--collect-all", "openpyxl",
    "--collect-all", "loguru",  
    "--collect-all", "colorlog"
]

if os.path.exists("aero_ico2.ico"):
    build_command.extend(["--icon", "aero_ico2.ico"])
    print("Found aero_ico2.ico, using it as application icon.")
else:
    print("Warning: aero_ico2.ico not found. Building without custom icon.")

print("Running PyInstaller...")
try:
    subprocess.run(build_command, check=True)
    print("Build successful! The executable is located in the 'dist' folder.")
except subprocess.CalledProcessError as e:
    print(f"Error during build: {e}")
    
# Clean up build artifacts
print("Cleaning up build artifacts...")
if os.path.exists("build"):
    shutil.rmtree("build")
if os.path.exists("AeroGA_GUI.spec"):
    os.remove("AeroGA_GUI.spec")

print("Done.")
