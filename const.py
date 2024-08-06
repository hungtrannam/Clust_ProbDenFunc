

def venv(env_name=".venv"):
    """
    Create a virtual environment.
    """
    import subprocess
    import sys, pathlib, os
    try:
        subprocess.check_call([sys.executable, "-m", "venv", env_name])        
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while creating virtual environment: {e}")
        print(f"Command: {e.cmd}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output}")

def installPackages(env_name=".venv"):
    """
    Install necessary packages listed in requirements.txt within the virtual environment.
    """
    import subprocess
    import pathlib
    python_executable = pathlib.Path(env_name) / "bin" / "python"
    try:
        subprocess.check_call([python_executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing packages: {e}")
        print(f"Command: {e.cmd}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output}")

        
import pathlib

if not pathlib.Path(".venv").exists():
    print("Virtual environment not found. Creating and installing packages...")
    venv()
    installPackages()
    print("\n\n\n\nSetup complete. You can now use the tool.")
else:
    installPackages()
    print("Virtual environment already exists. You can now use the tool.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import tqdm
from typing import Dict
from pydantic import BaseModel, Field, ConfigDict, validate_call
from enum import Enum
import time

# if __name__ == "__main__":






