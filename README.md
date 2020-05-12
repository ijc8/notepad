# notepad
multimodal music composition / composing music naturally

## Setup

    virtualenv -p python3 venv
    source venv/bin/activate
    pip install -r requirements.txt

## Other Requirements

- Install fluidsynth 2+
- Install SDL libraries: libsdl1.2-dev, libsdl-mixer1.2, libsdl2-2.0
- Python3.7
- Runs on Ubuntu 18.04, OSX (Mojave) and Android

## Table of Contents

| Filename      | Description   |
| ------------- |:-------------:|
| /export/       | Folder containing PDF and wave files |
| /icons/        | Folder containing various icons      |
| /model/        | Folder containing gesture template database    |
| /saved/        | Folder containing saved notepads so that one can open later      |
| /test/         | Folder containing files necessary for testing audio transcription performance      |
| buildozer.spec      | Build file necessary for deploying on Android      |
| dollarpy.py       | DollarPy implementation. Forked a little bit to improve the speed.      |
| fluidsynth.py       | FluidSynth python implementation. Forked slightly to meet our needs      |
| gesturedatabase.py/kv       | UI and business logic for managing templates |
| historymanager.py/kv       | UI and business logic for managing the most recent gestures inputted |
| main.py     | Everything starts here. Call by python main.py |
| notepad.kv     | UI file |
| recorder.py   | Helper class for recording audio |
| transcribe.py   | Logic for transcribing |
| util.py and helpers.py   | Helper functions |



