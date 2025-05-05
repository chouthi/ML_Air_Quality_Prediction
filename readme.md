## OVERVIEW

- This application predicts air quality using machine learning models based on environmental data such as temperature, humidity, and wind speed. It avoids deep learning and instead relies on classical algorithms like Linear Regression or Random Forest.

## HOW TO RUN
### Step 1:
- Create virtual env for python version 3.11:
For **window**: ```& "C:\Program Files\Python311\python.exe" -m venv .venv ```
For **macOS**: ```python3.11 -m venv .venv```
### Step 2:
- Active python version in venv(virtual environment):
If you are using **window** before continue run this command: ```Set-ExecutionPolicy -Scope Process -
ExecutionPolicy Bypass```
For **window**: ```.venv\Scripts\activate```
For **macOS**: ```source .venv/bin/activate```
## Step 3:
- Install the requirements:
```pip install -r requirements.txt``` or ```pip3 install -r requirements.txt```
## Step 4:
- run: ```python .\src\preprocess.py```