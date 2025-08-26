# 터미널에서 실행
# pip install Flask

# (yolov_env) C:\Users\ngins\Git\python.yolo>pip install Flask
# Collecting Flask
#   Downloading flask-3.1.1-py3-none-any.whl.metadata (3.0 kB)
# Collecting blinker>=1.9.0 (from Flask)
#   Downloading blinker-1.9.0-py3-none-any.whl.metadata (1.6 kB)
# Collecting click>=8.1.3 (from Flask)
#   Downloading click-8.2.1-py3-none-any.whl.metadata (2.5 kB)
# Collecting itsdangerous>=2.2.0 (from Flask)
#   Downloading itsdangerous-2.2.0-py3-none-any.whl.metadata (1.9 kB)
# Requirement already satisfied: jinja2>=3.1.2 in c:\users\ngins\.conda\envs\yolov_env\lib\site-packages (from Flask) (3.1.6)
# Requirement already satisfied: markupsafe>=2.1.1 in c:\users\ngins\.conda\envs\yolov_env\lib\site-packages (from Flask) (3.0.2)
# Collecting werkzeug>=3.1.0 (from Flask)
#   Downloading werkzeug-3.1.3-py3-none-any.whl.metadata (3.7 kB)
# Requirement already satisfied: colorama in c:\users\ngins\.conda\envs\yolov_env\lib\site-packages (from click>=8.1.3->Flask) (0.4.6)
# Downloading flask-3.1.1-py3-none-any.whl (103 kB)
# Downloading blinker-1.9.0-py3-none-any.whl (8.5 kB)
# Downloading click-8.2.1-py3-none-any.whl (102 kB)
# Downloading itsdangerous-2.2.0-py3-none-any.whl (16 kB)
# Downloading werkzeug-3.1.3-py3-none-any.whl (224 kB)
# Installing collected packages: werkzeug, itsdangerous, click, blinker, Flask
# Successfully installed Flask-3.1.1 blinker-1.9.0 click-8.2.1 itsdangerous-2.2.0 werkzeug-3.1.3                                                                                                    

import flask
print(flask.__version__)

# (yolov_env) C:\Users\ngins\Git\python.yolo>C:/Users/ngins/.conda/envs/yolov_env/python.exe c:/Users/ngins/Git/python.yolo/JonginSeok/Flask/install.py
# c:\Users\ngins\Git\python.yolo\JonginSeok\Flask\install.py:27: DeprecationWarning: The '__version__' attribute is deprecated and will be removed in Flask 3.2. Use feature detection or 'importlib.metadata.version("flask")' instead.
#   print(flask.__version__)
# 3.1.1