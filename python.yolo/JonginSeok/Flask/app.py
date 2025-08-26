from flask import Flask
# LOG
# (yolov_env) C:\Users\ngins\Git\python.yolo>C:/Users/ngins/.conda/envs/yolov_env/python.exe c:/Users/ngins/Git/python.yolo/JonginSeok/Flask/simple.py
#  * Serving Flask app 'simple'
#  * Debug mode: on
# WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
#  * Running on http://127.0.0.1:5000
# Press CTRL+C to quit
#  * Restarting with stat
#  * Debugger is active!
#  * Debugger PIN: 876-360-080

#  * Detected change in 'c:\\Users\\ngins\\Git\\python.yolo\\JonginSeok\\Flask\\simple.py', reloading
#  * Restarting with stat
#   File "c:\Users\ngins\Git\python.yolo\JonginSeok\Flask\simple.py", line 12
#     (yolov_env) C:\Users\ngins\Git\python.yolo>C:/Users/ngins/.conda/envs/yolov_env/python.exe c:/Users/ngins/Git/python.yolo/JonginSeok/Flask/simple.py
#                 ^
# SyntaxError: invalid syntax

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, Flask!'

if __name__ == '__main__':
    app.run(debug=True)
