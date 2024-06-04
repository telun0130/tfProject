from flask import Flask, render_template, request
from predict import predictor
from train import trainer

app = Flask(__name__)

# @app.route('/')

# if(__name__ == "__main__"):
#     app.run(debug=True)