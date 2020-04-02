from flask import jsonify
from flask import Flask
from flask import request
from flask import send_file

import processes

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    payload = request.get_json()
    processes.train(payload)
    return jsonify({'status': 200})

@app.route('/test', methods=['POST'])
def test():
    payload = request.get_json()
    prd, lbs = processes.test(payload)
    return jsonify({"prediction": prd, "test_label": lbs})

@app.route('/infer', methods=['POST'])
def infer():
    payload = request.get_json() 
    output = processes.infer(payload)
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')




