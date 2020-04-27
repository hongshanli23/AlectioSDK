from flask import jsonify
from flask import Flask, Response
from flask import request
from flask import send_file

import traceback
import sys



class Pipeline(object):
    def __init__(self, name, train_fn, test_fn, infer_fn):
        self.app = Flask(name)

        self.train_fn = train_fn
        self.test_fn = test_fn
        self.infer_fn = infer_fn

        # train 
        self.app.add_url_rule(
            '/train', 'train', self._train, methods=['POST']
            )
        
        # test
        self.app.add_url_rule(
            '/test', 'test', self._test
            )

        # infer
        self.app.add_url_rule(
            '/infer', 'infer', self._infer
            )


    def _train(self):
        # labeled = map(int, request.get_json()['labeled'])
        payload = request.get_json()
        try:
            self.train_fn(payload)
            return jsonify({'status': 200})

        except Exception as e:
            traceback.print_exceptions(sys.exc_info())
            return jsonify({'status': 500})
    
    def _test(self):
        payload = request.get_json()
        try:
            prd, lbs = self.test_fn(payload)
            return jsonify(
                    {'status': 200,
                     'prd': prd,
                     'lbs': lbs
                })

        except Exception as e:
            traceback.print_exceptions(sys.exc_info())
            return jsonify({'status': 500})
            
    
    def _infer(self):
        payload = request.get_json()
        try:
            output = self.infer_fn(payload)
            return jsonify({
                'status': 200,
                'output': output
                })
        except Exception as e:
            traceback.print_exceptions(sys.exc_info())
            return jsonify({
                'status': 500
                })

    def __call__(self, debug=False, host='0.0.0.0'):
        self.app.run(debug=debug, host=host)






