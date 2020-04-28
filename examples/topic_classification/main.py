from alectio_sdk.flask_wrapper import Pipeline
from processes import train,test,infer

app = Pipeline('yolov3', train, test, infer)


if __name__ == '__main__':
    app(debug=True)
