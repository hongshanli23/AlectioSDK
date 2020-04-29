from alectio_sdk.flask_wrapper import Pipeline
from processes import train,test,infer

# put the train/test/infer processes into the constructor
app = Pipeline(name='daily_dialog', train_fn=train, 
               test_fn=test, infer_fn=infer)


if __name__ == '__main__':
    app(debug=True)
