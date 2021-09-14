from flask import Flask, render_template, request
from bert_model import BertSeqCls

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=["POST"])
def predict():
    post = request.values['post']
    comment = request.values['comment']
    print(post)
    print(comment)
    model = BertSeqCls()
    pre = model.pre_api(post, comment)
    return str(pre)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8800, debug=True)
