from flask import Flask, render_template,request
app = Flask(__name__)
import pickle
import joblib
model = pickle.load(open("profit.pkl",'rb'))
#model = pickle.load(open("profit.pkl",'rb'))
ct = joblib.load('column')

@app.route('/')
def hello_world():
    return render_template("index.html")
@app.route('/guest', methods =["post"])
def Guest():
    ms = request.form["ms"]
    ad = request.form["ad"]
    rd = request.form["rd"]
    s = request.form["s"]
    data =[[ms,ad,rd,s]]
    prediction = model.predict(ct.transform(data))
    prediction = prediction[0][0]
    return render_template("index.html",y = "profit could be" + ' ' + str(prediction))
if __name__ == '__main__':

    app.run(debug = True)

                           #@app.route('/user')
#def user ():
   # return "hellow user welcome"
