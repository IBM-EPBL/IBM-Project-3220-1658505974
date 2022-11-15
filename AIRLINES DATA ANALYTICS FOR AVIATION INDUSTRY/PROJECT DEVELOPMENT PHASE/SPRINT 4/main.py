from flask import render_template,Flask,request
import pickle

appl=Flask(__name__)
file=open("model.pkl","rb")

knn=pickle.load(file)
file.close()

@appl.route("/", methods=["GET","POST"])
def index():
    if request.method=="POST":
        myDict = request.form
        type1= myDict["elevation_ft"]
        pred = [type1]
        res=knn.predict([pred])[0]
        return render_template('result.html',elevation_ft=type1,res=res)
    return render_template('index.html')
    return 'OK'
if __name__ == "__main__":
    appl.run(debug=True)
  