from PIL.Image import Image
from flask import Flask, request, render_template,jsonify
import main2 as m2
import cv2
import os
from PIL import Image
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
# from flask_cors import CORS, cross_origin
UPLOAD_FOLDER = r'G:\DS\PERSONAL PROJECTS\Study Projects\DL specific\Computer Vision\MASK DETECTOR\static\uploads'
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/',methods=['GET','POST']) ## render home page
def homepage():
    return render_template('home.html')

@app.route('/predict',methods =['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # image.save(image.filename)
        # i = open(image.filename)
        im = cv2.imread(r'G:/DS/PERSONAL PROJECTS/Study Projects/DL specific/Computer Vision/MASK DETECTOR/static/uploads/'+ filename)
        #im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
        #rgbaimage = Image.open(request.files['file'].stream)

        #im = rgbaimage.convert('RGB')
        #im.save("hp.png")
        #return "successfully uploaded"
        #im.shape
        c = m2.Classifier()
        i = c.get_classification(im)
        # #
        image = Image.fromarray(i,'RGB')
        image.save(r'G:/DS/PERSONAL PROJECTS/Study Projects/DL specific/Computer Vision/MASK DETECTOR/static/'+filename+'.png')
        img = 'static/'+filename+'.png'
        return render_template("predict.html",user_image=img)



if __name__ == "__main__":
    app.run(debug=True)
