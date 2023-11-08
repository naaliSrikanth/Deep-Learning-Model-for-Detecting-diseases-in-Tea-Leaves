import numpy as np
import os
from flask import Flask, app, request, render_template
import tensorflow
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat
from tensorflow.keras.applications.inception_v3 import preprocess_input
import requests
from flask import Flask, request, render_template, redirect, url_for


#Loading the model
modeln=load_model("tea.h5")
app=Flask(__name__)

#default home page or route
@app.route('/')
def index():
 return render_template('index.html')






@app.route('/predict',methods=['POST','GET'])
def predict():
  if request.method=="POST":
    f=request.files[ 'image']
    basepath=os.path.dirname (__file__)#getting the current path i
    #print ("current path", basepath)
    filepath=os.path.join(basepath,'uploads', f.filename) #from any
    #print ("upload folder is", filepath)
    f.save(filepath)

    img=image.load_img(filepath,target_size=(180,180))
    x=image.img_to_array(img)#img to array
    x=np.expand_dims(x,axis=0)#used for adding one more dimension
    #print(x)
    img_data=preprocess_input(x)
    prediction=np.argmax(modeln.predict(img_data))
    index=[ 'Anthracnose',
    'algal leaf'
    'bird eye spot',
    'brown blight',
    'gray light',
    'healthy',
    'red leaf spot','white spot']
    nresult=str(index[prediction])
    

    return render_template('teapred.html',prediction=nresult)


"""Running our application"""
if __name__=="__main__":
  app.run(debug =True,port=8080)