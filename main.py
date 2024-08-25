import os
import sqlite3
import joblib
import pickle
import numpy as np
from flask import *
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array

UPLOAD_FOLDER = './database/Uploaded'
DB_PATH = './database/DeepCareX.db'

app = Flask(__name__)
app.secret_key="[$*4^$6$6$]"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Database
## Insert
def insert(name,email,password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    query = f"INSERT INTO USER VALUES('{name}','{email}','{password}')"
    cursor.execute(query)
    conn.commit()
    conn.close()

def insert_contact(name,email,contact,msg):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    query = f"INSERT INTO CONTACT VALUES('{name}','{email}','{contact}','{msg}')"
    cursor.execute(query)
    conn.commit()
    conn.close()

def insert_newsletter(email):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    query = f"INSERT INTO NEWSLETTER VALUES('{email}')"
    cursor.execute(query)
    conn.commit()
    conn.close()

def insert_patients(name,id_,email,contact,country,state,age,gender,disease,result,pin):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    query = f"INSERT INTO PATIENTS VALUES('{name}','{email}','{id_}','{contact}','{country}','{state}','{pin}','{gender}','{age}','{disease}','{result}')"
    cursor.execute(query)
    conn.commit()
    conn.close()

## Search
def search(name,password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    result = cursor.execute(f"SELECT * FROM USER WHERE NAME='{name}' and PASSWORD='{password}'")
    if len(result.fetchall())>0:
        val = True
    else:
        val = False
    conn.close()
    return val

def reg_check(name,email):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    result = cursor.execute(f"SELECT * FROM USER WHERE NAME='{name}' and EMAIL='{email}'")
    if len(result.fetchall())>0:
        val = True
    else:
        val = False
    conn.close()
    return val


## Preprocess_image
def prediction(img_path,model_path):
    model = load_model(model_path)
    img = load_img(img_path,target_size=(128,128))
    img = img_to_array(img)
    img = img.reshape(-1,128,128,3)
    img = img/255.0
    output = model.predict(img)
    return [np.argmax(output),output.max()]


@app.route('/', methods=['GET','POST'])
def home():
    return render_template('home.html')

@app.route('/home',methods=['GET','POST'])
def login():
    new_username = request.form.get('username',False)
    new_useremail = request.form.get('useremail', False)
    new_userpassword = request.form.get('userpassword', False)
    new_userpassword_confirm = request.form.get('confirm_userpassword', False)
    name = request.form.get('name', False)
    password = request.form.get('password', False)
    if name and password and not new_useremail and not new_username and not new_userpassword and not new_userpassword_confirm:
        if  search(name,password):
            flash("logged in successfully")
            session["user"] = name
            return render_template('Alzheimer.html',login_user=session['user'])
        else:
            flash("Invalid Credential, Please try again",'danger')
            return redirect(url_for("home"))
    elif not name and not password and new_username and new_useremail and new_userpassword and new_userpassword_confirm:
        if new_userpassword != new_userpassword_confirm:
            flash('Invalid password confirmation','danger')
            return redirect(url_for("home"))
        elif reg_check(new_username,new_useremail):
            flash('Already a Member, Please login','danger')
            return redirect(url_for("home"))
        else:
            insert(new_username,new_useremail,new_userpassword)
            session["user"] = new_username
            flash('Registerd Successfully')
            return render_template('Alzheimer.html',login_user=session['user'])
    else:
        flash("Invalid Credential, Please try again",'danger')
        return redirect(url_for("home"))

@app.route('/logout',methods=['GET','POST'])
def logout():
    session.pop("user",None)
    return redirect(url_for("home"))

@app.route('/Alzheimer')
def Alzheimer():
    if "user" in session:
        return render_template('Alzheimer.html',login_user=session['user'])
    else:
        return redirect(url_for("home"))

@app.route('/Breast_Cancer')
def Breast_Cancer():
    if "user" in session:
        return render_template('Breast_Cancer.html',login_user=session['user'])
    else:
        return redirect(url_for("home"))

@app.route('/Brain_Tumor')
def Brain_Tumor():
    if "user" in session:
        return render_template('Brain_Tumor.html',login_user=session['user'])
    else:
        return redirect(url_for("home"))

@app.route('/Covid')
def Covid():
    if "user" in session:
        return render_template('Covid-19.html',login_user=session['user'])
    else:
        return redirect(url_for("home"))

@app.route('/Diabetes')
def Diabetes():
    if "user" in session:
        return render_template('Diabetes.html',login_user=session['user'])
    else:
        return redirect(url_for("home"))


@app.route('/Hepatitis')
def Hepatitis():
    if "user" in session:
        return render_template('Hepatitis.html',login_user=session['user'])
    else:
        return redirect(url_for("home"))

@app.route('/Pneumonia')
def Pneumonia():
    if "user" in session:
        return render_template('Pneumonia.html',login_user=session['user'])
    else:
        return redirect(url_for("home"))

@app.route('/Kidney')
def Kidney():
    if "user" in session:
        return render_template('Kidney.html',login_user=session['user'])
    else:
        return redirect(url_for("home"))

@app.route('/Contact')
def Contact():
    return render_template('Contact.html')

@app.route('/Reply', methods=['GET','POST'])
def Reply():
    name = request.form.get('name',False)
    email = request.form.get('email',False)
    contact = request.form.get('contact',False)
    msg = request.form.get('message',False)
    if name and email and contact and msg:
        if len(contact) >= 10 and len(contact)<=13:
            insert_contact(name,email,contact,msg)
            flash('Message Sent Successfully','success')
            return redirect(url_for("Contact"))
        else:
            flash('Contact Number should be between 10-13 digits, if more than 10 digits then first few digits will be COUNTRY CODE','danger')
            return redirect(url_for("Contact"))
    else:     
        flash('Missing Fields','danger')
        return redirect(url_for("Contact"))

@app.route('/About')
def About():
    return render_template('About.html')

@app.route('/Newletter',methods=['GET','POST'])
def Newsletter():
    email = request.form.get('email',False)
    if email:
        insert_newsletter(email)
        flash('Thank you, for taking the subscribtion of our weekly newsletter','success')
        return redirect(url_for("home"))
    else:
        flash('Enter your email id for getting newsletter','danger')
        return redirect(url_for("home"))

@app.route('/Breast_Cancer_Report',methods=['GET','POST'])
def Breast_Cancer_Report():
    name = request.form.get('name',False)
    email = request.form.get('email',False)
    id_ = request.form.get('id',False)
    contact = request.form.get('contact',False)
    country = request.form.get('country',False)
    state = request.form.get('state',False)
    pin = request.form.get('pin',False)
    gender = request.form.get('gender',False)
    age = request.form.get('age',False)
    radius = float(request.form.get('radius',0))
    texture = float(request.form.get('texture',0))
    perimeter = float(request.form.get('perimeter',0))
    area = float(request.form.get('area',0))
    smoothness = float(request.form.get('smoothness',0))
    save = request.form.get('save')
    model = joblib.load('../Models/Breast Cancer/breast_cancer.pkl')
    in_data = np.array([[radius,texture,perimeter,area,smoothness]])
    pred = model.predict(in_data)
    if pred[0] == 0:
        result = "benign"
    elif pred[0] == 1:
        result = "malignant"
    if save == 'on':
        insert_patients(name, id_, email, contact, country, state, age, gender, "Breast Cancer", result, pin)
        flash(f"{name}'s data is saved")
        return render_template('output.html',id_=id_,name=name,age=age,gender=gender,result=result,disease="Breast Cancer")
    else:
        return render_template('output.html',id_=id_,name=name,age=age,gender=gender,result=result,disease="Breast Cancer")

@app.route('/Hepatitis_Report',methods=['GET','POST'])
def Hepatitis_Report():
    name = request.form.get('name',False)
    email = request.form.get('email',False)
    id_ = request.form.get('id',False)
    contact = request.form.get('contact',False)
    country = request.form.get('country',False)
    state = request.form.get('state',False)
    pin = request.form.get('pin',False)
    gender = request.form.get('gender',False)
    if gender == "Male":
        sex = 0
    elif gender == "Female":
        sex = 1
    age = int(request.form.get('age',0))
    alb = float(request.form.get('ALB',0))
    alp = float(request.form.get('ALP',0))
    alt = float(request.form.get('ALT',0))
    ast = float(request.form.get('AST',0))
    bil = float(request.form.get('BIL',0))
    che = float(request.form.get('CHE',0))
    col = float(request.form.get('COL',0))
    crea = float(request.form.get('CREA',0))
    ggt =  float(request.form.get('GGT',0))
    prot = float(request.form.get('PROT',0))
    save = request.form.get('save')
    model = joblib.load('../Models/Hepatitis C/hep.pkl')
    in_data = np.array([[age,sex,alb,alp,alt,ast,bil,che,col,crea,ggt,prot]])
    pred = model.predict(in_data)
    if pred[0] == 0:
        result = "No Hepatitis"
    elif pred[0] == 1:
        result = "Hepatitis C positive"
    if save == 'on':
        insert_patients(name, id_, email, contact, country, state, age, gender, "Hepatitis C", result, pin)
        flash(f"{name}'s data is saved")
        return render_template('output.html',id_=id_,name=name,age=age,gender=gender,result=result,disease="Hepatitis C")
    else:
        return render_template('output.html',id_=id_,name=name,age=age,gender=gender,result=result,disease="Hepatitis C")

@app.route('/Diabetes_Report',methods=['GET','POST'])
def Diabetes_Report():
    name = request.form.get('name',False)
    email = request.form.get('email',False)
    id_ = request.form.get('id',False)
    contact = request.form.get('contact',False)
    country = request.form.get('country',False)
    state = request.form.get('state',False)
    pin = request.form.get('pin',False)
    gender = request.form.get('gender',False)
    if gender == "Male":
        male = 1
        female = other = 0
    elif gender == "Female":
        female = 1
        male = other = 0
    elif gender == "Others":
        other = 1
        male = female =0
    age = float(request.form.get('age',0))
    hyp = request.form.get('hyp')
    if hyp == "No":
        hypt = 0
    else:
        hypt = 1
    hd = request.form.get('hd')
    if hd == "No":
        hedi = 0
    else:
        hedi = 1
    bmi = float(request.form.get('bmi'))
    hemo = float(request.form.get('hemo'))
    blood = float(request.form.get('blood'))
    smoke = request.form.get('smoke')
    if smoke == "Never":
        never = 1
        no_info = not_current = current =0
    elif smoke == "No Info":
        no_info = 1
        never = not_current = current =0
    elif smoke == "Not Current":
        not_current = 1
        no_info = never = current =0
    elif smoke == "Current":
        current = 1
        no_info = not_current = never =0
    save = request.form.get('save')     
    model = joblib.load("../Models/Diabetes/diab_xg1.pkl")
    in_data = np.array([[age,hypt,hedi,bmi,hemo,blood,current,never,no_info,not_current,female,male,other]])
    pred = model.predict(in_data)
    if pred[0] == 0:
        result = "No Diabetes"
    elif pred[0] == 1:
        result = "Diabetes Positive"
    if save == 'on':
        insert_patients(name, id_, email, contact, country, state, int(age), gender, "Diabetes", result, pin)
        flash(f"{name}'s data is saved")
        return render_template('output.html',id_=id_,name=name,age=int(age),gender=gender,result=result,disease="Diabetes")
    else:
        return render_template('output.html',id_=id_,name=name,age=int(age),gender=gender,result=result,disease="Diabetes")


@app.route('/Alzheimer_Report',methods=['GET','POST'])
def Alzheimer_Report():
    name = request.form.get('name',False)
    email = request.form.get('email',False)
    id_ = request.form.get('id',False)
    contact = request.form.get('contact',False)
    country = request.form.get('country',False)
    state = request.form.get('state',False)
    pin = request.form.get('pin',False)
    gender = request.form.get('gender',False)
    age = request.form.get('age',False)
    img = request.files['image']
    save = request.form.get('save')
    filename = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(img.filename))
    img.save(filename)
    pred = prediction(filename,'../Models/Alzheimer/Alzheimer_CNN.hdf5')
    if pred[0] == 0:
        result = f"MildDemented ({pred[1]*100:.2f}%)"
    elif pred[0] == 1:
        result = f"ModerateDemented ({pred[1]*100:.2f}%)"
    elif pred[0] == 2:
        result = f"NonDemented ({pred[1]*100:.2f}%)"
    elif pred[0] == 3:
        result = f"VeryMildDemented ({pred[1]*100:.2f}%)"
    if save == 'on':
        insert_patients(name, id_, email, contact, country, state, age, gender, "Alzheimer", result, pin)
        flash(f"{name}'s data is saved")
        return render_template('output.html',id_=id_,name=name,age=age,gender=gender,result=result,disease="Alzheimer")
    else:
        return render_template('output.html',id_=id_,name=name,age=age,gender=gender,result=result,disease="Alzheimer")

@app.route('/Brain_Tumor_Report',methods=['GET','POST'])
def Brain_Tumor_Report():
    name = request.form.get('name',False)
    email = request.form.get('email',False)
    id_ = request.form.get('id',False)
    contact = request.form.get('contact',False)
    country = request.form.get('country',False)
    state = request.form.get('state',False)
    pin = request.form.get('pin',False)
    gender = request.form.get('gender',False)
    age = request.form.get('age',False)
    img = request.files['image']
    save = request.form.get('save')
    filename = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(img.filename))
    img.save(filename)
    pred = prediction(filename,'../Models/Brain_Tumor/Brain_Tumor_VGG19.hdf5')
    if pred[0] == 0:
        result = f"Glioma ({pred[1]*100:.2f}%)"
    elif pred[0] == 1:
        result = f"Meningioma ({pred[1]*100:.2f}%)"
    elif pred[0] == 2:
        result = f"No tumor ({pred[1]*100:.2f}%)"
    elif pred[0] == 3:
        result = f"Pituitary ({pred[1]*100:.2f}%)"
    if save == 'on':
        insert_patients(name, id_, email, contact, country, state, age, gender, "Brain Tumor", result, pin)
        flash(f"{name}'s data is saved")
        return render_template('output.html',id_=id_,name=name,age=age,gender=gender,result=result,disease="Brain Tumor")
    else:
        return render_template('output.html',id_=id_,name=name,age=age,gender=gender,result=result,disease="Brain Tumor")

@app.route('/Covid_Report',methods=['GET','POST'])
def Covid_Report():
    name = request.form.get('name',False)
    email = request.form.get('email',False)
    id_ = request.form.get('id',False)
    contact = request.form.get('contact',False)
    country = request.form.get('country',False)
    state = request.form.get('state',False)
    pin = request.form.get('pin',False)
    gender = request.form.get('gender',False)
    age = request.form.get('age',False)
    img = request.files['image']
    save = request.form.get('save')
    filename = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(img.filename))
    img.save(filename)
    pred = prediction(filename,'../Models/COVID/Covid.hdf5')
    if pred[0] == 0:
        result = f"COVID Positive ({pred[1]*100:.2f}%)"
    elif pred[0] == 1:
        result = f"COVID Negative ({pred[1]*100:.2f}%)"
    if save == 'on':
        insert_patients(name, id_, email, contact, country, state, age, gender, "Covid-19", result, pin)
        flash(f"{name}'s data is saved")
        return render_template('output.html',id_=id_,name=name,age=age,gender=gender,result=result,disease="Covid CT-Scan")
    else:
        return render_template('output.html',id_=id_,name=name,age=age,gender=gender,result=result,disease="Covid CT-Scan")

@app.route('/Pneumonia_Report',methods=['GET','POST'])
def Pneumonia_Report():
    name = request.form.get('name',False)
    email = request.form.get('email',False)
    id_ = request.form.get('id',False)
    contact = request.form.get('contact',False)
    country = request.form.get('country',False)
    state = request.form.get('state',False)
    pin = request.form.get('pin',False)
    gender = request.form.get('gender',False)
    age = request.form.get('age',False)
    img = request.files['image']
    save = request.form.get('save')
    filename = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(img.filename))
    img.save(filename)
    pred = prediction(filename,'../Models/Pneumonia/Pneumonia_DenseNet201.hdf5')
    if pred[0] == 0:
        result = f"NORMAL ({pred[1]*100:.2f}%)"
    elif pred[0] == 1:
        result = f"PNEUMONIA ({pred[1]*100:.2f}%)"
    if save == 'on':
        insert_patients(name, id_, email, contact, country, state, age, gender, "Pneumonia", result, pin)
        flash(f"{name}'s data is saved")
        return render_template('output.html',id_=id_,name=name,age=age,gender=gender,result=result,disease="Pneumonia")
    else:
        return render_template('output.html',id_=id_,name=name,age=age,gender=gender,result=result,disease="Pneumonia")

@app.route('/Kidney_Report',methods=['POST','GET'])
def Kidney_Report():
    name = request.form.get('name',False)
    email = request.form.get('email',False)
    id_ = request.form.get('id',False)
    contact = request.form.get('contact',False)
    country = request.form.get('country',False)
    state = request.form.get('state',False)
    pin = request.form.get('pin',False)
    gender = request.form.get('gender',False)
    age = request.form.get('age',False)
    img = request.files['image']
    save = request.form.get('save')
    filename = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(img.filename))
    img.save(filename)
    pred = prediction(filename,'../Models/Kidney/Kidney.hdf5')
    if pred[0] == 0:
        result = f"Kidney Cyst ({pred[1]*100:.2f}%)"
    elif pred[0] == 1:
        result = f"Normal ({pred[1]*100:.2f}%)"
    elif pred[0] == 2:
        result = f"Kidney Stone ({pred[1]*100:.2f}%)"
    elif pred[0] == 3:
        result = f"Kidney Tumor ({pred[1]*100:.2f}%)"
    if save == 'on':
        insert_patients(name, id_, email, contact, country, state, age, gender, "Kidney Disease", result, pin)
        flash(f"{name}'s data is saved")
        return render_template('output.html',id_=id_,name=name,age=age,gender=gender,result=result,disease="Kidney Disease")
    else:
        return render_template('output.html',id_=id_,name=name,age=age,gender=gender,result=result,disease="Kidney Disease")

@app.route('/Report')
def result():
    return render_template('output.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000, host="0.0.0.0")