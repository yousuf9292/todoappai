from flask import Flask,render_template,url_for,redirect,session,request,jsonify,url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import FloatField,SubmitField,TextField,BooleanField
from wtforms.validators import InputRequired
import numpy as np
import joblib
from keras.models import load_model


def return_predictions(model,scaler,sample_json):
	milk=sample_json["Milk"]
	tea=sample_json["Tea"]
	coffe=sample_json["Coffe"]
	onion=sample_json["Onion"]
	yogurt=sample_json["Yogurt"]
	bread=sample_json["Bread"]
	ketchup=sample_json["Ketchup"]
	egg=sample_json["Egg"]
	fish=sample_json["Fish"]
	softdrinks=sample_json["SoftDrink"]
	rice=sample_json["Rice"]
	months=sample_json["periods"]

	data=[[milk,tea,coffe,onion,yogurt,bread,ketchup,egg,fish,softdrinks,rice]]
	scaled_data=scaler.transform(data)

	length=1
	n_features=11
	forecast=[]
	first_eval_batch=scaled_data[-length:]
	current_batch=first_eval_batch.reshape((1,length,n_features))

	for i in range(months):
		current_pred=model.predict(current_batch)[0]
		forecast.append(current_pred)
		current_batch=np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
	return np.round(scaler.inverse_transform(forecast)).tolist()




app= Flask(__name__)
app.config['SECRET_KEY']="this must be secret"
Bootstrap(app)


class Form(FlaskForm):
	periods=FloatField('Months',validators=[InputRequired(message='number only')])
	milk=BooleanField("Milk")
	tea=BooleanField("Tea")
	coffe=BooleanField("Coffee")
	onion=BooleanField("Onion")
	yogurt=BooleanField("Yogurt")
	bread=BooleanField("Bread")
	ketchup=BooleanField("Ketchup")
	egg=BooleanField("Egg")
	fish=BooleanField("Fish")
	softdrinks=BooleanField("SoftDrink")
	rice=BooleanField("Rice")
	submit=SubmitField('Submit')



@app.route('/form',methods=['GET','POST'])
def form():
	form=Form()
	if form.validate_on_submit():
		session['periods']=int(form.periods.data)
		session['milk']=float(form.milk.data)
		session['tea']=float(form.tea.data)
		session['coffe']=float(form.coffe.data)
		session['onion']=float(form.onion.data)
		session['yogurt']=float(form.yogurt.data)
		session['bread']=float(form.bread.data)
		session['ketchup']=form.ketchup.data
		session['egg']=float(form.egg.data)
		session['fish']=float(form.fish.data)
		session['softdrinks']=float(form.softdrinks.data)
		session['rice']=float(form.rice.data)
		
		return redirect(url_for('predictions'))
	return render_template("form.html",form=form)

model=load_model('C:/Users/yousuf/Data.h5')
scaler_model=joblib.load('C:/Users/yousuf/Data.pkl')


@app.route('/predictions')
def predictions():
	content={}

	content['periods']=session['periods']
	content['Milk']=session['milk']
	content['Tea']=session['tea']
	content['Coffe']=session['coffe']
	content['Onion']=session['onion']
	content['Yogurt']=session['yogurt']
	content['Bread']=session['bread']
	content['Ketchup']=session['ketchup']
	content['Egg']=session['egg']
	content['Fish']=session['fish']
	content['SoftDrink']=session['softdrinks']
	content['Rice']=session['rice']

	results=return_predictions(model,scaler_model,content)

	return render_template('prediction.html',results=results,content=content)


@app.route('/api',methods=['POST'])
def api():
	content=request.json
	results=return_predictions(model,scaler_model,content)
	return jsonify({'results':results})

if __name__ == '__main__':
	app.run(debug=True)