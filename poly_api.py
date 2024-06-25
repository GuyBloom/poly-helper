from datetime import timedelta
from flask import Flask, request, render_template, render_template_string, session, url_for, redirect
from flask_session import Session
import pandas as pd
from deconstructor import final_get_answer 
import numpy as np
import os
from dotenv import load_dotenv, dotenv_values 
import redis


app = Flask(__name__)
# app.secret_key = 'skibidi_biden'


app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
Session(app)

server_session = Session(app)

@app.route('/', methods=['GET', 'POST']) 
def enter_text():
    if request.method == 'POST':
        # Save the form data to the session object
        session['text_input'] = request.form.get('textarea')
        print(session['text_input'])
        return redirect(url_for('get_text'))
    return render_template('home.html')

@app.route('/get_text', methods=('GET', 'POST'))
def get_text(paragraph=None):
    answer = final_get_answer(session['text_input'], debug=True)
    paragraph=answer
    data_table = pd.read_csv('results_table.csv')
    return render_template('results.html', paragraph=paragraph, tables = [data_table.to_html(classes='data')], titles=data_table.columns.values)



