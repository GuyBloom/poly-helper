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
def home():
    if request.method == 'POST':
        # Save the form data to the session object
        session['text_input'] = request.form.get('textarea')
        print(session['text_input'])
        return redirect(url_for('get_text'))
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/get_text', methods=('GET', 'POST'))
def get_text(paragraph=None):
    answer = final_get_answer(session['text_input'], debug=True)
    paragraph=answer
    data_table = pd.read_csv('results_table.csv')
    name_list = data_table['name'].values
    exp_list = data_table['expl'].values
    perc_list = data_table['percent_match'].values
    return render_template('viewresults.html', names = name_list, expls = exp_list, matches = perc_list)

@app.route('/test_results')
def test_results(paragraph= "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse rhoncus augue neque, nec malesuada justo tempus ut. Phasellus efficitur, ante condimentum pulvinar maximus, nulla est malesuada dui, quis gravida metus justo at eros. Phasellus non cursus ligula. Aliquam erat volutpat. Donec semper dolor sollicitudin, finibus turpis egestas, scelerisque magna. Phasellus faucibus erat lorem, vitae varius risus commodo vel. Curabitur tempus quis ante et tempus. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec non erat ullamcorper orci ultrices lobortis et nec sem. Suspendisse sed est molestie nunc sollicitudin sollicitudin. Aenean accumsan nec nisi laoreet ornare. Phasellus eget feugiat neque. Proin maximus lacinia pharetra. Sed vitae nulla in eros tempor vehicula in sit amet massa. Sed ullamcorper posuere pretium."):
    data_table = pd.read_csv('test_results.csv')
    name_list = data_table['name'].values
    exp_list = data_table['expl'].values
    perc_list = data_table['percent_match'].values
    return render_template('testresults.html', paragraph=paragraph, names = name_list, expls = exp_list, matches = perc_list)



