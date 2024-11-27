import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_mail import Mail, Message
from flask import send_from_directory
import os
import random
from bs4 import BeautifulSoup
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import joblib
from flask import abort, request




fighter_profile = pd.read_csv('model/Fighter Profile CSV.csv')
fight_history = pd.read_csv('model/Fight History CSV.csv')
scaler = joblib.load('model/scaler.pkl')
custom_objects = {
    "ExponentialDecay": tf.keras.optimizers.schedules.ExponentialDecay,
    # Include any other custom objects here
}
with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model('model/trained_model.h5')


# Function to calculate common opponents' performance
def common_opponents_performance(fight_history, fighter1_id, fighter2_id):
    fighter1_opponents = fight_history[
        (fight_history['Fighter1_ID'] == fighter1_id) | (fight_history['Fighter2_ID'] == fighter1_id)]
    fighter2_opponents = fight_history[
        (fight_history['Fighter1_ID'] == fighter2_id) | (fight_history['Fighter2_ID'] == fighter2_id)]

    common_opponents = set(fighter1_opponents['Fighter1_ID']) & set(
        fighter2_opponents['Fighter1_ID']) | set(fighter1_opponents['Fighter2_ID']) & set(
        fighter2_opponents['Fighter2_ID'])
    common_opponents.discard(fighter1_id)
    common_opponents.discard(fighter2_id)

    fighter1_wins = 0
    fighter2_wins = 0

    for opponent in common_opponents:
        fighter1_wins += len(fighter1_opponents[fighter1_opponents['Winner_ID'] == fighter1_id])
        fighter2_wins += len(fighter2_opponents[fighter2_opponents['Winner_ID'] == fighter2_id])

    return fighter1_wins - fighter2_wins


# Function to calculate feature differences
def calculate_stat_difference(fight_history, fighter_profile):
    diff_data = []

    for _, fight in fight_history.iterrows():
        fighter1_id = fight['Fighter1_ID']
        fighter2_id = fight['Fighter2_ID']

        # Randomly swap fighter order
        if random.randint(0, 1):
            fighter1_id, fighter2_id = fighter2_id, fighter1_id

        fighter1 = fighter_profile[fighter_profile['ID'] == fighter1_id].iloc[0]
        fighter2 = fighter_profile[fighter_profile['ID'] == fighter2_id].iloc[0]

        # Calculate feature differences
        win_diff = fighter1['Win'] - fighter2['Win']
        loss_diff = fighter1['Loss'] - fighter2['Loss']
        draw_diff = fighter1['Draw'] - fighter2['Draw']

        # Added differences for the new statistics
        SLpM_diff = fighter1['SLpM'] - fighter2['SLpM']
        Str_Acc_diff = fighter1['Str_Acc'] - fighter2['Str_Acc']
        SApM_diff = fighter1['SApM'] - fighter2['SApM']
        Str_Def_diff = fighter1['Str_Def'] - fighter2['Str_Def']
        TD_Avg_diff = fighter1['TD_Avg'] - fighter2['TD_Avg']
        TD_Acc_diff = fighter1['TD_Acc'] - fighter2['TD_Acc']
        TD_Def_diff = fighter1['TD_Def'] - fighter2['TD_Def']
        Sub_Avg_diff = fighter1['Sub_Avg'] - fighter2['Sub_Avg']

        winner = 1 if fight['Winner_ID'] == fighter1_id else 0

        # Calculate common opponents' performance
        common_opponents_diff = common_opponents_performance(fight_history, fighter1_id, fighter2_id)

        diff_data.append([win_diff, loss_diff, draw_diff,
                          SLpM_diff, Str_Acc_diff, SApM_diff, Str_Def_diff,
                          TD_Avg_diff, TD_Acc_diff, TD_Def_diff, Sub_Avg_diff,
                          common_opponents_diff,
                          winner])
    return pd.DataFrame(diff_data, columns=['Win_diff', 'Loss_diff', 'Draw_diff',
                                            'SLpM_diff', 'Str_Acc_diff', 'SApM_diff', 'Str_Def_diff',
                                            'TD_Avg_diff', 'TD_Acc_diff', 'TD_Def_diff', 'Sub_Avg_diff',
                                            'Common_Opponents_diff',  # Add the new feature name to the columns
                                            'Winner'])

def make_prediction(fighter1, fighter2):
    # Normalize the fighter names to ensure consistent lookup
    fighter1 = fighter1.lower().strip()
    fighter2 = fighter2.lower().strip()

    # Fetch fighter stats
    fighter1_stats = fighter_profile.loc[fighter_profile['Name'].str.lower().str.strip() == fighter1].iloc[0]
    fighter2_stats = fighter_profile.loc[fighter_profile['Name'].str.lower().str.strip() == fighter2].iloc[0]

    # Sort fighters alphabetically to ensure consistent order
    if fighter1_stats['Name'] > fighter2_stats['Name']:
        fighter1_stats, fighter2_stats = fighter2_stats, fighter1_stats

    # Calculate feature differences
    input_data = [
        fighter1_stats['Win'] - fighter2_stats['Win'],
        fighter1_stats['Loss'] - fighter2_stats['Loss'],
        fighter1_stats['Draw'] - fighter2_stats['Draw'],
        fighter1_stats['SLpM'] - fighter2_stats['SLpM'],
        fighter1_stats['Str_Acc'] - fighter2_stats['Str_Acc'],
        fighter1_stats['SApM'] - fighter2_stats['SApM'],
        fighter1_stats['Str_Def'] - fighter2_stats['Str_Def'],
        fighter1_stats['TD_Avg'] - fighter2_stats['TD_Avg'],
        fighter1_stats['TD_Acc'] - fighter2_stats['TD_Acc'],
        fighter1_stats['TD_Def'] - fighter2_stats['TD_Def'],
        fighter1_stats['Sub_Avg'] - fighter2_stats['Sub_Avg'],
        common_opponents_performance(fight_history, fighter1_stats['ID'], fighter2_stats['ID'])
    ]

    # Add interaction features
    input_data.append(input_data[3] * input_data[4])  # SLpM_diff_x_Str_Acc_diff
    input_data.append(input_data[7] * input_data[8])  # TD_Avg_diff_x_TD_Acc_diff
    input_data.append(input_data[7] * input_data[9])  # TD_Avg_diff_x_TD_Def_diff

    # Convert input_data to a pandas DataFrame
    input_df = pd.DataFrame([input_data], columns=[
        'Win_diff', 'Loss_diff', 'Draw_diff',
        'SLpM_diff', 'Str_Acc_diff', 'SApM_diff', 'Str_Def_diff',
        'TD_Avg_diff', 'TD_Acc_diff', 'TD_Def_diff', 'Sub_Avg_diff',
        'Common_Opponents_diff',
        'SLpM_diff_x_Str_Acc_diff', 'TD_Avg_diff_x_TD_Acc_diff', 'TD_Avg_diff_x_TD_Def_diff'
    ])

    # Apply the scaler to the input features
    processed_input = scaler.transform(input_df)

    # Make the prediction using the pre-trained model
    prediction = model.predict(processed_input)

    # Determine the predicted winner and the probability of the prediction
    predicted_winner = fighter1_stats['Name'] if prediction[0][0] >= 0.5 else fighter2_stats['Name']
    prediction_probability = prediction[0][0] if prediction[0][0] >= 0.5 else 1 - prediction[0][0]

    return predicted_winner, prediction_probability






class BlogPost:
    def __init__(self, filename, title, date, preview):
        self.filename = filename
        self.title = title
        self.date = date
        self.preview = preview


def extract_title_from_html(content):
    soup = BeautifulSoup(content, 'html.parser')
    title_element = soup.find('h1')
    if title_element:
        return title_element.text
    return ''


def extract_date_from_html(content):
    soup = BeautifulSoup(content, 'html.parser')
    date_element = soup.find('div', class_='post-date')
    if date_element:
        return date_element.text
    return ''


def extract_preview_from_html(content):
    soup = BeautifulSoup(content, 'html.parser')
    preview_element = soup.find('div', class_='post-preview')
    if preview_element:
        return preview_element.text
    return ''
    


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'email' in request.form:
            email = request.form['email']

            # Send an email
            msg = Message("New user signed up",
                          sender="your_email@gmail.com",
                          recipients=["danieltolmienz@gmail.com"])
            msg.body = f"User with email {email} has signed up for announcements."
            mail.send(msg)

            return "Email saved"
    return render_template('home.html')


@app.route('/validate_password', methods=['POST'])
def validate_password():
    password = request.form.get('password')
    if password == 'helloworld':
        # Render the predictor page (index.html)
        return render_template('index.html')
    else:
        # Redirect back to the home page and show an error message
        flash('Incorrect password. Please try again.')
        return redirect(url_for('home'))


@app.route('/testimonials')
def testimonials():
    return render_template('testimonials.html')


@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/ufc')
def ufc():
    return render_template('ufc.html')
    
@app.route('/soon')
def soon():
    return render_template('soon.html')


@app.route('/signup', methods=['POST'])
def signup():
    email = request.form.get('email')

    # Send the email address to your email
    msg = Message("New user signed up",
                  sender="your_email@gmail.com",
                  recipients=["danieltolmienz@gmail.com"])
    msg.body = f"User with email {email} has signed up for announcements."
    mail.send(msg)

    # Redirect the user to a confirmation page or back to the home page
    flash('Thank you for signing up! You will receive updates and announcements.')
    return redirect(url_for('home'))


@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('query')
    suggestions = fighter_profile[fighter_profile['Name'].str.contains(query, case=False)]['Name'].tolist()
    return jsonify(suggestions)


@app.route('/sitemap.xml', methods=['GET'])
def sitemap():
    with open('sitemap.xml', 'r') as file:
        sitemap_data = file.read()
    response = app.response_class(sitemap_data, content_type='application/xml')
    return response


@app.route('/blog')
def blog():
    blog_posts = []
    for filename in os.listdir('output'):
        if filename.endswith('.html'):
            filepath = os.path.join('output', filename)
            with open(filepath, 'r') as file:
                content = file.read()
                title = extract_title_from_html(content)
                date = extract_date_from_html(content)
                preview = extract_preview_from_html(content)
                post = BlogPost(filename, title, date, preview)
                blog_posts.append(post)
    return render_template('blog.html', blog_posts=blog_posts)


@app.route('/output/<path:filename>')
def custom_static(filename):
    return send_from_directory('output', filename)



@app.route('/images/<path:path>')
def send_images(path):
    return send_from_directory('images', path)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.form
        fighter1 = data['fighter1']
        fighter2 = data['fighter2']

        predicted_winner, prediction_probability = make_prediction(fighter1, fighter2)
        result = {
            'winner': predicted_winner,
            'probability': float(prediction_probability)
        }

        return jsonify(result)
    return render_template('index.html')

@app.before_request
def block_malicious_paths():
    blocked_paths = ["/xmlrpc.php", "/admin.php", "/pegi.php"]
    if request.path in blocked_paths:
        return abort(403)  # Forbidden

@app.route('/fightnight')
def fightnight():
    fights = [
        {"fighter1": "Shavkat Rakhmonov", "fighter2": "Ian Garry"},
        {"fighter1": "Ciryl Gane", "fighter2": "Alexander Volkov"},
        {"fighter1": "Bryce Mitchell", "fighter2": "Kron Gracie"},
        {"fighter1": "Nate Landwehr", "fighter2": "Dooho Choi"},
        {"fighter1": "Dominick Reyes", "fighter2": "Anthony Smith"},
        {"fighter1": "Themba Gorimbo", "fighter2": "Vicente Luque"},
        {"fighter1": "Bryan Battle", "fighter2": "Randy Brown"},
        {"fighter1": "Movsar Evloev", "fighter2": "Aljamain Sterling"},        
        {"fighter1": "Chris Weidman", "fighter2": "Eryk Anders"},
        {"fighter1": "Michael Chiesa", "fighter2": "Max Griffin"},
        {"fighter1": "Clay Guida", "fighter2": "Chase Hooper"},
        {"fighter1": "Cody Durden", "fighter2": "Joshua Van"},
        {"fighter1": "Tallison Teixeira", "fighter2": "Lukasz Brzeski"},
        
        # Add more fights as needed
    ]

    predictions = []
    for fight in fights:
        predicted_winner, prediction_probability = make_prediction(fight["fighter1"], fight["fighter2"])
        predictions.append({
            "fighter1": fight['fighter1'],
            "fighter2": fight['fighter2'],
            "winner": predicted_winner,
            "probability": prediction_probability * 100  # convert to percentage
        })

    return render_template('fightnight.html', predictions=predictions)



if __name__ == '__app__':
   # (host='0.0.0.0', port=8081, debug=True)
   app.run
