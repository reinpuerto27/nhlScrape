import streamlit as st
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from bs4 import BeautifulSoup
import re

st.title("🏒 NHL Scraper")

def extract_values_from_html(file, filename):
    soup = BeautifulSoup(file, 'html.parser')
    
    url_comment = soup.find(string=re.compile("saved from url"))
    match = re.search(r'(\d{4}-\d{2}-\d{2})', url_comment) if url_comment else None
    game_date = match.group(1) if match else filename

    data = []
    
    game_rows = soup.find_all('th', class_='text-center align-middle')
    
    game_details = []
    
    for row in game_rows:
        logo_imgs = row.find_all('img')
        logos = [os.path.basename(img['src']).replace('.png', '') if img and img['src'].endswith('.png') else 'N/A' for img in logo_imgs]

        odds_divs = row.find_all('div', class_='pointer header-row-data height-small font-small')
        odds = [div.get_text(strip=True).replace('+', '') if div else 'N/A' for div in odds_divs]

        try:
            odds = [int(o) if o.lstrip('-').isdigit() else 0 for o in odds]
        except ValueError:
            odds = [0, 0] 

        scores_divs = row.find_all('div', class_='text-bolder header-row-data height-small')
        scores = [div.get_text(strip=True) for div in scores_divs] if scores_divs else ['N/A', 'N/A']
        
        winner_team = logos[0] if len(scores) == 2 and int(scores[0]) > int(scores[1]) else logos[1] if len(logos) == 2 else 'Draw'
        game_details.append((logos[0], logos[1], odds[0], odds[1], winner_team))
    
    user_rows = soup.find_all('tr')

    user_data = []

    for user_row in user_rows:
        username_div = user_row.find('td', class_='username-container')
        username = username_div.get_text(strip=True) if username_div else "Unknown User"

        pick_cells = user_row.find_all('td', class_=re.compile(r'static-picks-col-width'))
        pick_teams = []

        for cell in pick_cells:
            img = cell.find('img')
            if img:
                team_abbr = os.path.basename(img['src']).replace('.png', '')
            else:
                no_pick_div = cell.find('div', class_="no-pick d-flex justify-content-center")
                team_abbr = "N/A" if no_pick_div else "Unknown"

            pick_teams.append(team_abbr)

        while len(pick_teams) < len(game_details):
            pick_teams.append("N/A")
        
        if username != "Unknown User" and len(pick_teams) == len(game_details):
            user_data.append((username, pick_teams))

    for user, pick_teams in user_data:
        for i, game in enumerate(game_details):
            pick_team = pick_teams[i] if i < len(pick_teams) else "Unknown"
            
            pick_profit = 0
            if pick_team == game[0]:
                pick_profit = 100 * (game[2] / 100) if game[2] > 0 else 100 / (abs(game[2]) / 100)
            elif pick_team == game[1]:
                pick_profit = 100 * (game[3] / 100) if game[3] > 0 else 100 / (abs(game[3]) / 100)
            
            result_profit = pick_profit if pick_team == game[4] else -100 if game[4] != "Draw" else 0
            data.append([game_date, game[0], game[1], game[2], game[3], user, pick_team, game[4], round(pick_profit, 2), round(result_profit, 2)])

    return data

uploaded_files = st.file_uploader("Upload NHL HTML files", type=["html"], accept_multiple_files=True)

if uploaded_files:
    all_data = []
    for uploaded_file in uploaded_files:
        file_data = extract_values_from_html(uploaded_file, uploaded_file.name)
        all_data.extend(file_data)
    
    df = pd.DataFrame(all_data, columns=["Date", "Home Team", "Away Team", "Home Odds", "Away Odds", "User", "Pick", "Result", "Pick Profit", "Result Profit"])
    
    st.write("📊 Scraped Results")
    st.dataframe(df, use_container_width=True)

    if st.button("Predict Winner"):
            df_original = df[["Date", "Home Team", "Away Team", "Result"]]
            df = df.drop(columns=["Date"], errors="ignore")

            label_encoders = {}
            for col in ["Home Team", "Away Team", "User"]:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le

            X = df[["Home Team", "Away Team", "Home Odds", "Away Odds", "User"]]
            y = label_encoders["Home Team"].fit_transform(df["Result"])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
            model.fit(X_train, y_train)
            df["Predicted Result"] = model.predict(X)
            df["Predicted Result"] = label_encoders["Home Team"].inverse_transform(df["Predicted Result"])

            df_output = df_original.copy()
            df_output["Predicted Result"] = df["Predicted Result"]
            df_output = df_output.groupby("Date").first().reset_index()

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

            st.write("Predicted Winners:")
            st.dataframe(df_output)