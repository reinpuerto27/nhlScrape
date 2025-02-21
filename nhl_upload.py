import streamlit as st
import os
import pandas as pd
from bs4 import BeautifulSoup
import re

st.title("🏒 NHL Scraper - Upload HTML File")

def extract_values_from_html(file):
    soup = BeautifulSoup(file, 'html.parser')
    
    url_comment = soup.find(string=re.compile("saved from url"))
    match = re.search(r'(\d{4}-\d{2}-\d{2})', url_comment) if url_comment else None
    game_date = match.group(1) if match else "Unknown Date"
    
    st.write(f"📅 Extracted Date: {game_date}")

    data = []
    
    game_rows = soup.find_all('th', class_='text-center align-middle')
    st.write(f"Found {len(game_rows)} games")
    
    game_details = []
    
    for row in game_rows:
        logo_imgs = row.find_all('img')
        logos = [os.path.basename(img['src']).replace('.png', '') if img and 'whole_files' in img['src'] else 'N/A' for img in logo_imgs]

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
        
        pick_imgs = user_row.find_all('img')
        pick_teams = [os.path.basename(img['src']).replace('.png', '') for img in pick_imgs if len(os.path.basename(img['src'])) == 7 and img['src'].endswith('.png')]
        
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

    return pd.DataFrame(data, columns=["Date", "Home Team", "Away Team", "Home Odds", "Away Odds", "User", "Pick", "Result", "Pick Profit", "Result Profit"])

uploaded_file = st.file_uploader("Upload NHL HTML file", type=["html"])

if uploaded_file:
    df = extract_values_from_html(uploaded_file)
    st.write("### 📊 Scraped Results")
    st.dataframe(df, height=600, use_container_width=True)
