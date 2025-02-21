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
    
    data = []
    rows = soup.find_all('th', class_='text-center align-middle')
    st.write(f"Found {len(rows)} rows")

    for row in rows:
        logo_imgs = row.find_all('img')
        logos = [os.path.basename(img['src']).replace('.png', '') if img else 'N/A' for img in logo_imgs]

        percentages_divs = row.find_all('div', class_=['d-flex', 'align-items-center', 'justify-content-center'])
        percentages = [div.get_text(strip=True).replace('%', '') if div else 'N/A' for div in percentages_divs]
        pick_team = logos[0] if len(percentages) == 2 and int(percentages[0]) > int(percentages[1]) else logos[1] if len(logos) == 2 else 'Draw'

        odds_divs = row.find_all('div', class_='pointer header-row-data height-small font-small')
        odds = [div.get_text(strip=True).replace('+', '') if div else 'N/A' for div in odds_divs]

        try:
            odds = [int(o) if o != 'N/A' else 0 for o in odds]
        except ValueError:
            odds = [0, 0]
        
        scores_divs = row.find_all('div', class_='text-bolder header-row-data height-small')
        scores = [div.get_text(strip=True) for div in scores_divs] if scores_divs else ['N/A', 'N/A']
        winner_team = logos[0] if len(scores) == 2 and int(scores[0]) > int(scores[1]) else logos[1] if len(logos) == 2 else 'Draw'

        pick_profit = None
        if pick_team == logos[0]:
            pick_profit = 100 * (odds[0] / 100) if odds[0] > 0 else 100 / (abs(odds[0]) / 100)
        elif pick_team == logos[1]:
            pick_profit = 100 * (odds[1] / 100) if odds[1] > 0 else 100 / (abs(odds[1]) / 100)
        else:
            pick_profit = 0

        result_profit = pick_profit if pick_team == winner_team else -100 if winner_team != "DRAW" else 0

        if len(logos) == 2 and len(odds) == 2 and len(percentages) == 2:
            data.append([game_date, logos[0], logos[1], odds[0], odds[1], pick_team, winner_team, pick_profit, result_profit])

    return pd.DataFrame(data, columns=["Date", "Home Team", "Away Team", "Home Odds", "Away Odds", "Pick", "Result", "Pick Profit", "Result Profit"])

uploaded_file = st.file_uploader("Upload NHL HTML file", type=["html"])

if uploaded_file:
    df = extract_values_from_html(uploaded_file)
    st.write("### 📊 Scraped Results")
    st.dataframe(df, use_container_width=True)
