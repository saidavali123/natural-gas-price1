#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask App for LPG and HP Gas Price Prediction
(Random Forest based on dataset features)
Converts predicted INR -> USD using a live exchange rate (with fallback).
"""

from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
import requests
from datetime import datetime

app = Flask(__name__, static_url_path='/static')

# -----------------------------
# Config / fallback exchange rate
# -----------------------------
# Fallback (mid-market) INR -> USD rate used if API fails.
# This value is taken from recent exchange-data providers (example mid-market ~0.0113613).
FALLBACK_INR_TO_USD = 0.0113613  # 1 INR = 0.0113613 USD (example fallback)

EXCHANGE_API_URL = "https://api.exchangerate.host/latest"  # free, no-key API

def get_inr_to_usd_rate():
    """
    Try to fetch live INR -> USD mid-market rate.
    Returns float (USD per 1 INR).
    On failure, returns FALLBACK_INR_TO_USD.
    """
    try:
        # Query exchangerate.host - request base INR, ask for USD
        resp = requests.get(EXCHANGE_API_URL, params={"base": "INR", "symbols": "USD"}, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        # Example response: {"motd":..., "success":true, "base":"INR", "date":"2025-10-18", "rates":{"USD":0.01136}}
        rate = data.get("rates", {}).get("USD")
        if rate is None:
            raise ValueError("No USD rate in response")
        return float(rate)
    except Exception as e:
        # Log the error and return fallback
        print(f"⚠️ Warning: could not fetch live rate ({e}). Using fallback {FALLBACK_INR_TO_USD}")
        return float(FALLBACK_INR_TO_USD)


# -----------------------------
# 1. Load and Train Models
# -----------------------------
def train_models():
    try:
        data_path = os.path.join("data", "natural gas.csv")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        # Load dataset
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()

        # Validate columns
        if "Date" not in df.columns or "LPG_Price" not in df.columns or "HP_Price" not in df.columns:
            raise ValueError("CSV must contain 'Date', 'LPG_Price', and 'HP_Price' columns.")

        # Clean and prepare data
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"])

        # Extract temporal features
        df["Year"] = df["Date"].dt.year
        df["Month_num"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day

        # Choose features (all columns except Date and targets)
        feature_cols = [col for col in df.columns if col not in ["Date", "LPG_Price", "HP_Price"]]
        X = df[feature_cols].fillna(0)

        # Train two separate models
        models = {}
        for gas_col in ["LPG_Price", "HP_Price"]:
            y = df[gas_col].astype(float)
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X, y)
            models[gas_col] = (model, feature_cols)

        print(f"✅ Models trained for LPG and HP gas on {len(df)} records.")
        print(f"📅 Range: {df['Date'].min().date()} → {df['Date'].max().date()}")
        return models

    except Exception as e:
        print(f"❌ Error training models: {e}")
        raise

# Train at startup
models = train_models()


# -----------------------------
# 2. Routes
# -----------------------------
@app.route("/")
def home():
    # default template values; ensure index.html reads these variables
    return render_template("index.html", predic_text="--.--", predic_text_inr="--.--", predic_text_usd="--.--", gasType="Gas Type")


@app.route("/prediction", methods=["POST"])
def prediction():
    try:
        gas_type = request.form.get("gasType")
        year = int(request.form.get("year"))
        month = int(request.form.get("month"))
        day = int(request.form.get("day"))

        if gas_type not in ["HP Gas", "LPG Gas"]:
            return render_template("index.html", predic_text="Invalid Gas Type", predic_text_inr="--.--", predic_text_usd="--.--", gasType=gas_type)

        # Select model
        model_key = "HP_Price" if "HP" in gas_type else "LPG_Price"
        model, features = models[model_key]

        # Build feature input with safe defaults
        input_data = pd.DataFrame(columns=features)
        for col in features:
            col_lower = col.lower()
            if col_lower == "year":
                input_data.at[0, col] = year
            elif "month" in col_lower:
                input_data.at[0, col] = month
            elif "day" in col_lower:
                input_data.at[0, col] = day
            else:
                # Use column mean from training data if accessible, else 0
                # Try to obtain mean from model training data if possible (models may not store training X),
                # so as a simple fallback we set a neutral value (0). You can enhance this by storing training means.
                input_data.at[0, col] = 0.0

        # Fill any remaining NaNs with 0
        input_data = input_data.fillna(0.0)

        # Predict price (assumed to be in INR)
        predicted_price_inr = float(model.predict(input_data)[0])

        # Get live INR -> USD rate (USD per 1 INR)
        inr_to_usd = get_inr_to_usd_rate()

        # Convert to USD
        predicted_price_usd = predicted_price_inr * inr_to_usd

        # Formatting (2 decimal places)
        predicted_price_inr_str = f"₹{predicted_price_inr:,.2f}"
        predicted_price_usd_str = f"${predicted_price_usd:,.2f}"

        # Logging
        now = datetime.now().isoformat(sep=" ", timespec="seconds")
        print(f"{now} ✅ Predicted {gas_type} on {year}-{month:02d}-{day:02d}: INR {predicted_price_inr:.2f} | USD {predicted_price_usd:.2f} (rate={inr_to_usd:.6f})")

        return render_template(
            "index.html",
            predic_text=f"{predicted_price_usd_str}",      # USD display
            predic_text_inr=f"{predicted_price_inr_str}",  # INR display
            predic_text_usd=f"{predicted_price_usd_str}",  # optional duplicate field
            gasType=gas_type
        )

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return render_template("index.html", predic_text="--.--", predic_text_inr="--.--", predic_text_usd="--.--", gasType="Error")


# -----------------------------
# 3. Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
