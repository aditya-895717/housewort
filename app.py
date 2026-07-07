from flask import Flask, request, render_template, jsonify
import os
import re
import io
import base64
import logging

import joblib
import pandas as pd
import requests as http_requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch

from dotenv import load_dotenv
from whitenoise import WhiteNoise

load_dotenv()

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/', prefix='static')

FALLBACK_IMAGE = "https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=800&h=600&fit=crop"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
APP_URL = os.environ.get("APP_URL", "")
APP_NAME = os.environ.get("APP_NAME", "SmartArchitect")

model_lr = None
scaler = None
try:
    model_lr = joblib.load('ML_Models/model.joblib')
    scaler = joblib.load('ML_Models/scaler.joblib')
except Exception as e:
    logging.error(f"Failed to load ML models: {e}")


def generate_floor_plan(input_data):
    fig = None
    try:
        total_area = max(int(input_data['area']), 500)
        bedrooms = max(1, min(int(input_data['bedrooms']), max(1, total_area // 200)))
        bathrooms = int(input_data['bathrooms'])
        stories = int(input_data['stories'])
        parking = int(input_data['parking'])
        has_basement = input_data['basement'] == 'yes'
        has_guestroom = input_data['guestroom'] == 'yes'

        aspect_ratio = 1.5
        total_width = max((total_area * aspect_ratio) ** 0.5, 10)
        total_length = max(total_area / total_width, 10)

        fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
        ax.set_xlim(0, total_width + 10)
        ax.set_ylim(0, total_length + 10)
        ax.set_aspect('equal')
        ax.axis('off')

        plt.title(
            f'Floor Plan - {input_data["area"]} sq ft ({input_data["bedrooms"]}BR/{bathrooms}BA)',
            fontsize=18, fontweight='bold', pad=20
        )

        outer_wall = FancyBboxPatch(
            (5, 5), total_width, total_length,
            boxstyle="round,pad=0.1",
            edgecolor='black', facecolor='lightgray',
            linewidth=3, alpha=0.3
        )
        ax.add_patch(outer_wall)

        ax.annotate('', xy=(5, total_length + 6), xytext=(total_width + 5, total_length + 6),
                    arrowprops=dict(arrowstyle='<->', lw=1.5, color='red'))
        ax.text(total_width / 2 + 5, total_length + 7, f'{total_width:.1f} ft',
                ha='center', fontsize=10, color='red', fontweight='bold')

        ax.annotate('', xy=(total_width + 6, 5), xytext=(total_width + 6, total_length + 5),
                    arrowprops=dict(arrowstyle='<->', lw=1.5, color='red'))
        ax.text(total_width + 7.5, total_length / 2 + 5, f'{total_length:.1f} ft',
                rotation=90, va='center', fontsize=10, color='red', fontweight='bold')

        current_y = 5
        current_x = 5
        rooms = []

        living_width = max(total_width * 0.6, 1)
        living_length = max((total_area * 0.3) / living_width, 1)
        rooms.append({'name': 'Living & Dining', 'x': current_x, 'y': current_y,
                      'width': living_width, 'height': living_length,
                      'color': '#e8f4f8', 'icon': '🛋️'})

        kitchen_width = max(total_width * 0.4, 1)
        kitchen_length = max((total_area * 0.15) / kitchen_width, 1)
        rooms.append({'name': 'Kitchen', 'x': current_x + living_width, 'y': current_y,
                      'width': kitchen_width, 'height': kitchen_length,
                      'color': '#fff4e6', 'icon': '🍳'})

        current_y += living_length

        master_width = max(total_width * 0.5, 1)
        master_length = max((total_area * 0.2) / master_width, 1)
        rooms.append({'name': 'Master Bedroom', 'x': current_x, 'y': current_y,
                      'width': master_width, 'height': master_length,
                      'color': '#f0e6ff', 'icon': '🛏️'})

        master_bath_width = max(total_width * 0.25, 1)
        master_bath_length = max((total_area * 0.05) / master_bath_width, 1)
        rooms.append({'name': 'Master Bath', 'x': current_x + master_width, 'y': current_y,
                      'width': master_bath_width, 'height': master_bath_length,
                      'color': '#e6f7ff', 'icon': '🚿'})

        if bedrooms > 1:
            extra_beds = bedrooms - 1
            bedroom_area = max((total_area * 0.15) / extra_beds, 25)
            bedroom_width = max(total_width * 0.25, 1)
            bedroom_length = max(bedroom_area / bedroom_width, 1)
            x_offset = current_x + master_width + master_bath_width
            for i in range(extra_beds):
                rooms.append({
                    'name': f'Bedroom {i + 2}',
                    'x': x_offset,
                    'y': current_y + (i * bedroom_length),
                    'width': bedroom_width,
                    'height': bedroom_length,
                    'color': '#ffe6f0',
                    'icon': '🛏️'
                })

        current_y += master_length

        if bathrooms > 1:
            extra_baths = bathrooms - 1
            bath_width = max(total_width / extra_baths, 1)
            bath_length = max((total_area * 0.05) / bath_width, 1)
            for i in range(extra_baths):
                rooms.append({
                    'name': f'Bathroom {i + 2}',
                    'x': current_x + (i * bath_width),
                    'y': current_y,
                    'width': bath_width,
                    'height': bath_length,
                    'color': '#e6f7ff',
                    'icon': '🚽'
                })
            current_y += bath_length

        if has_guestroom:
            guest_width = max(total_width * 0.4, 1)
            guest_length = max((total_area * 0.1) / guest_width, 1)
            rooms.append({'name': 'Guest Room', 'x': current_x, 'y': current_y,
                          'width': guest_width, 'height': guest_length,
                          'color': '#fff0f5', 'icon': '🛌'})

        if parking > 0:
            parking_width = max(total_width * 0.6, 1)
            rooms.append({
                'name': f'Garage ({parking} cars)',
                'x': current_x + total_width * 0.4 if has_guestroom else current_x,
                'y': current_y,
                'width': parking_width,
                'height': 15,
                'color': '#f5f5f5',
                'icon': '🚗'
            })

        for room in rooms:
            rect = Rectangle(
                (room['x'], room['y']), room['width'], room['height'],
                edgecolor='black', facecolor=room['color'], linewidth=2, alpha=0.7
            )
            ax.add_patch(rect)
            cx = room['x'] + room['width'] / 2
            cy = room['y'] + room['height'] / 2
            ax.text(cx, cy + room['height'] / 4, room['icon'],
                    ha='center', va='center', fontsize=24)
            ax.text(cx, cy - room['height'] / 6, room['name'],
                    ha='center', va='center', fontsize=9, fontweight='bold')
            ax.text(cx, cy - room['height'] / 3,
                    f"{room['width']:.1f}' x {room['height']:.1f}'",
                    ha='center', va='center', fontsize=8, color='blue', style='italic')
            ax.text(cx, room['y'] + 1,
                    f"({room['width'] * room['height']:.0f} sq ft)",
                    ha='center', va='bottom', fontsize=7, color='gray')

        for door_x, door_y in [(5, total_length / 2 + 5), (total_width * 0.6 + 5, 5.5)]:
            ax.add_patch(patches.Arc((door_x, door_y), 3, 3, angle=0,
                                     theta1=0, theta2=90, color='brown', linewidth=2))

        legend_y = 2
        ax.text(2, legend_y, 'Floor Plan Legend', fontsize=12, fontweight='bold')
        ax.text(2, legend_y - 0.8, '- All dimensions in feet', fontsize=9)
        ax.text(2, legend_y - 1.5, '- Walls shown in black (6" thick)', fontsize=9)
        ax.text(2, legend_y - 2.2, f'- Total Area: {input_data["area"]} sq ft',
                fontsize=9, fontweight='bold')
        ax.text(2, legend_y - 2.9, f'- Stories: {stories}', fontsize=9)

        compass_x, compass_y = total_width + 3, 2
        ax.arrow(compass_x, compass_y, 0, 2, head_width=0.5, head_length=0.3,
                 fc='black', ec='black')
        ax.text(compass_x, compass_y + 2.5, 'N', ha='center', fontsize=12, fontweight='bold')

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    except Exception as e:
        logging.error(f"Floor plan generation error: {e}")
        return None
    finally:
        if fig is not None:
            plt.close(fig)


def get_house_image_url(house_description):
    if not OPENROUTER_API_KEY:
        return FALLBACK_IMAGE
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        if APP_URL:
            headers["HTTP-Referer"] = APP_URL
        if APP_NAME:
            headers["X-Title"] = APP_NAME

        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Find and provide a direct URL to a high-quality image of a modern residential "
                        f"house with these specifications: {house_description}. Provide only the image URL "
                        f"from a reliable architecture or real estate photography source."
                    )
                }
            ]
        }

        response = http_requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            urls = re.findall(r'https?://(?:[a-zA-Z0-9$\-_.&+!*\'(),]|%[0-9a-fA-F]{2})+', content)
            if urls:
                return urls[0]
    except Exception as e:
        logging.error(f"OpenRouter API error: {e}")

    return FALLBACK_IMAGE


def generate_house_description(input_data):
    parts = [
        f"{input_data['stories']}-story house",
        f"with {input_data['bedrooms']} bedrooms",
        f"and {input_data['bathrooms']} bathrooms",
        f"approximately {input_data['area']} square feet",
    ]
    if input_data['basement'] == 'yes':
        parts.append("with basement")
    if input_data['guestroom'] == 'yes':
        parts.append("and guest room")
    if input_data['airconditioning'] == 'yes':
        parts.append("air conditioning")
    if input_data['parking'] > 0:
        parts.append(f"{input_data['parking']}-car garage")
    if input_data['mainroad'] == 'yes':
        parts.append("located on main road")
    if input_data['prefarea'] == 'yes':
        parts.append("in preferred area")
    parts.append(f"{input_data.get('furnishing_status', 'unfurnished')} interiors")
    return ", ".join(parts)


def predict_price(input_data):
    expected_columns = [
        'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
        'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
        'parking', 'prefarea', 'furnishingstatus_semi-furnished',
        'furnishingstatus_unfurnished'
    ]
    model_data = {col: input_data[col] for col in expected_columns if col in input_data}
    data = pd.DataFrame(model_data, index=[0])[expected_columns]

    yes_no_cols = [
        'mainroad', 'guestroom', 'basement', 'hotwaterheating',
        'airconditioning', 'prefarea',
        'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished'
    ]
    non_bin_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

    mapping = {'yes': 1, 'no': 0}
    for col in yes_no_cols:
        if col in data.columns:
            data[col] = data[col].map(mapping)

    data[non_bin_vars] = scaler.transform(data[non_bin_vars])
    return model_lr.predict(data)[0]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ping')
def ping():
    return jsonify({"status": "ok"}), 200


@app.route('/predict', methods=['POST'])
def predict():
    try:
        furnishing_status = request.form.get('furnishing_status', '').strip()
        if not furnishing_status:
            return render_template('index.html',
                                   error="Please select a furnishing status.")

        try:
            area = int(request.form['area'])
            bedrooms = int(request.form['bedrooms'])
            bathrooms = int(request.form['bathrooms'])
            stories = int(request.form['stories'])
            parking = int(request.form['parking'])
        except (KeyError, ValueError, TypeError):
            return render_template(
                'index.html',
                error="Invalid or missing numeric input. Please check area, bedrooms, bathrooms, stories, and parking."
            )

        if model_lr is None or scaler is None:
            return render_template('index.html',
                                   error="ML models are unavailable. Please contact the administrator.")

        furnishing_semi = 'yes' if furnishing_status == 'semi-furnished' else 'no'
        furnishing_unfurnished = 'yes' if furnishing_status == 'unfurnished' else 'no'

        input_data = {
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'mainroad': request.form.get('mainroad', 'no'),
            'guestroom': request.form.get('guestroom', 'no'),
            'basement': request.form.get('basement', 'no'),
            'hotwaterheating': request.form.get('hotwaterheating', 'no'),
            'airconditioning': request.form.get('airconditioning', 'no'),
            'parking': parking,
            'prefarea': request.form.get('prefarea', 'no'),
            'furnishingstatus_semi-furnished': furnishing_semi,
            'furnishingstatus_unfurnished': furnishing_unfurnished
        }

        prediction = round(predict_price(input_data), 2)

        display_data = input_data.copy()
        display_data['furnishing_status'] = furnishing_status

        floor_plan_image = generate_floor_plan(display_data)
        house_description = generate_house_description(display_data)
        house_image = get_house_image_url(house_description)

        return render_template(
            'index.html',
            prediction=prediction,
            house_image=house_image,
            floor_plan_image=floor_plan_image,
            house_description=house_description,
            input_data=display_data
        )

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return render_template('index.html',
                               error="An unexpected error occurred. Please try again.")


if __name__ == '__main__':
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
