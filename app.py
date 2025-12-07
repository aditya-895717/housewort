from flask import Flask, request, render_template, send_file
import joblib
import pandas as pd
import requests
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import io
import base64
import os

app = Flask(__name__)

# Loading the trained models
model_lr = joblib.load('ML_Models/model.joblib')
scaler = joblib.load('ML_Models/scaler.joblib')

PERPLEXITY_API_KEY = "pplx-2I1gnppYMuoksHNQZy06To9ZaAnmDMi7MNxqpY8DsrCAk"
PERPLEXITY_API_ENDPOINT = "https://api.perplexity.ai/chat/completions"


def generate_floor_plan(input_data):
    """Generate a 2D floor plan sketch with dimensions"""
    
    # Calculate dimensions based on area and rooms
    total_area = input_data['area']
    bedrooms = input_data['bedrooms']
    bathrooms = input_data['bathrooms']
    stories = input_data['stories']
    parking = input_data['parking']
    has_basement = input_data['basement'] == 'yes'
    has_guestroom = input_data['guestroom'] == 'yes'
    
    # Calculate approximate dimensions (assuming rectangular layout)
    aspect_ratio = 1.5  # width to length ratio
    total_width = (total_area * aspect_ratio) ** 0.5
    total_length = total_area / total_width
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
    ax.set_xlim(0, total_width + 10)
    ax.set_ylim(0, total_length + 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    plt.title(f'Floor Plan - {total_area} sq ft ({bedrooms}BR/{bathrooms}BA)', 
              fontsize=18, fontweight='bold', pad=20)
    
    # Draw outer walls
    outer_wall = FancyBboxPatch((5, 5), total_width, total_length, 
                                boxstyle="round,pad=0.1", 
                                edgecolor='black', facecolor='lightgray', 
                                linewidth=3, alpha=0.3)
    ax.add_patch(outer_wall)
    
    # Add dimension annotations for outer walls
    # Width dimension (top)
    ax.annotate('', xy=(5, total_length + 6), xytext=(total_width + 5, total_length + 6),
                arrowprops=dict(arrowstyle='<->', lw=1.5, color='red'))
    ax.text(total_width/2 + 5, total_length + 7, f'{total_width:.1f} ft', 
            ha='center', fontsize=10, color='red', fontweight='bold')
    
    # Length dimension (right)
    ax.annotate('', xy=(total_width + 6, 5), xytext=(total_width + 6, total_length + 5),
                arrowprops=dict(arrowstyle='<->', lw=1.5, color='red'))
    ax.text(total_width + 7.5, total_length/2 + 5, f'{total_length:.1f} ft', 
            rotation=90, va='center', fontsize=10, color='red', fontweight='bold')
    
    # Calculate room dimensions
    current_y = 5
    current_x = 5
    
    rooms = []
    
    # Living/Dining Area (30% of total area)
    living_width = total_width * 0.6
    living_length = (total_area * 0.3) / living_width
    rooms.append({
        'name': 'Living & Dining',
        'x': current_x,
        'y': current_y,
        'width': living_width,
        'height': living_length,
        'color': '#e8f4f8',
        'icon': '🛋️'
    })
    
    # Kitchen (15% of total area)
    kitchen_width = total_width * 0.4
    kitchen_length = (total_area * 0.15) / kitchen_width
    rooms.append({
        'name': 'Kitchen',
        'x': current_x + living_width,
        'y': current_y,
        'width': kitchen_width,
        'height': kitchen_length,
        'color': '#fff4e6',
        'icon': '🍳'
    })
    
    current_y += living_length
    
    # Master Bedroom (20% of total area)
    master_width = total_width * 0.5
    master_length = (total_area * 0.2) / master_width
    rooms.append({
        'name': 'Master Bedroom',
        'x': current_x,
        'y': current_y,
        'width': master_width,
        'height': master_length,
        'color': '#f0e6ff',
        'icon': '🛏️'
    })
    
    # Master Bathroom (5% of total area)
    master_bath_width = total_width * 0.25
    master_bath_length = (total_area * 0.05) / master_bath_width
    rooms.append({
        'name': 'Master Bath',
        'x': current_x + master_width,
        'y': current_y,
        'width': master_bath_width,
        'height': master_bath_length,
        'color': '#e6f7ff',
        'icon': '🚿'
    })
    
    # Additional bedrooms
    if bedrooms > 1:
        bedroom_area = (total_area * 0.15) / (bedrooms - 1)
        bedroom_width = total_width * 0.25
        bedroom_length = bedroom_area / bedroom_width
        
        x_offset = current_x + master_width + master_bath_width
        for i in range(bedrooms - 1):
            rooms.append({
                'name': f'Bedroom {i+2}',
                'x': x_offset,
                'y': current_y + (i * bedroom_length),
                'width': bedroom_width,
                'height': bedroom_length,
                'color': '#ffe6f0',
                'icon': '🛏️'
            })
    
    current_y += master_length
    
    # Additional bathrooms
    if bathrooms > 1:
        bath_width = total_width / (bathrooms - 1)
        bath_length = (total_area * 0.05) / bath_width
        
        for i in range(bathrooms - 1):
            rooms.append({
                'name': f'Bathroom {i+2}',
                'x': current_x + (i * bath_width),
                'y': current_y,
                'width': bath_width,
                'height': bath_length,
                'color': '#e6f7ff',
                'icon': '🚽'
            })
        current_y += bath_length
    
    # Guest room if available
    if has_guestroom:
        guest_width = total_width * 0.4
        guest_length = (total_area * 0.1) / guest_width
        rooms.append({
            'name': 'Guest Room',
            'x': current_x,
            'y': current_y,
            'width': guest_width,
            'height': guest_length,
            'color': '#fff0f5',
            'icon': '🛌'
        })
    
    # Parking area (if available)
    if parking > 0:
        parking_width = total_width * 0.6
        parking_length = 15  # Fixed parking length
        rooms.append({
            'name': f'Garage ({parking} cars)',
            'x': current_x + total_width * 0.4 if has_guestroom else current_x,
            'y': current_y,
            'width': parking_width,
            'height': parking_length,
            'color': '#f5f5f5',
            'icon': '🚗'
        })
    
    # Draw all rooms
    for room in rooms:
        # Room rectangle
        rect = Rectangle((room['x'], room['y']), room['width'], room['height'],
                        edgecolor='black', facecolor=room['color'], 
                        linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        
        # Room label with icon
        center_x = room['x'] + room['width'] / 2
        center_y = room['y'] + room['height'] / 2
        ax.text(center_x, center_y + room['height']/4, room['icon'], 
                ha='center', va='center', fontsize=24)
        ax.text(center_x, center_y - room['height']/6, room['name'], 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Dimensions
        dim_text = f"{room['width']:.1f}' × {room['height']:.1f}'"
        ax.text(center_x, center_y - room['height']/3, dim_text, 
                ha='center', va='center', fontsize=8, 
                color='blue', style='italic')
        
        # Area
        area_text = f"({room['width'] * room['height']:.0f} sq ft)"
        ax.text(center_x, room['y'] + 1, area_text, 
                ha='center', va='bottom', fontsize=7, color='gray')
    
    # Add door symbols
    door_positions = [
        (5, total_length/2 + 5),  # Main entrance
        (total_width * 0.6 + 5, 5.5),  # Kitchen door
    ]
    
    for door_x, door_y in door_positions:
        door = patches.Arc((door_x, door_y), 3, 3, angle=0, 
                          theta1=0, theta2=90, color='brown', linewidth=2)
        ax.add_patch(door)
    
    # Add legend
    legend_y = 2
    ax.text(2, legend_y, '📐 Floor Plan Legend', fontsize=12, fontweight='bold')
    ax.text(2, legend_y - 0.8, '• All dimensions in feet', fontsize=9)
    ax.text(2, legend_y - 1.5, '• Walls shown in black (6" thick)', fontsize=9)
    ax.text(2, legend_y - 2.2, f'• Total Area: {total_area} sq ft', fontsize=9, fontweight='bold')
    ax.text(2, legend_y - 2.9, f'• Stories: {stories}', fontsize=9)
    
    # Add compass
    compass_x, compass_y = total_width + 3, 2
    ax.arrow(compass_x, compass_y, 0, 2, head_width=0.5, head_length=0.3, 
             fc='black', ec='black')
    ax.text(compass_x, compass_y + 2.5, 'N', ha='center', fontsize=12, fontweight='bold')
    
    # Save to bytes buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    
    # Encode to base64
    floor_plan_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return floor_plan_base64


def get_house_image_url(house_description):
    """Get a house architecture image URL using Perplexity API"""
    try:
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "user",
                    "content": f"Find and provide a direct URL to a high-quality image of a modern residential house with these specifications: {house_description}. Provide only the image URL from a reliable architecture or real estate photography source."
                }
            ]
        }
        
        response = requests.post(PERPLEXITY_API_ENDPOINT, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            import re
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
            
            if urls:
                return urls[0]
    except Exception as e:
        print(f"Error fetching image: {str(e)}")
    
    return "https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=800&h=600&fit=crop"


def generate_house_description(input_data):
    """Generate a detailed house description"""
    description_parts = []
    description_parts.append(f"{input_data['stories']}-story house")
    description_parts.append(f"with {input_data['bedrooms']} bedrooms")
    description_parts.append(f"and {input_data['bathrooms']} bathrooms")
    description_parts.append(f"approximately {input_data['area']} square feet")
    
    if input_data['basement'] == 'yes':
        description_parts.append("with basement")
    if input_data['guestroom'] == 'yes':
        description_parts.append("and guest room")
    if input_data['airconditioning'] == 'yes':
        description_parts.append("air conditioning")
    if input_data['parking'] > 0:
        description_parts.append(f"{input_data['parking']}-car garage")
    if input_data['mainroad'] == 'yes':
        description_parts.append("located on main road")
    if input_data['prefarea'] == 'yes':
        description_parts.append("in preferred area")
    
    furnishing = input_data.get('furnishing_status', 'unfurnished')
    description_parts.append(f"{furnishing} interiors")
    
    return ", ".join(description_parts)


def predict_price(input_data):
    """Function to preprocess input data and make predictions"""
    expected_columns = [
        'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 
        'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 
        'parking', 'prefarea', 'furnishingstatus_semi-furnished', 
        'furnishingstatus_unfurnished'
    ]
    
    model_data = {col: input_data[col] for col in expected_columns if col in input_data}
    data = pd.DataFrame(model_data, index=[0])
    data = data[expected_columns]
    
    yes_no_attributes = [
        'mainroad', 'guestroom', 'basement', 'hotwaterheating', 
        'airconditioning', 'prefarea', 'furnishingstatus_semi-furnished', 
        'furnishingstatus_unfurnished'
    ]
    non_bin_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    
    mapping = {'yes': 1, 'no': 0}
    for col in yes_no_attributes:
        if col in data.columns:
            data[col] = data[col].map(mapping)
    
    data[non_bin_vars] = scaler.transform(data[non_bin_vars])
    prediction_lr = model_lr.predict(data)
    
    return prediction_lr[0]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    furnishing_status = request.form['furnishing_status']

    if furnishing_status == 'furnished':
        furnishing_semi_furnished = 'no'
        furnishing_unfurnished = 'no'
    elif furnishing_status == 'semi-furnished':
        furnishing_semi_furnished = 'yes'
        furnishing_unfurnished = 'no'
    else:
        furnishing_semi_furnished = 'no'
        furnishing_unfurnished = 'yes'

    input_data = {
        'area': int(request.form['area']),
        'bedrooms': int(request.form['bedrooms']),
        'bathrooms': int(request.form['bathrooms']),
        'stories': int(request.form['stories']),
        'mainroad': request.form['mainroad'],
        'guestroom': request.form['guestroom'],
        'basement': request.form['basement'],
        'hotwaterheating': request.form['hotwaterheating'],
        'airconditioning': request.form['airconditioning'],
        'parking': int(request.form['parking']),
        'prefarea': request.form['prefarea'],
        'furnishingstatus_semi-furnished': furnishing_semi_furnished,
        'furnishingstatus_unfurnished': furnishing_unfurnished
    }

    prediction = round(predict_price(input_data), 2)
    
    display_data = input_data.copy()
    display_data['furnishing_status'] = furnishing_status
    
    # Generate floor plan
    floor_plan_image = generate_floor_plan(display_data)
    
    # Generate house description and get exterior image
    house_description = generate_house_description(display_data)
    house_image = get_house_image_url(house_description)

    return render_template('index.html', 
                         prediction=prediction, 
                         house_image=house_image,
                         floor_plan_image=floor_plan_image,
                         house_description=house_description,
                         input_data=display_data)


if __name__ == '__main__':
    app.run(debug=True)
