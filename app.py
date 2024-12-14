from flask import Flask, request, render_template
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import io

app = Flask(__name__)

# Function to calculate dominant colors
def get_dominant_colors(image, clusters=3):
    # Convert the image to RGB and resize for efficiency
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (100, 100))  # Reduce size for faster processing

    # Flatten the image
    image_flatten = image.reshape(-1, 3)

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    kmeans.fit(image_flatten)

    # Get cluster centers as dominant colors
    dominant_colors = kmeans.cluster_centers_.astype(int)
    return dominant_colors

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part"

        file = request.files['image']
        if file.filename == '':
            return "No selected file"

        if file:
            image = Image.open(io.BytesIO(file.read()))
            dominant_colors = get_dominant_colors(image)

            # Prepare color blocks to display
            color_blocks = [
                f'rgb({color[0]}, {color[1]}, {color[2]})' for color in dominant_colors
            ]
            return render_template('result.html', colors=color_blocks)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
