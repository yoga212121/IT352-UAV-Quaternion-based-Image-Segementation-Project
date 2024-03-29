from flask import Flask, render_template, request, send_from_directory, url_for
import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
import warnings
import shutil 
from time import time

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# if os.path.exists(UPLOAD_FOLDER):
#     shutil.rmtree(UPLOAD_FOLDER) 
# os.makedirs(UPLOAD_FOLDER)  

RESULTS_FOLDER = 'results'
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# if os.path.exists(RESULTS_FOLDER):
#     shutil.rmtree(RESULTS_FOLDER)
# os.makedirs(RESULTS_FOLDER) 

def multiply_quaternions(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + w2 * x1 + y1 * z2 - z1 * y2
    y = w1 * y2 + w2 * y1 + z1 * x2 - z2 * x1
    z = w1 * z2 + w2 * z1 + x1 * y2 - x2 * y1

    return np.array([w, x, y, z])

def applyMask(rgb_img, scaleFactor):
    q = scaleFactor * np.array([0.7071, 0.4082, 0.4082, 0.4082])  # Quaternion for +pi/2 rotation
    qt = scaleFactor * np.array([0.7071, -0.4082, -0.4082, -0.4082])  # Quaternion for -pi/2 rotation

    rgb_img = rgb_img / 255.0

    w, h, _ = rgb_img.shape
    rot_img = np.zeros(rgb_img.shape)

    for i in range(w):
        for j in range(h):
            a = np.array([0, rgb_img[i][j][0], rgb_img[i][j][1], rgb_img[i][j][2]])
            rot_a = multiply_quaternions(multiply_quaternions(q, a), qt)
            rot_img[i][j][0] = rot_a[1]
            rot_img[i][j][1] = rot_a[2]
            rot_img[i][j][2] = rot_a[3]

    return rot_img

def Seg_PCA(img_seg, window_size):
    n_components = 1
    pca = PCA(n_components=n_components)

    data = img_seg.reshape((window_size*window_size, 3))

    transformed_data = pca.fit_transform(data)

    basis_vectors = pca.components_
    projected_data = np.dot(data, basis_vectors.T)

    return projected_data.reshape((window_size, window_size, 1))

def QPCA_Seg(rot_img, window_size):
    w, h, c = rot_img.shape 
    nw = w + ((window_size - w % window_size) if (w % window_size != 0) else 0)
    nh = h + ((window_size - h % window_size) if (h % window_size != 0) else 0)

    pad_img = np.zeros((nw, nh, c))
    seg_img = np.zeros((nw, nh, 1))

    pad_img[:w, :h, :] = rot_img

    for i in range(0, nw, window_size):
        for j in range(0, nh, window_size):
            segment = pad_img[i:i+window_size, j:j+window_size, :]
            PCA_segment = Seg_PCA(segment, window_size)
            seg_img[i:i+window_size, j:j+window_size,:] = PCA_segment

    return seg_img

def normaliseImage(img):
    min_val = np.min(img)
    max_val = np.max(img)
    norm_img = (img - min_val) / (max_val - min_val)
    return norm_img

@app.route('/')
def index():
        return render_template('index.html')

from flask import jsonify

from time import time

from time import time

@app.route('/process', methods=['POST'])
def process():
    if 'image' in request.files:
        image_file = request.files['image']
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(image_path)

        start_time = time()  

        img = cv2.imread(image_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x_value = int(request.form['x'])
        window_size = int(request.form['window_size'])
        SF = 1 / np.sqrt(x_value)  
        rotated_img = applyMask(rgb_img, SF)
        QPCA_img = QPCA_Seg(rotated_img, window_size)
        norm_img = normaliseImage(QPCA_img)

        end_time = time()  
        processing_time = end_time - start_time  

        if processing_time >= 60:
            processing_time = processing_time / 60
            time_unit = "minutes"
        else:
            time_unit = "seconds"

        qpca_img_path = os.path.join(app.config['RESULTS_FOLDER'], 'qpca_' + image_file.filename)
        cv2.imwrite(qpca_img_path, (norm_img * 255).astype(np.uint8))

        processed_image_url = url_for('result_file', filename='qpca_' + image_file.filename)
        uploaded_image_path = url_for('uploaded_file', filename=image_file.filename)
        
        output_html = f'''
            <h2>OUTPUT IMAGE</h2>
            <img src="{processed_image_url}" alt="Processed Image">
            <p>Selected scale: {x_value}, Window Size: {window_size}, Processing Time: {processing_time:.2f} {time_unit}</p>
            <button class="download-button">
                <a href="{processed_image_url}" download>Download Image</a>
            </button>
        '''

        return jsonify({'outputHtml': output_html})

    else:
        return jsonify({'error': 'No image uploaded.'})



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

warnings.filterwarnings("ignore", message="invalid value encountered in divide")

if __name__ == '__main__':
    app.run(debug=True)
