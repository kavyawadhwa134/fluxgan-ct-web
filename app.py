from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
from fluxgan_core import load_checkpoint, predict_flux_burnup  # This is your model logic

app = Flask(__name__)

# Load model + checkpoint info once
generator, checkpoint_info = load_checkpoint('./fluxgan_model/checkpoint_1000.tar')
generator.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get enrichment input from frontend
        data = request.get_json()
        enrichment = float(data['enrichment'])

        # Run the model
        flux, burnup = predict_flux_burnup(generator, checkpoint_info, enrichment)

        return jsonify({
            'enrichment': enrichment,
            'flux': f"{flux:.4e} n/cm²-s",
            'burnup': f"{burnup:.20f} MWd/kgU"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
