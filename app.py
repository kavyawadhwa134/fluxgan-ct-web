from flask import Flask, render_template, request, jsonify
from fluxgan_core import load_checkpoint, predict_flux_burnup
import os

app = Flask(__name__)

# Load model only once
generator, checkpoint_info = load_checkpoint('./fluxgan_model/checkpoint_1000.tar')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        enrichment = float(request.form['enrichment'])
        flux, burnup = predict_flux_burnup(generator, checkpoint_info, enrichment)

        # Avoid negative outputs (optional safety)
        flux = max(flux, 0)
        burnup = max(burnup, 0)

        return jsonify({
            'enrichment': enrichment,
            'flux': f"{flux:.4e} n/cm²-s",
            'burnup': f"{burnup:.6g} MWd/kgU"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Render requirement: bind to 0.0.0.0 and dynamic port
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 500))  # PORT env var is provided by Render
    app.run(host='0.0.0.0', port=port, debug=True)
