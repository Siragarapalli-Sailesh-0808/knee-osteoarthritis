
#!/usr/bin/env python
from flask import Flask, render_template, request
import numpy as np
import onnxruntime as ort
import os
import base64
import uuid
import tempfile
import traceback
from PIL import Image
from werkzeug.utils import secure_filename




app = Flask(__name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
CONFIDENCE_THRESHOLD = 0.55
MAX_REFERENCE_IMAGES = 200
XRAY_DISTANCE_TOLERANCE = 1.15
STRICT_XRAY_THRESHOLD_MULTIPLIER = 1.5  # Stricter threshold for non-knee X-rays detection

# Load your trained model
 
verbose_name = {
0: "Normal",
1: "Doubtful",
2: "Mild",
3: "Moderate",
4: "Severe" 




}

 

session = ort.InferenceSession("knee.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


def allowed_file(filename):
	return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_xray_features(img_path):
	with Image.open(img_path) as pil_img:
		gray = pil_img.convert("L").resize((224, 224))
		arr = np.array(gray, dtype=np.float32) / 255.0

	intensity_hist, _ = np.histogram(arr, bins=32, range=(0.0, 1.0), density=True)

	gx = np.diff(arr, axis=1)
	gy = np.diff(arr, axis=0)
	gx = gx[:-1, :]
	gy = gy[:, :-1]
	grad_mag = np.sqrt(gx * gx + gy * gy)
	grad_mag = np.clip(grad_mag, 0.0, 1.0)
	grad_hist, _ = np.histogram(grad_mag, bins=24, range=(0.0, 1.0), density=True)

	dark_ratio = float(np.mean(arr < 0.12))
	bright_ratio = float(np.mean(arr > 0.88))

	return np.concatenate([
		intensity_hist,
		grad_hist,
		np.array([arr.mean(), arr.std(), dark_ratio, bright_ratio], dtype=np.float32)
	]).astype(np.float32)


def build_xray_reference():
	candidate_roots = [os.path.join("Test_samples"), os.path.join("model", "train")]
	features = []
	for root in candidate_roots:
		if not os.path.isdir(root):
			continue
		for dirpath, _, filenames in os.walk(root):
			for filename in sorted(filenames):
				if not allowed_file(filename):
					continue

				img_path = os.path.join(dirpath, filename)
				try:
					features.append(extract_xray_features(img_path))
				except Exception:
					continue

				if len(features) >= MAX_REFERENCE_IMAGES:
					break

			if len(features) >= MAX_REFERENCE_IMAGES:
				break

		if len(features) >= MAX_REFERENCE_IMAGES:
			break

	if len(features) < 10:
		return None

	feat_matrix = np.stack(features)
	mean_vec = np.mean(feat_matrix, axis=0)
	std_vec = np.std(feat_matrix, axis=0) + 1e-6
	z = (feat_matrix - mean_vec) / std_vec
	dists = np.sqrt(np.mean(z * z, axis=1))
	
	# Use 95th percentile instead of 99th for stricter threshold (reject outliers more aggressively)
	# This prevents hand/other X-rays from passing as knee X-rays
	strict_threshold = float(np.percentile(dists, 95) * 1.2)  # Reduced multiplier from 1.15 to 1.2 base
	
	# Additional safeguard: ensure threshold doesn't exceed 1.5 (knee X-rays typically < 1.2)
	strict_threshold = min(strict_threshold, 1.5)

	return {
		"mean": mean_vec,
		"std": std_vec,
		"threshold": strict_threshold
	}


XRAY_REFERENCE = build_xray_reference()


def looks_like_xray(img_path):
	try:
		with Image.open(img_path) as pil_img:
			rgb = pil_img.convert("RGB").resize((224, 224))
			rgb_arr = np.array(rgb, dtype=np.float32) / 255.0

		hsv_arr = np.array(rgb.convert("HSV"), dtype=np.float32) / 255.0
		mean_sat = float(np.mean(hsv_arr[:, :, 1]))
		channel_gap = float(np.mean(np.abs(rgb_arr[:, :, 0] - rgb_arr[:, :, 1]) + np.abs(rgb_arr[:, :, 1] - rgb_arr[:, :, 2]) + np.abs(rgb_arr[:, :, 0] - rgb_arr[:, :, 2])) / 3.0)

		# STRICT: Reject images with color - must be grayscale for X-rays
		if mean_sat > 0.25 or channel_gap > 0.15:
			return False

		# STRICT: Additional grayscale check - compute gray image stats
		with Image.open(img_path) as pil_img:
			gray = pil_img.convert("L")
			gray_arr = np.array(gray, dtype=np.float32) / 255.0
		
		# X-rays typically have wide distribution of pixel values
		# Reject if too dark (mean < 0.15) or too light (mean > 0.85)
		gray_mean = float(np.mean(gray_arr))
		gray_std = float(np.std(gray_arr))
		
		if gray_mean < 0.1 or gray_mean > 0.9:
			return False  # Image is too extremely dark or bright
		
		if gray_std < 0.08:
			return False  # No variation - likely not an X-ray
		
		# STRICT: Must match reference knee X-rays from training data
		if XRAY_REFERENCE is not None:
			feat = extract_xray_features(img_path)
			z = (feat - XRAY_REFERENCE["mean"]) / XRAY_REFERENCE["std"]
			dist = float(np.sqrt(np.mean(z * z)))
			# Only accept if matches reference with strict threshold
			if dist <= XRAY_REFERENCE["threshold"]:
				return True
			# Reject if doesn't match reference - no fallback
			return False

		# If no reference available, require strict grayscale characteristics
		return (mean_sat <= 0.25 and channel_gap <= 0.15 and gray_std >= 0.08)
	except Exception:
		return False


def predict_label(img_path):
	with Image.open(img_path) as pil_img:
		rgb = pil_img.convert("RGB").resize((224, 224))
		test_image = np.array(rgb, dtype=np.float32) / 255.0

	test_image = np.expand_dims(test_image, axis=0)
	predict_x = session.run(None, {input_name: test_image})[0]
	classes_x=np.argmax(predict_x,axis=1)
	confidence = float(np.max(predict_x))
	
	return verbose_name [classes_x[0]], confidence


def generate_ai_summary(prediction, confidence):
	guidance_map = {
		"Normal": {
			"severity": "Low",
			"causes": [
				"No clear osteoarthritic joint-space narrowing in this X-ray.",
				"Mild wear can still develop over time with age and repetitive load."
			],
			"precautions": [
				"Maintain healthy body weight to reduce knee load.",
				"Continue low-impact exercise: walking, cycling, swimming.",
				"Warm up before sports and avoid sudden high-impact stress."
			],
			"next_steps": [
				"Follow knee-strengthening routine 3-4 times weekly.",
				"Repeat screening if persistent knee pain appears."
			]
		},
		"Doubtful": {
			"severity": "Mild Risk",
			"causes": [
				"Early cartilage wear may be starting.",
				"Joint loading from excess weight or repetitive standing can contribute."
			],
			"precautions": [
				"Avoid prolonged stair climbing and deep squats.",
				"Use supportive footwear and proper posture.",
				"Start quadriceps and hamstring strengthening exercises."
			],
			"next_steps": [
				"Track pain, stiffness, and swelling for 2-4 weeks.",
				"Consult orthopedics if symptoms increase."
			]
		},
		"Mild": {
			"severity": "Moderate Risk",
			"causes": [
				"Mild osteoarthritic changes are present.",
				"Common contributors: age-related degeneration, prior knee injury, obesity."
			],
			"precautions": [
				"Prefer low-impact activities; avoid jumping/running on hard surfaces.",
				"Reduce body weight if overweight.",
				"Use supervised physiotherapy for muscle balance around knee."
			],
			"next_steps": [
				"Clinical review if pain lasts more than 2 weeks.",
				"Consider vitamin D/calcium assessment as advised by doctor."
			]
		},
		"Moderate": {
			"severity": "High Risk",
			"causes": [
				"Clear osteoarthritic progression with increased joint degeneration.",
				"Inflammation and mechanical overload likely contribute to symptoms."
			],
			"precautions": [
				"Avoid high-impact activity and heavy lifting.",
				"Use knee support when advised by clinician.",
				"Follow structured physiotherapy and pain-management plan."
			],
			"next_steps": [
				"Book orthopedic consultation for treatment planning.",
				"Discuss medication, injections, and rehabilitation options."
			]
		},
		"Severe": {
			"severity": "Very High Risk",
			"causes": [
				"Advanced osteoarthritic damage is likely present.",
				"Marked cartilage loss and chronic inflammation may explain severe pain/stiffness."
			],
			"precautions": [
				"Limit painful weight-bearing activity immediately.",
				"Use assistive support if needed (cane/walker per medical advice).",
				"Do not ignore swelling, locking, or mobility loss."
			],
			"next_steps": [
				"Urgent orthopedic evaluation is recommended.",
				"Discuss definitive options including advanced interventions."
			]
		}
	}

	default_summary = {
		"severity": "Unknown",
		"causes": ["Prediction class unavailable."],
		"precautions": ["Please upload a valid knee X-ray and retry."],
		"next_steps": ["Consult a specialist for clinical evaluation."]
	}

	base = guidance_map.get(prediction, default_summary)

	if confidence >= 0.9:
		confidence_note = "High confidence prediction."
	elif confidence >= CONFIDENCE_THRESHOLD:
		confidence_note = "Moderate confidence prediction."
	else:
		confidence_note = "Low confidence prediction. Clinical confirmation is important."

	red_flags = [
		"Persistent night pain or rapidly increasing pain.",
		"Knee swelling with fever or redness.",
		"Sudden inability to bear weight or severe locking."
	]

	return {
		"prediction": prediction,
		"confidence_percent": round(confidence * 100, 2),
		"confidence_note": confidence_note,
		"severity": base["severity"],
		"causes": base["causes"],
		"precautions": base["precautions"],
		"next_steps": base["next_steps"],
		"red_flags": red_flags,
		"disclaimer": "AI output is supportive information only, not a final medical diagnosis. Please consult an orthopedic specialist."
	}
 

# Load your trained model
 


@app.route("/")
@app.route("/first")
def first():
	return render_template('first.html')
    
@app.route("/login")
def login():
	return render_template('login.html')    
@app.route("/chart")
def chart():
	return render_template('chart.html')

@app.route("/performance")
def performance():
	return render_template('performance.html')


@app.route("/index", methods=['GET', 'POST'])
def index():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	predict_result = None
	confidence = None
	ai_summary = None
	error_message = None
	img_path = None

	if request.method == 'POST':
		try:
			img = request.files.get('my_image')

			if img is None or img.filename == "":
				error_message = "Please choose an image to upload."
			elif not allowed_file(img.filename):
				error_message = "Only PNG/JPG/JPEG/BMP files are supported."
			else:
				filename = secure_filename(img.filename)
				upload_dir = os.path.join(tempfile.gettempdir(), "uploads")
				os.makedirs(upload_dir, exist_ok=True)
				temp_filename = f"{uuid.uuid4().hex}_{filename}"
				temp_img_path = os.path.join(upload_dir, temp_filename)

				try:
					img.save(temp_img_path)

					with open(temp_img_path, "rb") as f:
						encoded = base64.b64encode(f.read()).decode("ascii")
					mime = img.mimetype if img.mimetype else "image/jpeg"
					img_path = f"data:{mime};base64,{encoded}"

					if not looks_like_xray(temp_img_path):
						error_message = "Please upload a valid knee X-ray image."
					else:
						predict_result, confidence = predict_label(temp_img_path)
						ai_summary = generate_ai_summary(predict_result, confidence)
				except Exception:
					traceback.print_exc()
					error_message = "Could not process this file. Please upload a valid image."
				finally:
					if os.path.exists(temp_img_path):
						os.remove(temp_img_path)
		except Exception:
			traceback.print_exc()
			error_message = "Upload failed due to a server issue. Please try again."


		 

	return render_template(
		"result.html",
		prediction=predict_result,
		img_path=img_path,
		error_message=error_message,
		confidence=confidence,
		ai_summary=ai_summary
	)



 

if __name__ == '__main__':
    # Only run development server locally, not on Vercel
    import os
    if os.getenv('VERCEL') is None:
        app.run(debug=True)

