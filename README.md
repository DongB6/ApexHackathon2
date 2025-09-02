🩺 Skin Disease Detection

Tagline: AI-powered early screening for healthier lives.

📖 Overview

Skin diseases affect millions of people worldwide, but access to dermatologists can be limited. Skin Disease Detection is an AI-powered application that uses deep learning to analyze skin images and provide preliminary condition insights.

⚠️ Disclaimer: This tool is for educational and awareness purposes only. It does not replace professional medical advice.

🚀 Features

Upload an image of a skin condition.

Deep learning model (CNN) analyzes the image.

Returns predicted condition(s) with confidence scores.

Simple and interactive UI built with Streamlit.

🛠️ Tech Stack

Python 3.10+

PyTorch (for CNN model)

Streamlit (for web app)

OpenCV / PIL (for image processing)

NumPy, Pandas (data handling)

⚙️ Installation

Clone the repository

git clone https://github.com/your-username/skin-disease-detection.git
cd skin-disease-detection


Set up virtual environment

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate


Install dependencies

pip install -r requirements.txt


Run the app

streamlit run app.py

📂 Project Structure
skin-disease-detection/
│── data/               # Dataset (not included in repo)
│── models/             # Saved models
│── notebooks/          # Jupyter notebooks for training & testing
│── app.py              # Streamlit application
│── model.py            # CNN model definition
│── train.py            # Training script
│── requirements.txt    # Dependencies
│── README.md           # Project documentation

📊 Model Details

Model type: Convolutional Neural Network (CNN)

Input: Preprocessed skin image (resized, normalized)

Output: Predicted label(s) + confidence score

Example normalization:

𝑥
′
=
𝑥
−
𝜇
𝜎
x
′
=
σ
x−μ
	​

💡 Inspiration

We were inspired by the need for accessible early detection tools in healthcare. With AI, even a smartphone can help raise awareness about skin health.

📚 Challenges

Finding open, diverse skin condition datasets.

Training CNNs within hackathon time limits.

Balancing speed and accuracy for demo purposes.

🌍 Future Plans

Expand to cover more conditions.

Improve accuracy with larger datasets.

Integrate telemedicine for doctor consultations.

🤝 Contributing

Pull requests are welcome! If you’d like to suggest improvements, feel free to open an issue.

📜 License

MIT License — free to use, modify, and distribute.