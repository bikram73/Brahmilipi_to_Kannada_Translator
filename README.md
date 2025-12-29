# Brahmilipi to Kannada Character Recognition System

An AI-powered character recognition system that converts Brahmilipi script images to Kannada Unicode characters using deep learning and computer vision techniques.

## 🚀 Features

- **Deep Learning Model**: CNN-based architecture with TensorFlow/Keras for accurate character recognition
- **Synthetic Data Generation**: Advanced pipeline creating training data with noise injection and geometric transformations
- **Web Interface**: Flask-based application with real-time image upload and prediction
- **Multi-Character Support**: Recognizes 7 core Kannada vowels and consonants (ಅ, ಆ, ಇ, ಈ, ಉ, ಊ, ಕ)
- **Image Processing**: Robust preprocessing pipeline with OpenCV for various image formats

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow, Keras, scikit-learn
- **Computer Vision**: OpenCV
- **Frontend**: HTML, CSS, JavaScript (jQuery)
- **Data Processing**: NumPy, JSON

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Bralmilipi_to_Kannada_Translator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (if not already trained)
   ```bash
   python src/train_model.py
   ```

## 🚀 Usage

### Running the Web Application

1. **Start the Flask server**
   ```bash
   python src/main.py
   ```

2. **Access the application**
   - Open your browser and navigate to `http://127.0.0.1:5000`
   - Upload an image containing Brahmilipi characters
   - Get instant Kannada character predictions

### Using the Model Programmatically

```python
from src.train_model import predict_character, generate_synthetic_images

# Generate test image
synthetic_images, labels = generate_synthetic_images(num_images_per_class=1)
test_image = synthetic_images[0]

# Predict character
predicted_char = predict_character(test_image)
print(f"Predicted character: {predicted_char}")
```

## 📁 Project Structure

```
Bralmilipi_to_Kannada_Translator/
├── src/
│   ├── main.py                 # Flask web application
│   ├── train_model.py          # Model training and prediction
│   ├── preprocess.py           # Image preprocessing utilities
│   ├── utils.py                # Helper functions
│   ├── templates/
│   │   ├── index.html          # Main web interface
│   │   └── display_images.html # Image display page
│   └── static/
│       └── uploads/            # Uploaded images directory
├── data/
│   └── mapping.json            # Character mappings
├── character_mappings.json     # Model character mappings
├── kannada_synthetic_character_model.h5  # Trained model
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🧠 Model Architecture

- **Input Layer**: 64x64x1 grayscale images
- **Convolutional Layers**: 2 Conv2D layers with BatchNormalization
- **Pooling**: MaxPooling2D for feature reduction
- **Regularization**: Dropout layers (0.25-0.5) to prevent overfitting
- **Output**: 7-class softmax classification for Kannada characters

## 📊 Model Performance

- **Training Accuracy**: ~19% (with synthetic data)
- **Validation Accuracy**: ~22%
- **Test Accuracy**: ~25%

*Note: Performance can be improved with real character image datasets*

## 🔄 Supported Characters

| Brahmilipi | Kannada | Unicode |
|------------|---------|---------|
| Image1     | ಅ       | U+0C85  |
| Image2     | ಆ       | U+0C86  |
| Image3     | ಇ       | U+0C87  |
| Image4     | ಈ       | U+0C88  |
| Image5     | ಉ       | U+0C89  |
| Image6     | ಕ       | U+0C95  |
| Image7     | ಊ       | U+0C8A  |

## 🚧 Limitations

- Currently trained on synthetic data - real character images would improve accuracy
- Limited to 7 characters - can be extended to full Kannada alphabet
- Model accuracy needs improvement with better training data

## 🔮 Future Enhancements

- [ ] Expand character set to complete Kannada alphabet
- [ ] Implement real character image dataset collection
- [ ] Add data augmentation techniques
- [ ] Improve model architecture for better accuracy
- [ ] Add batch processing capabilities
- [ ] Implement character sequence recognition

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Created with ❤️ for preserving and digitizing ancient scripts

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- OpenCV community for computer vision tools
- Flask team for the web framework
- Contributors to the Kannada Unicode standard