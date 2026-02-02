# Gun Detection - Object Detection System

A deep learning-based object detection system that identifies and locates guns in images using Faster R-CNN with ResNet50 backbone. This project includes a complete ML pipeline from data ingestion to a production-ready REST API.

## 🎯 Features

- **Deep Learning Model**: Faster R-CNN with ResNet50 backbone for accurate gun detection
- **REST API**: FastAPI-based endpoint for real-time predictions
- **Data Pipeline**: Automated data ingestion from Kaggle with preprocessing
- **Training Monitoring**: TensorBoard integration for tracking training metrics
- **Version Control**: DVC for data and model versioning
- **Cloud Integration**: Google Cloud Storage support for data management
- **Production Ready**: Easy deployment with Uvicorn

## 📋 Project Structure

```
Gun Detection/
├── main.py                          # FastAPI application
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup configuration
├── README.md                        # This file
├── src/                             # Source code
│   ├── __init__.py
│   ├── logger.py                    # Logging configuration
│   ├── custom_exception.py          # Custom exception handling
│   ├── data_ingestion.py            # Dataset download & extraction
│   ├── data_processing.py           # Data preprocessing & augmentation
│   ├── model_architecture.py        # Faster R-CNN model definition
│   └── model_training.py            # Training pipeline
├── config_GunO/                     # Configuration files
│   ├── __init__.py
│   └── data_ingestion_config.py     # Data ingestion settings
├── artifacts/                       # Generated files
│   ├── models/
│   │   └── fasterrcnn.pth          # Trained model checkpoint
│   ├── raw/                         # Raw dataset
│   │   ├── Images/                 # Training images
│   │   └── Labels/                 # YOLO format annotations
│   ├── models.dvc                  # DVC model tracking
│   └── raw.dvc                     # DVC data tracking
├── tensorboard_logs/                # Training logs
│   └── [timestamp]/                # Tensorboard event files
├── notebook/                        # Jupyter notebooks
│   └── guns-object-detection.ipynb
└── logs/                            # Application logs
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10
- CUDA 11.0+ (for GPU acceleration, optional)
- Git

### Installation

1. **Clone/Setup the project**:
   ```bash
   cd "Gun Detection"
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup the package**:
   ```bash
   pip install -e .
   ```

## 🔧 Configuration

### Kaggle Dataset Setup

To download the gun detection dataset from Kaggle:

1. Create a `.env` file in the project root:
   ```bash
   KAGGLE_API_TOKEN=your_kaggle_api_token_here
   ```

2. Get your Kaggle API token from [Kaggle Settings](https://www.kaggle.com/account)

3. Configure dataset in `config_GunO/data_ingestion_config.py`:
   ```python
   DATASET_NAME = "your-kaggle-dataset-name"
   TARGET_DIR = "artifacts"
   ```

## 📊 Training the Model

### Run the complete training pipeline:

```python
from src.data_ingestion import DataIngestion
from src.model_training import ModelTraining
import torch

# Data Ingestion
data_ingestion = DataIngestion("dataset_name", "artifacts")
data_ingestion.download_and_extract()

# Model Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = ModelTraining(
    model_class=YourModel,
    num_classes=2,  # gun and background
    learning_rate=1e-4,
    epochs=10,
    dataset_path="artifacts/raw",
    device=device
)
trainer.train()
```

### Monitor Training with TensorBoard

```bash
tensorboard --logdir=tensorboard_logs
```

Then open `http://localhost:6006` in your browser.

## 🔮 Using the API

### Start the API Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server will be available at `http://localhost:8000`

### API Endpoints

#### 1. Welcome Endpoint
```
GET /
```
**Response**:
```json
{"message": "Welcome to the Guns Object Detection API"}
```

#### 2. Prediction Endpoint
```
POST /predict/
```
**Parameters**:
- `file` (UploadFile): Image file (JPG, PNG, etc.)

**Response**: PNG image with bounding boxes around detected guns

**Example using curl**:
```bash
curl -F "file=@path/to/image.jpg" http://localhost:8000/predict/ -o output.png
```

**Example using Python**:
```python
import requests

with open("image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict/", files=files)
    
    # Save the result
    with open("output.png", "wb") as out:
        out.write(response.content)
```

## 📈 Model Details

### Architecture
- **Base Model**: Faster R-CNN with ResNet50 backbone
- **Pretrained on**: ImageNet
- **Fine-tuned on**: Gun detection dataset
- **Classes**: 2 (Gun, Background)
- **Confidence Threshold**: 0.7

### Training Configuration
- **Optimizer**: Adam
- **Learning Rate**: 1e-4
- **Batch Size**: Configurable
- **Loss Function**: Multi-task loss (classification + localization)

## 📦 Dependencies

Key packages used:
- `torch` - Deep learning framework
- `torchvision` - Computer vision utilities
- `fastapi` - REST API framework
- `uvicorn` - ASGI server
- `Pillow` - Image processing
- `numpy` - Numerical computing
- `dvc` - Data version control
- `tensorboard` - Training visualization
- `kagglehub` - Kaggle dataset download
- `google-cloud-storage` - Cloud storage integration

See `requirements.txt` for complete list.

## 📝 Dataset Format

Images are stored with YOLO format annotations:
- **Images**: `artifacts/raw/Images/`
- **Labels**: `artifacts/raw/Labels/` (one `.txt` file per image)

Label format (YOLO):
```
<class_id> <x_center> <y_center> <width> <height>
```
- Coordinates are normalized (0-1 range)
- `class_id`: 0 for gun, 1 for background

## 🔍 Logging

Application logs are saved in the `logs/` directory. Logs include:
- Data ingestion status
- Model training progress
- API requests and responses
- Errors and exceptions

Access logs via:
```python
from src.logger import get_logger
logger = get_logger(__name__)
```

## ⚙️ Advanced Configuration

### GPU/CPU Selection
The model automatically detects GPU availability:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Model Checkpoint Loading
```python
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
model.load_state_dict(torch.load("artifacts/models/fasterrcnn.pth"))
model.eval()
```

## 🐛 Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce batch size in training configuration

### Issue: Kaggle API token not found
**Solution**: Ensure `.env` file exists with `KAGGLE_API_TOKEN` variable set

### Issue: Model not detecting guns
**Solution**: 
- Check confidence threshold (currently 0.7)
- Ensure model is loaded correctly
- Verify image quality and format

### Issue: API returns 404
**Solution**: Ensure the file path is correct in POST request

## 📚 References

- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [DVC Documentation](https://dvc.org/doc)

## 🤝 Contributing

For improvements and bug fixes:
1. Create a new branch
2. Make changes
3. Test thoroughly
4. Submit a pull request

## 📄 License

This project is intended for educational and research purposes.

## 👨‍💻 Author

**Rishabh Anand**

---

## 📞 Support

For issues, questions, or suggestions, please open an issue in the project repository.
Email- Rishabhanand0200@gmail.com

---

**Last Updated**: February 2, 2026
