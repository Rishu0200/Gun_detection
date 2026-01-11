import os
import shutil
import zipfile
import tempfile
import kagglehub
from pathlib import Path
from dotenv import load_dotenv
from src.logger import get_logger
from src.custom_exception import CustomException
from config_GunO.data_ingestion_config import DATASET_NAME, TARGET_DIR

load_dotenv()
logger = get_logger(__name__)

# Kaggle authentication
try:
    KAGGLE_API_TOKEN = os.getenv("KAGGLE_API_TOKEN")
    if not KAGGLE_API_TOKEN:
        raise ValueError("KAGGLE_API_TOKEN environment variable not found")
    logger.info("KAGGLE_API_TOKEN retrieved successfully")
except (AttributeError, ValueError) as e:
    logger.error(f"Failed to retrieve KAGGLE_API_TOKEN: {str(e)}")
    raise CustomException("KAGGLE_API_TOKEN not found", e)

class DataIngestion:
    def __init__(self, dataset_name: str, target_dir: str):
        self.dataset_name = dataset_name
        self.target_dir = Path(target_dir)  
        
    def create_raw_dir(self) -> str:
        """Create raw directory if it doesn't exist."""
        raw_dir = self.target_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Raw directory ready: {raw_dir}")
        return str(raw_dir)
    
    def extract_images_and_labels(self, dataset_path: str, raw_dir: str):
        """Handle both direct images and Images/Labels folder structures."""
        try:
            dataset_path = Path(dataset_path)
            raw_path = Path(raw_dir)
            
            
            images_dest = raw_path / "Images"
            images_dest.mkdir(exist_ok=True)
            
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            all_images = []
            for ext in image_extensions:
                all_images.extend(dataset_path.rglob(ext))
            
            logger.info(f"Found {len(all_images)} image files")
            
            # Copy all images to raw_dir/Images/
            for img_path in all_images:
                rel_path = img_path.relative_to(dataset_path)
                dest_path = images_dest / rel_path.name
                shutil.copy2(img_path, dest_path)
                logger.debug(f"Copied {img_path.name} to Images/")
            
            # Handle Labels if they exist (txt files for YOLO/object detection)
            label_files = list(dataset_path.rglob("*.txt"))
            if label_files:
                labels_dest = raw_path / "Labels"
                labels_dest.mkdir(exist_ok=True)
                for lbl_path in label_files:
                    shutil.copy2(lbl_path, labels_dest / lbl_path.name)
                logger.info(f"Copied {len(label_files)} label files")
            else:
                logger.warning("No label files (.txt) found - creating empty Labels folder")
                (raw_path / "Labels").mkdir(exist_ok=True)
                
            logger.info(f"Data organized: {len(all_images)} images in {images_dest}")
            
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            raise CustomException("Error during data organization", e)

    
    def download_dataset(self, raw_dir: str):
        """Download dataset using kagglehub."""
        try:
            dataset_path = kagglehub.dataset_download(self.dataset_name)
            logger.info(f"Downloaded dataset to: {dataset_path}")
            self.extract_images_and_labels(dataset_path, raw_dir)
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            raise CustomException("Dataset download failed", e)
    
    def run(self):
        """Execute complete data ingestion pipeline."""
        try:
            raw_dir = self.create_raw_dir()
            self.download_dataset(raw_dir)
            logger.info("Data ingestion pipeline completed successfully")
        except Exception as e:
            logger.error(f"Data ingestion pipeline failed: {str(e)}")
            raise CustomException("Data ingestion pipeline failed", e)

if __name__ == "__main__":
    data_ingestion = DataIngestion(DATASET_NAME, TARGET_DIR)
    data_ingestion.run()
