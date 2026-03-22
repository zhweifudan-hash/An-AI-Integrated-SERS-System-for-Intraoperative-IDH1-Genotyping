# An-AI-Integrated-SERS-System-for-Intraoperative-IDH1-Genotyping

pip install pandas numpy scikit-learn tqdm pybaselines torch torchvision opencv-python matplotlib

1. ResNet Regression & Visualization
2. python raman_resnet_explainable.py \
    --csv_path data/your_data.csv \
    --start_idx 30 \
    --end_idx 530 \
    --backbone resnet50 \
    --epochs 50

LASSO Feature Selection
python lasso_feature_selection.py
