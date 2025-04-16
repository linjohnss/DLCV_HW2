# NYCU Computer Vision 2025 Spring HW2

**StudentID:** 312554027  
**Name:** 林晉暘

---

## 📌 Introduction

This project targets digit recognition with two subtasks:  
Task 1—localising every individual digit, and  
Task 2—predicting the entire multi‑digit number from a single image.  
The dataset contains 30 062 training, 3 340 validation, and 13 068 test RGB images, annotated in COCO format.  
All solutions must employ Faster~R‑CNN ren2015faster, refrain from external data, and remain reproducible; however, architectural changes within Faster R‑CNN are encouraged and ImageNet pre‑training is permitted.


---

## 🛠️ How to install

1. Create and activate a virtual environment:
   ```bash
    conda create -n cv_hw2 python=3.9
    conda activate cv_hw2
   ```

2. Install dependencies:
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   ```

---

## 🚀 How to train

You can train the model by running:

```bash
python main.py 
```

---

## 🧪 Evaluation

After training, you can evaluate the best model and generate validation confusion matrix:

```bash
python main.py --eval_only --output_dir ./outputs
```

---

## 📊 Performance snapshot

![image](leaderboard.png)

