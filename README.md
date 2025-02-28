# AI-powered-Voice-assisted-Object-Locator (ECE 492 Capstone G08):

## Group members:

1. Parth Dadhania (SID: 1722612)
2. Het Bharatkumar Patel (SID: 1742431)
3. Chinmoy Sahoo (SID: 1729807)
4. Dang Nguyen (SID: 1740770)

## 📜 Setup Guide for Team (WSL & Raspberry Pi)

This guide ensures that every team member can **seamlessly** set up, configure, and run the **AI-powered Voice-Assisted Object Locator (AIVOL)** project in an **identical development environment** across **WSL Ubuntu & Raspberry Pi**.

---

## 🚀 Quick Setup Guide for AIVOL

📌 **Follow these steps carefully to ensure a smooth and identical setup on your machine.**

🔹 **Supported Platforms:**  
✅ **Windows Subsystem for Linux (WSL) Ubuntu (Development)**  
✅ **Raspberry Pi OS (Deployment/Production)**

---

## 📌 Step 1: Clone the GitHub Repository

Navigate to your **desired project directory** and run:

```bash
git clone https://github.com/PrthD/AI-powered-Voice-assisted-Object-Locator.git
cd AI-powered-Voice-assisted-Object-Locator
```

---

## 📌 Step 2: Run the Setup Script

Make the setup script executable:

```bash
chmod +x setup.sh
```

Then, run the script:

```bash
./setup.sh
```

This will:
✔ **Install necessary system dependencies**  
✔ **Ensure Python 3.11.4 is installed using pyenv**  
✔ **Create and activate a virtual environment**  
✔ **Install all Python dependencies from `requirements.txt`**  
✔ **Download YOLO model weights**

---

## 📌 Step 3: Verify Installation

Once the setup is complete, verify that everything is correctly installed:

### **3.1 Check Python Version**

```bash
python --version
```

✔ Should output: `Python 3.11.4`

### **3.2 Check Installed Packages**

```bash
pip list
```

✔ Should list all dependencies (e.g., `opencv-python`, `SpeechRecognition`, `PyAudio`, `pyttsx3`, `mediapipe`, `ultralytics`, `torch`, `torchvision`).

### **3.3 Verify YOLO Model is Installed**

```bash
ls -lh models/yolo/yolo.weights
```

✔ Should show the **YOLO model weights file** (`yolo.weights`).

---

## 📌 Step 4: Running the Project

Now that everything is set up, run the main program:

```bash
python3 src/main.py
```

---

## 📌 Troubleshooting Guide

### **🐛 Issue: `pyenv: command not found`**

Run:

```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Then restart your terminal:

```bash
exec "$SHELL"
```

---

## 📌 Updating the Project

Whenever there’s a new update, run:

```bash
git pull origin main
./setup.sh
```

---

## 🎯 Summary of Steps

| Step | Description                                              |
| ---- | -------------------------------------------------------- |
| 1️⃣   | **Clone the GitHub repository**                          |
| 2️⃣   | **Run the `setup.sh` script**                            |
| 3️⃣   | **Verify installation** (`python --version`, `pip list`) |
| 4️⃣   | **Run the main program** (`python src/main.py`)          |
| 5️⃣   | **Troubleshoot issues if needed**                        |
| 6️⃣   | **Pull updates and re-run `setup.sh`**                   |

---

## 🎉 You're Now Ready to Develop & Deploy!

🚀 **This guide ensures that all team members have an identical setup, making collaboration seamless and error-free!** 🚀
