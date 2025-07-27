# SolarGuard: Intelligent Defect Detection on Solar Panels using  DeepLearning

**Problem Statement:**

Solar energy is a crucial renewable resource, but the accumulation of dust, snow, bird droppings, and physical/electrical damage on solar panels reduces their efficiency. While manual monitoring is time-consuming and expensive, automated detection can help improve efficiency and reduce maintenance costs.
This project aims to develop deep learning models for both classification and object detection to accurately identify and localize different types of obstructions or damages on solar panels. The objective is to:
  1) Classify solar panel images into six categories: Clean, Dusty, Bird-Drop, Electrical-Damage, Physical-Damage, and Snow-Covered.
  2) Detect and localize the presence of dust, bird droppings, or damages on the panel using object detection models.

**Business Use Cases:**
1) **Automated Solar Panel Inspection:** Develop an AI-based system to automatically classify and detect issues on solar panels, reducing the need for manual inspections.
2) **Optimized Maintenance Scheduling:** Identify which panels require immediate cleaning or repair, optimizing maintenance efforts and reducing operational costs.
3) **Efficiency Monitoring:** Analyze the impact of different obstructions on solar panel efficiency and generate reports for performance improvement.
4) **Smart Solar Farms:** Integrate AI models into smart solar farms to trigger alerts for cleaning/repair, ensuring maximum energy production.

**Objective:**
**Aim:**
Develop a classification model to categorize solar panel images into one of six conditions:
Clean
Dusty
Bird-Drop
Electrical-Damage
Physical-Damage
Snow-Covered
**Use Case:**
Automate the process of identifying the condition of solar panels from images.
Provide insights into common issues affecting panel efficiency.
Help solar maintenance teams prioritize cleaning and repair tasks.
Possible Inputs (Features):
Raw solar panel images
**Target:**
A category label indicating the panel condition

**Approach:**
**1. Data Preprocessing & Annotation:**
Perform image augmentation to balance the dataset.
Resize images to a suitable dimension for deep learning models.
Annotate images with bounding boxes for object detection tasks.
Normalize pixel values for better model performance.

**2. Model Training:**
Classification: Train CNN models (ResNet, EfficientNet, MobileNet) for panel condition classification.

**3. Model Evaluation:**
Classification Metrics: Accuracy, Precision, Recall, F1-Score.

**4. Deployment:**
Deployed a Streamlit web app and got classification results for panel conditions.
