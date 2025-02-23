# Race Strategy Optimizer - Tire Compound Prediction

# Link to Youtube Video: https://youtu.be/SYUkSQ-9xs0

## Project Overview  
This project focuses on optimizing race strategy by predicting the best tire compound for different track and weather conditions.  
The goal is to build a machine learning model that helps teams choose the best tires for performance and durability.  
The dataset consists of multiple factors such as temperature, track surface, weather, and driving style.  
We compared traditional machine learning (Logistic Regression) with deep learning (Neural Networks) for performance evaluation.  

---

## Dataset Used  
The dataset includes various features that impact tire selection:  
- **Lap Number**  
- **Track Temperature**  
- **Weather Conditions**  
- **Track Surface Type**  
- **Downforce Setup**  
- **Driving Style**  
- **Race Strategy**  
- **Competitor Strategy**  
- **Laps Remaining**  
- **Tire Compound (Target Variable)**  

---

## Results and Comparison  

| Model       | Optimizer | Regularizer | Epochs | Early Stopping | Layers | Learning Rate | Accuracy | F1-score | Precision | Recall | Dropout |
|------------|------------|-------------|--------|---------------|--------|---------------|----------|----------|-----------|--------|----------|
| Logistic Regression | N/A | L2 (0.01) | N/A | No | N/A | N/A | 0.2900 | 0.2948 | 0.2874 | 0.3464 | None |
| Neural Network 1 | Adam | None | 100 | No | 3 | 0.001 | 0.9100 | 0.9074 | 0.9154 | 0.9100 | None |
| Neural Network 2 | Adam | L2 (0.001) | 150 | Yes | 5 | 0.01 | 0.9300 | 0.9215 | 0.9320 | 0.9300 | None |
| Neural Network 3 | RMSprop | None | 100 | Yes | 5 | 0.005 | 0.9400 | 0.9348 | 0.9405 | 0.9400 | 0.2 |

---

## **Summary of Findings**  
1. **Comparison of ML and Neural Networks**  
   - The **Logistic Regression model** had poor performance with an **accuracy of 0.29** and an **F1-score of 0.29**, showing that it was not suitable for this classification problem.  
   - The **Neural Networks significantly outperformed** traditional ML, with the **best-performing model reaching 94% accuracy**.  

2. **Best Model and Hyperparameter Settings**  
   - The **fourth model** (Neural Network with **RMSprop optimizer, 5 layers, and a learning rate of 0.005**) performed the best.  
   - Adding **dropout layers and tuning the learning rate** helped improve generalization.  

3. **Effect of Optimizer and Regularization**  
   - **Adam optimizer** worked well, but **RMSprop with a higher learning rate (0.005) performed the best**.  
   - **L2 regularization** helped in avoiding overfitting.  
   - **Early stopping** prevented overtraining and helped in selecting the best model.  

---

## **Final Conclusion**  
- **Neural Networks clearly outperformed Logistic Regression**, confirming that deep learning is better suited for this task.  
- **The best combination was the fourth model**, which achieved the highest accuracy and F1-score.  
- **Hyperparameter tuning**, including optimizers, learning rates, and regularization, played a crucial role in improving performance.  

---

## **Model Usage - Making Predictions**  

To make predictions with the best model:

```python
from tensorflow.keras.models import load_model
import numpy as np

def make_predictions(model_path, X):
    model = load_model(model_path)
    predictions_prob = model.predict(X)
    predictions = np.argmax(predictions_prob, axis=-1)
    return predictions

# Example usage:
model_path = "saved_models/final_model.keras"
test_data = np.array([[10, 35.0, 2, 1, 0, 1, 1, 0, 2, 15]])  # Example input
prediction = make_predictions(model_path, test_data)

print("Predicted Tire Compound:", prediction[0])
