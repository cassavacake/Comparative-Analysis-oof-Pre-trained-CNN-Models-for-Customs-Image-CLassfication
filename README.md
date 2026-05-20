# Comparative-Analysis-oof-Pre-trained-CNN-Models-for-Customs-Image-CLassfication
# Google Collab Link: https://colab.research.google.com/drive/1SuL5CMI_C6-nrsFCoDEqz_F03rF7aK_k

#Overview
This laboratory work compares three pre-trained CNN models — MobileNetV2, EfficientNetB0, and ResNet50 — for classifying a custom plant/shrub image dataset. Each model was fine-tuned using transfer learning and evaluated using a comprehensive set of metrics including accuracy, precision, recall, F1-score, confusion matrix, ROC curve, AUC score, and Grad-CAM heatmaps.

The dataset consists of 20 plant classes with a minimum of 250 images per class, exceeding the 200-image requirement. All base models were loaded with pretrained ImageNet weights and frozen during training — only the custom classifier head was trained.

🎯 Objectives
#	Objective
1	Use three (3) pre-trained CNN model architectures
2	Train them on a custom dataset with ≥ 20 classes and ≥ 200 images per class
3	Evaluate each model using Accuracy, Loss, Precision, Recall, F1-score, Confusion Matrix, ROC Curve, and AUC
4	Compare model performance across all metrics
5	Apply Grad-CAM for model explainability
6	Publish results via GitHub and Google Colab
🛠 Tools and Technologies
Category	Tool / Library
🖥️ Platform	Google Colab (GPU runtime)
🐍 Language	Python 3
🧠 Framework	TensorFlow 2.x / Keras
📊 Metrics	scikit-learn
📈 Visualization	matplotlib, opencv-python (cv2)
🗄️ Data Handling	numpy, pandas
☁️ Storage	Google Drive
🤖 Pretrained Models	MobileNetV2, EfficientNetB0, ResNet50
🌱 Dataset
Detail	Value
📁 Total Classes	20
🖼️ Total Images	5,653
🏋️ Training Images	4,523 (80%)
✅ Validation Images	1,130 (20%)
📏 Min Images Per Class	250 ✓ (required: 200)
🔲 Image Size	224 × 224
📦 Batch Size	32

# 📊 Results and Outputs
🏆 Model Performance Summary

| Model | Train Accuracy | Train Loss | Test Accuracy | Test Loss | Precision | Recall | F1-score | AUC |
|---|---|---|---|---|---|---|---|---|
| MobileNetV2 | 0.708 | 1.03 | 0.738 | 0.9931 | 0.7461 | 0.738 | 0.7371 | 0.9713 |
| EfficientNetB0 | 0.0503 | 2.9994 | 0.035 | 3.0002 | 0.0012 | 0.035 | 0.0024 | 0.4828 |
| NASNetMobile | 0.2988 | 2.3653 | 0.424 | 2.2078 | 0.2726 | 0.248 | 0.2468 | 0.8005 |
# 📉 Training Summary
| Model | Train Accuracy | Train Loss | Test Accuracy | Test Loss |
|---|---|---|---|---|
| NASNetMobile | 0.7732 | 0.8009 | 0.8018 | 0.7352 |
| EfficientNetB0 | 0.7115 | 1.0334 | 0.7743 | 0.8875 |
| MobileNetV2 | 0.6863 | 1.0592 | 0.7504 | 0.9571 |
# 📋 Full Part 12 Comparison Table
| Model | Train Accuracy | Train Loss | Test Accuracy | Test Loss | Precision | Recall | F1-score | AUC |
|---|---|---|---|---|---|---|---|---|
| Pre-trained (Model 1) MobileNetV2 | 0.708 | 1.03 | 0.738 | 0.9931 | 0.7461 | 0.738 | 0.7371 | 0.9713 |
| Pre-trained (Model 2) EfficientNetB0 | 0.0503 | 2.9994 | 0.035 | 3.0002 | 0.0012 | 0.035 | 0.0024 | 0.4828 |
| Pre-trained (Model 3) NASNetMobile | 0.2988 | 2.3653 | 0.424 | 2.2078 | 0.2726 | 0.248 | 0.2468 | 0.8005 |
| Model from Teachable Machine | 0.6365 | 1.2125 | 0.5265 | 1.5330 | 0.5263 | 0.5141 | 0.5103 | 0.8963 |
| 1st Model | 0.9996 | 0.0057 | 0.5929 | 2.9057 | 0.5904 | 0.5859 | 0.5826 | 0.9230 |
| 2nd Model Enhancement | 0.6374 | 1.1523 | 0.5363 | 1.5020 | 0.5469 | 0.5309 | 0.5213 | 0.9274 |
| 3rd Model – The Good Model | 0.8941 | 0.3810 | 0.7770 | 0.9146 | 0.7792 | 0.7778 | 0.7738 | 0.9771 |

🔻 5 Weakest Classes Per Model
MobileNetV2 — Weakest Classes
EfficientNetB0 — Weakest Classes
ResNet50 — Weakest Classes

🔥 Grad-CAM Results
All three models were tested on the same validation image: Hibiscus_Syriacus (class index 11)

Model	Predicted Class	Correct?	Confidence
MobileNetV2	Hibiscus_Syriacus	✅	0.4706
EfficientNetB0	Hibiscus_Syriacus	✅	0.9008
ResNet50	Hibiscus_Syriacus	✅	0.9515
All three models correctly predicted the class. Grad-CAM heatmaps showed that all models focused on the flower region, but ResNet50 had the most concentrated and precise activation on the petals and flower center. MobileNetV2's heatmap was broader and less focused, which matches its lower confidence score.

# ❓ Guide Question Answers
 ## A. Model Performance
## 1. Which pre-trained model achieved the highest accuracy? Why?

ResNet50 achieved the highest accuracy at 81.50%. I think this is because ResNet50 uses residual connections (skip connections), which help prevent vanishing gradients and allow the network to learn more stable and complex features. Even with the base frozen, its deeper pre-trained features were better at distinguishing our 20 plant classes. The architecture has more capacity to capture visual details like leaf textures and flower shapes compared to MobileNetV2 or EfficientNetB0.

## 2. Which model had the lowest performance? What could be the reason?

MobileNetV2 had the lowest test accuracy at 73.45%. MobileNetV2 is built for lightweight and fast inference, so it has fewer parameters and less representational capacity. It's great for mobile deployments but wasn't as strong for distinguishing between 20 visually similar plant classes — especially ones like Barberries and Aronia_Melanocarpa that look quite alike.

## 3. How did loss values compare across models?

ResNet50 had the lowest test loss at 0.7352, meaning it was more confident and accurate. EfficientNetB0 was in the middle at 0.8875, and MobileNetV2 had the highest at 0.9571. Higher training losses for MobileNetV2 and EfficientNetB0 (~1.03–1.06) compared to ResNet50 (~0.80) suggest that ResNet50's features transferred more effectively to our dataset.

B. Evaluation Metrics
4. Why is accuracy not enough to evaluate a model?

Accuracy alone doesn't give the full picture — especially when classes aren't perfectly balanced. A model could score high accuracy by performing well on common classes while completely failing on rare ones. That's why we also measure precision (how many predicted positives were truly correct), recall (how many actual positives were captured), and F1-score (the balance between both). In our dataset, some classes like Barberries only had 39 validation samples, so accuracy alone could mask poor performance on those smaller classes.

## 5. Which model had the best F1-score? What does it indicate?

ResNet50 had the best macro F1-score at 0.8112. A high F1-score means the model has a good balance between precision and recall across all 20 classes — it doesn't just predict confidently, it also captures most of the actual instances of each class. This makes ResNet50 the most reliable model overall, not just the most accurate one.

## 6. How did Precision and Recall differ across models?

For all three models, precision and recall were close to each other, meaning the models were fairly balanced:

Model	Macro Precision	Macro Recall
ResNet50	0.8128	0.8116
EfficientNetB0	0.7795	0.7703
MobileNetV2	0.7352	0.7323
ResNet50 maintained the smallest gap between precision and recall. MobileNetV2 showed higher precision than recall for some classes (e.g., Rubus: precision 0.8108, recall 0.6250), meaning it was too conservative and missed actual instances.

## C. Confusion Matrix Analysis
7. Which classes were frequently misclassified?


## 8. What patterns did you observe in the confusion matrix?

The main pattern is that models struggled most with visually similar shrubs — like plants with small dark berries or similar leaf structures. Classes with very distinctive visual features (like Callistemon or Lapageria) had much higher recall across all models. As model power increased from MobileNetV2 → EfficientNetB0 → ResNet50, the number of off-diagonal misclassifications in the confusion matrix decreased steadily.

D. ROC and AUC
## 9. Which model had the highest AUC score?

ResNet50 had the highest overall AUC at 0.9841. Per-class highlights include Lapageria (0.9993) and Forsythia (0.9984). All three models performed well here — MobileNetV2 scored 0.9739 and EfficientNetB0 scored 0.9783 — but ResNet50 consistently led across individual classes too.

## 10. What does AUC tell us about model performance?

AUC measures how well the model separates each class from the rest, regardless of the classification threshold. A score of 1.0 means perfect separation; 0.5 means no better than random. All our models scored above 0.97, showing they're very good at ranking the correct class higher than others. This is especially useful for multi-class problems because it evaluates across all decision thresholds, not just one fixed cutoff.

# E. Explainability (Grad-CAM)
11. What did Grad-CAM reveal about model decision-making?

Grad-CAM revealed which parts of the image each model activated most strongly when making a prediction. For the Hibiscus_Syriacus test image, all three models highlighted the flower region — which is exactly the right area. This confirmed that the models were learning meaningful visual patterns rather than random noise or background features.

## 12. Did the model focus on relevant image regions?

Yes. All three models focused on the flower region of the Hibiscus_Syriacus image. However, MobileNetV2's heatmap was broader and less precise, activating on a wider area including some background. EfficientNetB0 and ResNet50 showed stronger, more localized activation on the petals and flower center. The more powerful models focused on the right areas with more precision.

## 13. Which model produced the most meaningful heatmaps?

ResNet50 produced the most meaningful Grad-CAM heatmaps. Its activations were concentrated on the flower petals and center — the most relevant features for identifying Hibiscus_Syriacus. This is consistent with its confidence score of 0.9515, compared to EfficientNetB0 (0.9008) and MobileNetV2 (0.4706).

F. Model Comparison and Improvement
## 14. Which model would you recommend for deployment? Why?

I'd recommend ResNet50 for deployment. It had the best accuracy (81.50%), F1-score (0.8112), AUC (0.9841), and the most focused Grad-CAM visualizations. If deployment is on a mobile or resource-constrained device, EfficientNetB0 is a better tradeoff — it still achieved 77.52% accuracy while being more efficient.

## 15. How can you further improve your best-performing model?

Strategy	Description
🔓 Fine-tuning	Unfreeze deeper base layers and retrain with a lower learning rate
🔄 Data Augmentation	Add random flips, rotations, color jitter, and cropping
⏳ Longer Training	More epochs with a learning rate scheduler
📂 More Data	Collect more images for weaker classes like Barberries and Aronia_Melanocarpa
⚖️ Class Weights	Use class weighting or oversampling for underperforming classes
G. Real-World Application
16. How can your model be applied in real-world scenarios?

Use Case	Description
🌻 Gardening Apps	Users photograph a plant and get an instant species ID
🌍 Environmental Monitoring	Quickly catalogue plant species during field surveys
📚 Education	Students use it for botany learning in the field
🌾 Agriculture	Identify unfamiliar plants, weeds, or invasive species
17. What are the risks of deploying an inaccurate model?

☠️ Toxic plant misidentification — misidentifying a poisonous plant as safe could be dangerous
🌾 Bad agricultural decisions — wrong species ID could lead to missed treatment of invasive plants
📉 Loss of user trust — repeated wrong answers will drive users away
⚖️ Class bias — weak performance on underrepresented classes means some species always get misidentified
Always display the model's confidence score and remind users to verify critical decisions with a professional.

## 18. How can this system be integrated into a mobile/web app?

Platform	Approach
📱 Mobile App	Export to TFLite, run on-device via Android/iOS
🌐 Web App	Use TensorFlow.js for browser inference, or Flask/FastAPI as a backend API
☁️ Cloud API	Host on Google Cloud / AWS, expose as a REST endpoint for any frontend
🔭 Observations
✅ ResNet50 dominated across all metrics — highest accuracy, F1-score, AUC, and best Grad-CAM focus
📈 All three models achieved AUC > 0.97, meaning they're all strong at class ranking even if absolute accuracy differs
⚠️ Barberries and Aronia_Melanocarpa were the hardest classes to classify across all three models — likely due to similar visual traits
🔧 The corrected preprocessing pipeline (using model-specific preprocess_input) made a clear difference compared to the generic Rescaling(1./255) approach
MobileNetV2's Grad-CAM confidence was only 0.47 vs. ResNet50's 0.95 — a major gap showing how different their feature representations are
The 1st Model in the full table showed 99.96% training accuracy but only 59.29% test accuracy — a textbook case of overfitting
 Challenges Encountered
# Challenge	Description
🔧 Preprocessing Mismatch	First run used generic Rescaling(1./255) for all models — EfficientNetB0 and ResNet50 require specific preprocessing; fixed in a second corrected run
💾 GPU Memory	Had to clear the Keras session and collect garbage between training runs to avoid Colab OOM errors
🔍 Grad-CAM Layer Lookup	Finding the last Conv2D layer inside each nested base model required reverse layer iteration — different models had different layer names (Conv_1, top_conv, conv5_block3_3_conv)
⏹️ Early EarlyStopping	Some models stopped before 10 epochs because val_loss plateaued — a known limitation of fully frozen transfer learning
🌿 Visually Similar Classes	Barberries, Aronia_Melanocarpa, and Actinidia were hard to distinguish even for ResNet50 — more data or fine-tuning is likely needed
✅ Conclusion
In this laboratory work, I trained and compared three pre-trained CNN models — MobileNetV2, EfficientNetB0, and ResNet50 — on a custom 20-class plant image dataset using transfer learning.

ResNet50 was the clear winner: highest test accuracy (81.50%), best macro F1-score (0.8112), highest AUC (0.9841), and most focused Grad-CAM visualizations. EfficientNetB0 came in second, while MobileNetV2 was the weakest — though still reasonable given its lightweight design.

The activity reinforced that accuracy alone isn't enough — F1-score, AUC, confusion matrix analysis, and Grad-CAM together give a much more complete picture of model performance. It also showed how important proper preprocessing is for pre-trained models, and how transfer learning can work well even without fine-tuning the base layers.

# 💭 Reflection
This was honestly one of the more complex lab works so far. Training three separate models, evaluating them across multiple metrics, and applying Grad-CAM on top of everything — there's a lot going on at once. The part that caught me off guard the most was how much the preprocessing function matters. I didn't think using the wrong normalization for EfficientNetB0 or ResNet50 would have such a visible impact on training, but it did — enough that we had to redo the whole training run.

Looking at the confusion matrices was also really interesting. Seeing the same classes (Barberries, Aronia_Melanocarpa) struggle consistently across all three models made me realize that sometimes the issue is the data itself, not just the model. Those two classes look really similar, and no amount of model swapping is going to fully fix that without better data or augmentation.

Grad-CAM was probably my favorite part of this lab because it made everything feel more real. Instead of just looking at numbers in a table, I could actually see where each model was looking. It explains why ResNet50 was so confident — it focused on the right parts of the flower — and why MobileNetV2 was unsure — its heatmap was spread out all over the place. That's something I want to explore more going forward.
