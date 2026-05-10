# Comparative-Analysis-oof-Pre-trained-CNN-Models-for-Customs-Image-CLassfication

##Google Collab Link: https://colab.research.google.com/drive/1viw3urLbhPcytO2EXFlHM562XI4h8qUX

#  A. Model Evaluation Analysis

## 1.  What were the weakest-performing classes based on the confusion matrix?
The weakest-performing classes were the ones with the lowest F1-scores because they had the poorest balance between precision and recall. Based on the classification report, these were:

Class	Precision	Recall	F1-Score
Brake Fern	0.62	0.75	0.68
Lemon Button Fern	0.67	0.72	0.69
Pacific Maidenhair Fern	0.73	0.65	0.69
Sickle Fern	0.69	0.65	0.67

These classes likely had more misclassifications in the confusion matrix because the model struggled to correctly distinguish them from other fern species.


## 2. How did Precision, Recall, and F1-score vary across classes?

The model performance varied significantly across the different fern classes:

High-performing classes
Staghorn Fern achieved the highest F1-score of 0.93, with perfect precision (1.00) and high recall (0.88).
Bird Nest Fern and Water Fern also performed strongly with F1-scores above 0.90.
Moderate-performing classes
Classes such as Asparagus Fern, Hairy Lip Fern, and western Sword Fern had balanced scores around 0.80–0.83.
Low-performing classes
Brake Fern, Sickle Fern, and Pacific Maidenhair Fern had lower precision and recall, leading to F1-scores below 0.70.

Overall:

Precision ranged from 0.62 to 1.00
Recall ranged from 0.65 to 0.94
F1-score ranged from 0.67 to 0.93

This variation indicates that some fern species were easier for the model to classify than others.


## 3. What does a low recall indicate in the model?

A low recall means that the model failed to correctly identify many actual instances of a class.

Recall=
TP+FN
TP
	​


Where:

TP = True Positives
FN = False Negatives

Low recall indicates:

The model produces many false negatives
Many real samples of that class are being missed
The model struggles to detect that class consistently

For example, if a fern class has recall = 0.65, it means only 65% of the actual samples were correctly identified, while 35% were incorrectly classified as other classes.

## 4. How does AUC score reflect model performance compared to accuracy?

The AUC (Area Under the ROC Curve) measures how well the model separates classes across different classification thresholds, while accuracy only measures the percentage of correct predictions.

Accuracy=
TP+TN+FP+FN
TP+TN
	

Key differences:

Accuracy
Simple overall correctness measure
Can be misleading when classes are imbalanced
In this model, the accuracy is 0.79 (79%)
AUC Score
Evaluates the model’s ability to distinguish between classes
Considers both true positive rate and false positive rate
More reliable for comparing classification quality
Higher AUC means better class separation

Interpretation:

A model may have high accuracy but poor AUC if it predicts dominant classes well but struggles with minority classes. A high AUC indicates stronger overall discriminative capability, even if accuracy is moderate.
Therefore, AUC provides a deeper evaluation of classification performance than accuracy alone.

# B. Model Improvement

## 5. How did data augmentation affect validation accuracy?

Data augmentation improved the model’s validation accuracy by increasing the diversity of training images without collecting new data. Techniques such as rotation, flipping, zooming, shifting, and brightness adjustments helped the CNN learn more generalized features of the fern species. As a resul the model became less sensitive to image orientation, lighting, and scale variations. Validation accuracy improved because the model could better recognize unseen images. Overfitting was reduced since the network did not memorize only the original training samples.


## 6. Why is Batch Normalization important in CNNs?

In this custom CNN, Batch Normalization was added after each convolutional layer. This:

Helped the model train more smoothly
Allowed the deeper CNN architecture to learn better plant features
Reduced training instability
Supported better overall validation performance

## 7. What role did Dropout play in improving the model?

 The high dropout values from the original guide (0.4 and 0.5) caused severe underfitting. The improved model used graduated, lighter values:

Dropout layers used: 0.05 → 0.10 → 0.15 → 0.20 → 0.30
This was better than using very high dropout values, which previously made the model underfit. The adjusted dropout values helped balance learning and regularization effectively.

## 8. How did Early Stopping prevent overfitting?
EarlyStopping monitors the validation loss during training. When the validation loss stops improving for a set number of epochs, training automatically stops and the best model weights are restored.

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
The model trained until epoch 59
EarlyStopping restored the best weights from epoch 47
This prevented the model from continuing to train after validation loss stopped improving
The final model used the best validation performance instead of the last training epoch

##C. Performance Comparison

## 9. What improvements were observed after modifying the model?
After modifying the custom CNN, all major metrics improved significantly:

Metric	Baseline Model	Improved Custom CNN	Difference
Training Accuracy	0.6365	0.8941	+0.2576
Validation Accuracy	0.5265	0.7770	+0.2504
Training Loss	1.2125	0.3810	-0.8314
Validation Loss	1.5330	0.9146	-0.6184
Macro Precision	0.5263	0.7792	+0.2529
Macro Recall	0.5141	0.7778	+0.2637
Macro F1-Score	0.5103	0.7738	+0.2635
AUC Score	0.9223	0.9771	+0.0548
The improved model exceeded the Teachable Machine / previous baseline AUC score of 0.8963, achieving a final AUC score of 0.9771.

## 10. Which enhancement contributed the most to performance improvement? Why?
The biggest improvement came from the deeper custom CNN architecture combined with Batch Normalization and learning rate scheduling:

Deeper architecture — allowed the model to learn more complex plant features
Batch Normalization — stabilized training and prevented gradient instability
ReduceLROnPlateau — lowered the learning rate when validation loss plateaued, helping the model continue improving in later stages
Data augmentation, class weights, Dropout, and EarlyStopping all contributed supporting roles, but the deeper CNN architecture and better training control were the primary reasons for the significant performance gain.

## 11. Did the gap between training and validation accuracy decrease? Explain.
The gap did not decrease — it slightly increased.

Baseline Model	Improved Model
Training Accuracy	0.6365	0.8941
Validation Accuracy	0.5265	0.7770
Generalization Gap	0.1100	0.1171
The gap increased slightly from 0.1100 → 0.1171. However, this is not a bad result because:

Both training and validation performance improved significantly
Validation accuracy increased by approximately 25 percentage points
Validation loss decreased from 1.5330 → 0.9146
This means that although the gap did not decrease, the improved model still generalized much better overall compared to the baseline.

# D. Explainability — Grad-CAM
## 12. How did Grad-CAM help in understanding model predictions?
Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes which parts of an input image most influenced the CNN's prediction by producing a heatmap overlay on the original image.

Instead of only seeing the predicted class, Grad-CAM showed the image regions the model focused on when making its decision.

In the sample image:

True class: Hibiscus_Syriacus
Predicted class: Hibiscus_Syriacus 
Confidence: 0.91
The heatmap highlighted the flower region, confirming the model was using relevant visual features such as the flower shape, color, and petal structure.

## 13. Did the improved model focus on more relevant regions? Provide evidence.
Based on the Grad-CAM result generated in the notebook, the model focused mostly on the relevant flower region rather than on the background.

The selected sample image showed a pink Hibiscus_Syriacus flower
The model correctly predicted the class with 0.91 confidence
The heatmap showed stronger activation around the flower area, especially around the central and lower-right flower region
This is evidence that the CNN learned meaningful visual features for that sample.

 Note: The Grad-CAM result shown was generated for the baseline model. To fully prove that the improved model focused on more relevant regions, Grad-CAM should also be applied to the improved custom CNN and compared against the baseline Grad-CAM output.

## 14. Why is explainability important in real-world AI applications?
Explainability is important because it helps users understand why an AI model made a certain prediction. In real-world applications, it is not enough for a model to only give an answer — users also need to know whether the model is focusing on the correct features.

For plant classification, Grad-CAM can show whether the model is focusing on the plant or flower itself instead of unrelated background elements or lighting. This matters because:

Benefit	Description
Transparency	Users can verify the model is using the right features
Debugging	If the model focuses on backgrounds, the dataset or preprocessing may need fixing
Trust	Stakeholders gain confidence when predictions can be explained
Responsible AI	Critical for deployment in domains like agriculture, medicine, or security
Explainability supports transparency, debugging, trust, and responsible AI use in any real-world syste
