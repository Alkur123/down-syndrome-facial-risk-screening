# Model Card: Down Syndrome Facial Risk Screening Ensemble

## Model Details
*   **Name**: Down Syndrome Risk Screening Ensemble
*   **Version**: 1.0 (Research Prototype)
*   **Model Architecture**: Ensemble of EfficientNet-B0 and ResNet-18
*   **Input**: Aligned Facial Images (RGB)
*   **Output**: Risk Probability Score (0-1) + Grad-CAM Heatmap

## Intended Use
*   **Primary Use Case**: Research into model interpretability and behavior in medical-adjacent visual tasks.
*   **Secondary Use Case**: Educational demonstration of bias and uncertainty in AI systems.
*   **Target Audience**: AI Researchers, Ethicists, Computer Vision students.

## Out-of-Scope Use Cases
> [!WARNING]
> **STRICTLY PROHIBITED:**
> *   Clinical diagnosis or medical screening.
> *   Prenatal screening or decision making.
> *   Any commercial deployment or productization.
> *   Use without human-in-the-loop oversight.

## Training Data
*   **Source**: Publicly available datasets (curated and anonymized).
*   **Preprocessing**: MTCNN for face detection and alignment. Standard ImageNet normalization.
*   **Limitations**: The dataset may contain biases related to ethnicity, age, and image quality. It is not a representative clinical sample.

## Performance & Evaluation
*   **Test Set**: Held-out partition of the collected dataset.
*   **Metrics**: AUC (0.9667), Accuracy (89.89%).
*   **Caveats**: High performance on a curated dataset does not translate to real-world clinical reliability. Simplistic background or lighting correlations may inflate scores.

## Ethical Considerations
*   **False Positives**: Could cause unnecessary anxiety.
*   **False Negatives**: Could provide false reassurance.
*   **Bias**: The model may underperform on underrepresented demographic groups.
*   **Interpretability**: Grad-CAM heatmaps are provided to foster trust but must be understood as model attention, not medical explanation.

## BibTeX
```bibtex
@misc{down_syndrome_screening_2024,
  title={Down Syndrome Facial Risk Screening: A Research Prototype},
  author={A JASWANTH},
  year={2024},
  publisher={HuggingFace Spaces}
}
```
