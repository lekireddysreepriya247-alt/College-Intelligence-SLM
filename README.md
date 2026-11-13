
# College Intelligence SLM
### Small Language Model for College Information Question Answering

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Model](https://img.shields.io/badge/Model-FLAN--T5--Small-orange)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)
![Platform](https://img.shields.io/badge/Platform-HuggingFace-yellow)

## 1. Introduction

College Intelligence SLM is an AI-powered
question-answering system developed to provide
college-related information using a fine-tuned
Small Language Model.

The project uses FLAN-T5-small to generate
context-based answers from educational data.

## 2. Objectives

- Develop a Small Language Model.
- Prepare and clean educational datasets.
- Generate question-answer training examples.
- Fine-tune a pretrained language model.
- Evaluate model performance.
- Develop an interactive chatbot.
- Publish the trained model on Hugging Face.

## 3. Technologies Used

- Python
- Google Colab
- PyTorch
- Hugging Face Transformers
- Pandas
- Scikit-learn
- Gradio

## 4. Dataset Information

- Original records: 999
- QA examples: 7,282
- Training: 5,786
- Validation: 416
- Testing: 1,080

## 5. Model Configuration

- Base model: google/flan-t5-small
- Epochs: 3
- Batch size: 4
- Learning rate: 0.00005
- GPU: NVIDIA Tesla T4

## 6. Evaluation Results

- Exact Match Accuracy: 100%
- Average Token F1: 100%

These results represent the prepared held-out
test dataset and should not be interpreted as
general-purpose accuracy.

## 7. Project Structure

- notebooks/
- reports/
- screenshots/
- dataset/
- README.md
- requirements.txt
- .gitignore

## 8. Hugging Face Model

https://huggingface.co/sree02200/college-intelligence-slm

## 9. Features

- College information question answering
- Context-based answer generation
- Fine-tuned SLM
- Interactive chatbot
- Hugging Face deployment

## 10. Future Enhancements

- Improve information retrieval.
- Add more educational datasets.
- Handle ambiguous questions.
- Improve answer accuracy on unseen questions.
- Deploy the chatbot as a web application.

## 11. Conclusion

This project demonstrates how Small Language
Models can be used to develop educational
question-answering systems.

It combines dataset preparation, model
fine-tuning, evaluation and deployment.

## 12. Project Author

CSE AIML Academic Project
