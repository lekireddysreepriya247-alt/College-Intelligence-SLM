# College Intelligence SLM

A small language-model project for answering questions about college and program information. The accompanying Google Colab notebook fine-tunes `google/flan-t5-small` on question-answer examples prepared from a college-program JSON dataset, evaluates it on a held-out test split, and builds a semantic-retrieval-assisted question-answering function.

## Repository contents

- `College_Data_Intelligence_SLM_HuggingFace.ipynb` — primary, step-by-step Google Colab notebook.
- `college_intelligence_slm.py` — exported code cells from the notebook for source-code browsing. The notebook is the recommended way to run the workflow.
- `requirements.txt` — Python dependencies used by the notebook.
- `College_Intelligence_SLM_Project_Report.pdf` — project report (if included in this repository).

## Run the project

1. Open the `.ipynb` notebook in Google Colab.
2. Select a GPU runtime if available (`Runtime` → `Change runtime type` → `T4 GPU`, when offered).
3. Run the setup cell to install the dependencies.
4. Upload the college-program JSON dataset when prompted. The notebook expects a JSON list of records and uses the fields available in that dataset.
5. Run the notebook cells in order: data inspection and preparation, question-answer generation, institution-grouped train/validation/test split, tokenization, FLAN-T5-small fine-tuning, held-out evaluation, model saving, semantic retrieval, and chatbot testing.
6. The Hugging Face Hub publishing section is optional. Use a Hugging Face write token stored securely in Colab Secrets; never paste a token into a notebook or commit it to GitHub.

## Dataset and model files

The dataset is not bundled here. Upload the dataset separately in Colab, and only publish it to GitHub if you have permission to share it and it contains no private or restricted information. Trained model weights are large; keep them in Hugging Face Hub or another model-artifact store rather than committing them to this source repository.

## Evaluation

Run the notebook to generate the actual evaluation metrics for your dataset and training run. Report the values produced by your own run; do not treat example screenshots or placeholder values as guaranteed results.

## Security

- Do not commit Hugging Face access tokens, API keys, credentials, or Colab secret values.
- Do not commit private datasets or personally identifiable information.
- Keep large model checkpoints out of Git history.

## License

Add the license required by your course or project supervisor before distributing this repository.
