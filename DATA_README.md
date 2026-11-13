# College Intelligence SLM — Dataset Splits

These files contain the question-answer examples generated from the original college dataset using the same QA-generation and institution-group split procedure in the project notebook.

## Files and counts
- `train_data.csv`: 5,786 training examples
- `validation_data.csv`: 416 validation examples
- `test_data.csv`: 1,080 held-out test examples

Total QA examples: 7,282 from 999 source records and 103 institution groups.

## Columns
- `question`: question template
- `input_text`: instruction plus context supplied to the model
- `target_text`: expected answer
- `group_id`: institution grouping key used to keep institutions in one split
- `row_index`: source-row index used by the notebook split

## Important
These are QA training/evaluation examples, not the original raw college dataset. The split reproduces the notebook's `GroupShuffleSplit` procedure with `random_state=42`. Keep the test file held out from training.
