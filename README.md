# Key-words-learning-AI
A simple neural network script leveraging deep learning to identify key words from contextual text.

Input Data Requirements

To ensure proper functionality, the input data should be provided as JSON files structured as demonstrated in the example below:

text: A list of sentences or paragraphs that provide context for the key words to be identified.

label: A list of single-word predictions (key words) corresponding to each context in text.

Example Data Format:

data {
  "text": ["Antique collecting preserves history and tells unique stories.", ...],
  "label": ["History", ...]
}

Key Notes:

Index Matching: Ensure that the indexes of the key words in the label list correspond directly to their respective contextual text in the text list.

This alignment is critical for the model to learn and predict accurately.
