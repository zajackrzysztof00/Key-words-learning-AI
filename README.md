# Key-words-learning-AI
Simple neural network script, with deep learning AI. Find key words from context.

For propere working of ai make sure that you use as input data json files with data prepared as in example below:

text == List of text with context from witch look for key words
label == One word prediction

data {
  'text': ["Antique collecting preserves history and tells unique stories.", ... ],
  'label': ['History', ...]
}

Indexes of key words and text have to match each other for model to learn correctly.
