  <h1>Text Classification Project</h1>
  <p>This project consists of two main scripts:</p>
  <ul>
    <li><strong>Labeling Script:</strong> Uses a pre-trained BERT model to label unlabeled text data with emotion categories.</li>
    <li><strong>Multi-Class Classification Model:</strong> Trains a custom text classification model on labeled data using deep learning with GloVe embeddings and a Bi-LSTM architecture.</li>
  </ul>

  <div class="section">
    <h2>1. Labeling Script</h2>
    <p>This script loads an unlabeled dataset, uses a pre-trained BERT model for emotion classification, and saves the labeled data.</p>
    <h3>Requirements</h3>
    <ul>
      <li><code>pandas</code></li>
      <li><code>torch</code></li>
      <li><code>transformers</code></li>
    </ul>
    <h3>Instructions</h3>
    <ol>
      <li><strong>Set Up Paths:</strong>
        <p>Define the path to your input (<code>unlabeled.csv</code>) and output files (<code>labeled3.csv</code>) in the script.</p>
      </li>
      <li><strong>Load Pre-trained Model:</strong>
        <p>The script uses the <code>nateraw/bert-base-uncased-emotion</code> model for classification, which is downloaded from Hugging Face.</p>
      </li>
      <li><strong>Run the Script:</strong>
        <pre><code>code_for_labelling.py</code></pre>
        <p>This will save the labeled data to the specified CSV file.</p>
      </li>
    </ol>
    <h3>Code Summary</h3>
    <p>The labeling script uses a pre-trained BERT model from Hugging Face to classify each text entry into one of four emotion categories: <code>anger</code>, <code>joy</code>, <code>sadness</code>, or <code>neutral</code>. Here’s a detailed breakdown:</p>
    <ul>
      <li><strong>Text Processing:</strong> Each text entry is tokenized using BERT’s tokenizer, which converts words into tokens compatible with the BERT model.</li>
      <li><strong>Model Prediction:</strong> The BERT model outputs logits, which are converted into probabilities using softmax. The category with the highest probability is selected as the predicted emotion.</li>
      <li><strong>Filtering:</strong> Rows with emotions outside the desired categories are removed. Only the categories <code>anger</code>, <code>joy</code>, <code>sadness</code>, and <code>neutral</code> are retained, ensuring a focused dataset for multi-class training.</li>
      <li><strong>Output:</strong> The resulting labeled dataset is saved as a new CSV file, which can then be used for training a custom multi-class classifier.</li>
    </ul>
  </div>

  <div class="section">
    <h2>2. Multi-Class Classification Model</h2>
    <p>This script trains a multi-class text classification model using the labeled data (<code>labeled3.csv</code>) with TensorFlow and GloVe embeddings.</p>
    <h3>Requirements</h3>
    <ul>
      <li><code>numpy</code></li>
      <li><code>pandas</code></li>
      <li><code>tensorflow</code></li>
      <li><code>sklearn</code></li>
    </ul>
    <h3>Instructions</h3>
    <ol>
      <li><strong>Load Labeled Data:</strong>
        <p>Ensure <code>labeled3.csv</code> is in the same directory or adjust the path in the script.</p>
      </li>
      <li><strong>Preprocess Data:</strong>
        <p>Text is cleaned, tokenized, and converted into sequences. Labels are one-hot encoded for multi-class classification.</p>
      </li>
      <li><strong>Load GloVe Embeddings:</strong>
        <p>Ensure <code>glove.6B.300d.txt</code> (GloVe embeddings) is in the project directory. The script will create an embedding matrix for the model.</p>
      </li>
      <li><strong>Run the Script:</strong>
        <pre><code>multi_text_class.py</code></pre>
        <p>This will train the model and print out a classification report on the test data.</p>
      </li>
    </ol>
    <h3>Code Summary</h3>
    <p><strong>Model Architecture:</strong> This custom classification model uses a Bi-LSTM (Bidirectional Long Short-Term Memory) neural network architecture with dropout layers for improved generalization. The primary components include:</p>
    <ul>
      <li><strong>Embedding Layer:</strong> GloVe embeddings are loaded to provide word vectors for each token, giving the model rich word representations that capture semantic meanings.</li>
      <li><strong>Spatial Dropout:</strong> Applied to prevent overfitting by randomly setting a fraction of input units to zero during training, which forces the model to rely on a wider variety of features.</li>
      <li><strong>Bi-LSTM Layer:</strong> The bidirectional LSTM processes the text sequences in both forward and backward directions, enhancing the model’s ability to learn contextual dependencies in the text.</li>
      <li><strong>Global Max Pooling:</strong> Applied to the LSTM output to capture the most important features across the sequence for each filter, reducing the dimensionality and improving computational efficiency.</li>
      <li><strong>Fully Connected Layers:</strong> A dense layer with dropout regularization to further reduce overfitting. The final output layer uses a softmax activation for multi-class classification across four emotion categories.</li>
    </ul>
    <p><strong>Evaluation:</strong> The model is trained and evaluated on labeled text data. After training, a classification report is generated using the test set, providing precision, recall, and F1-scores for each emotion category. The model is also tested with an example sentence to validate its predictions in real-world scenarios.</p>
  </div>

</body>
</html>
