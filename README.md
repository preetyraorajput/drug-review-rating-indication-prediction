Predicting Patient Ratings and Drug Indications from Reviews Using Deep Learning

A deep learning project that reads real patient drug reviews (just the text) and predicts two things: the rating the patient gave (1–10) and the condition (indication) they were being treated for, out of 21 common categories — using an LSTM neural network, benchmarked honestly against traditional machine learning baselines.

Try it live: drug-review-rating-indication-prediction-6vsfnvk8as7bidwmv6rah.streamlit.app — no install needed, just open the link.

Dataset

44,161 real patient reviews collected from drugs.com across 40 drugs, parsed from saved HTML pages. Four brand/generic pairs were merged into a single drug identity (Ambien→Zolpidem, Paxil→Paroxetine, Wegovy→Semaglutide, Zepbound→Mounjaro), leaving 36 unique drug identities. Two real data-quality bugs were found and fixed during the build, not assumed away:

Drug-name-as-indication bug: when a patient didn't select a condition, drugs.com had bolded the drug's own name instead — the parser had mistakenly captured this as the "condition." Found via direct HTML inspection and fixed (153 of 44,161 rows affected).
Indication label duplication: near-duplicate condition labels (e.g. "Depression" and "Major Depressive Disorder") were merged into a documented 17-entry standardization map, while genuinely distinct conditions (Social Anxiety Disorder, Panic Disorder, OCD, PTSD) were kept separate.

Indications were bucketed into the top 20 most frequent conditions plus "Other" (21 classes). Train/test split: 35,328 / 8,833 (80/20), stratified by indication.

Models

Two Keras models share the same backbone, differing only in the output layer:

Rating model (regression, 1–10): Embedding(20000, 128, mask_zero=True) → Bidirectional(LSTM(64)) → Dropout(0.3) → Dense(32, relu) → Dense(1), MSE loss, MAE metric.
Indication model (21-class classification): same backbone → Dense(21, softmax), sparse categorical cross-entropy, class_weight='balanced' for the imbalanced classes.

Both use EarlyStopping(patience=3, restore_best_weights=True) with a validation split carved from training data only — never the test set — so model-selection decisions never leak test information.

Building fair baselines first

Before trusting any deep learning result, simple TfidfVectorizer + traditional ML baselines were built on the same cleaned text, using the same train/test split:

Task	Baseline model	Result
Rating	Ridge Regression (regularized)	Test MAE 1.91, Test R² 0.4695
Indication	Logistic Regression	Test Accuracy 67.59%, Macro F1 0.67

(A plain, unregularized Linear Regression baseline was tried first for rating and badly overfit — Test R² of -0.16, worse than guessing the average — which is why a regularized Ridge version replaced it for a fair comparison.)

Headline results — rating task
Model	Test MAE	Test R²
Ridge Regression + TF-IDF (baseline)	1.9113	0.4695
LSTM — final model	1.7312	0.4960

The LSTM beat the baseline by a real but modest margin. Pretrained embeddings made essentially no difference on this task (see below for why).

Headline results — indication task (the more interesting story)
Model	Test Accuracy	Macro F1
LSTM, random embeddings	59.16%	0.59
Logistic Regression + TF-IDF (baseline)	67.59%	0.67
LSTM, fine-tuned pretrained embeddings	65.53% (avg of 3 runs)	0.64
LSTM, fine-tuned embeddings + drug-identity feature — final model	72.23%	0.73

The honest part: the simple Logistic Regression baseline initially beat the deep learning model outright (67.59% vs 59.16%). Rather than hide that, it was investigated — a confusion matrix confirmed the LSTM's errors made clinical sense (Anxiety ↔ Depression, Diabetes ↔ Weight Loss — genuinely overlapping conditions), and a controlled embedding ablation showed pretrained GloVe embeddings, when fine-tuned (not frozen), closed most of the gap by giving the model a head start on word meaning it didn't have enough per-class data to learn from scratch.

The deciding move was adding a second, structurally different input: the drug's identity itself, fed into the model alongside the review text via the Keras functional API. Knowing which drug a review is about is a huge clue to the likely condition — something a bag-of-words baseline has no way to use. That pushed accuracy to 72.23%, clearly ahead of the baseline, and confirmed statistically significant.

Statistical validation

McNemar's test (paired significance test, same test set) confirmed the baseline's early edge over the plain fine-tuned-embeddings LSTM was real, not noise (p = 5.5×10⁻⁵). The fine-tuned-embeddings model was independently retrained 3 times to confirm its accuracy was stable (65.84%, 65.01%, 65.74% — under 1 point of spread). McNemar's test again confirmed the final drug-feature model's win over the baseline was overwhelmingly real (p ≈ 3.7×10⁻²²).

Interpretability

A leave-one-word-out test on the rating model — removing each word from a review and re-predicting — confirmed the model responds to genuinely meaningful language (words like "godsend" push predicted ratings up, "nightmare" pulls them down). A confusion matrix on the indication model showed its mistakes cluster around conditions that genuinely share symptoms or drugs (Anxiety/Depression, Diabetes/Weight Loss), not random noise.

Honest framing

This project deliberately avoids the claim "deep learning always wins." For indication classification, a simple, well-regularized linear baseline beat the LSTM outright at first — a real, statistically confirmed result. Deep learning's eventual win was conditional: it needed both transfer learning (fine-tuned pretrained embeddings) and access to information a bag-of-words model structurally cannot use (drug identity) to pull ahead. For the rating task, the LSTM's edge over a regularized baseline is real but modest, and pretrained embeddings didn't help there at all, because that task had enough data to learn good word representations from scratch, unlike the 21-way-split classification task.

Repository contents
File	What it is
Predicting Patient Ratings and Indications from Drug Reviews Using Deep Learning_ new.ipynb	Full notebook: data cleaning, preprocessing, both LSTM models, TF-IDF baselines, embedding ablation, drug-feature model, statistical tests, interpretability
combined_drug_reviews_Test.csv	The cleaned, combined dataset (44,161 reviews) used to train and evaluate all models
Capstone 2_Project_ Summary.pptx	Short summary presentation

Note: the raw scraped HTML pages (per-drug review pages from drugs.com) are not included in this repo — they're large and superseded by the cleaned, combined CSV above.

Disclaimer

This is an educational/illustrative project built on publicly visible drugs.com review text, not a validated clinical or diagnostic tool.
