'''
Use MLE and MAP to estimate the parameters of a multinomial distribution on the 20 
Newsgroups dataset. Explore the impact of different priors on the estimation. 
Tasks: 
● Load and preprocess the dataset. 
● Implement MLE for multinomial distribution parameter estimation. 
● Apply MAP estimation with various priors (e.g., Dirichlet priors). 
● Compare results and evaluate the effect of different priors. 

'''

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

# Load 20 Newsgroups dataset
data = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes")
)

# Convert text into word counts
vectorizer = CountVectorizer(
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(data.data)

y = data.target

words = vectorizer.get_feature_names_out()

print("Number of documents:", X.shape[0])
print("Number of words:", X.shape[1])


# Number of classes
n_classes = len(np.unique(y))
n_features = X.shape[1]


# Try different Dirichlet priors
alphas = [0.1, 1.0, 10.0]

for alpha in alphas:

    print("\nDirichlet Alpha:", alpha)

    for c in range(n_classes):

        # Select documents belonging to class c
        X_class = X[y == c]

        # Total count of each word in this class
        word_counts = np.asarray(
            X_class.sum(axis=0)
        ).flatten()

        total_words = word_counts.sum()

        # ---------------- MLE ----------------
        mle_prob = word_counts / total_words

        # ---------------- MAP ----------------
        # Dirichlet prior
        map_prob = (
            word_counts + alpha
        ) / (
            total_words + alpha * n_features
        )

        # Display first few probabilities
        if c == 0:
            print("\nFirst 5 words:")
            for i in range(5):
                print(
                    words[i],
                    "MLE =", mle_prob[i],
                    "MAP =", map_prob[i]
                )

            break