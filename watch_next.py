# A Model that uses NLP's TF-IDF vectorizer to compare a sample description of a film with movie descriptions found in movies.txt.
# Then finds the most similar one to the sample description.

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sample = '''Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth,
the Illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk can 
live in peace. Unfortunately, Hulk lands on the planet Sakaar where he is sold into slavery and trained as a gladiator.'''

def find_similar_movie(sample):
    # Read in the movies.txt file
    with open('movies.txt', 'r') as file:
        movies = file.readlines()
    
    # The model utilises TF-IDF vectorizer to transform text into vectors
    vectorizer = TfidfVectorizer()

    # Transform movie descrpitions into vectors
    movie_vectors = vectorizer.fit_transform(movies)
  
    # Transform the sample description into a vectors
    sample_vectors = vectorizer.transform([sample])

    # Calculates similarities between the sample description and all the movies descriptions
    cosine_similarities = np.dot(movie_vectors, sample_vectors.T)

    # Find the index of the movie which has the highest cosine simlarities 
    index = cosine_similarities.argmax()

    # Return the title of the most similar movie
    return movies[index].strip()

similar_movie = find_similar_movie(sample)
print(similar_movie)

