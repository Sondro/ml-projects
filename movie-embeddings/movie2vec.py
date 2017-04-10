import numpy as np


class Movie2Vec(object):

    def __init__(self, embedding_filename):
        self.idx, self.matrix = self._parse_embedding_file(embedding_filename)
        self.idx_inv = dict(map(reversed, self.idx.items()))

    def most_similar(self, movies, top_n=10):
        sims = self[movies].dot(self.matrix.T)
        n_closest_indices = np.argsort(sims)[-1: -top_n - 1: -1]
        n_closest_movies = self._decode_indices(n_closest_indices)
        return zip(n_closest_movies, sims[n_closest_indices])

    def similarity(self, movie1, movie2):
        return self[movie1].dot(self[movie2])

    def complete_analogy(self, movie1, movie2, movie3, top_n=1):
        predicted = self[movie3] + self[movie2] - self[movie1]
        nearest = np.argsort(predicted.dot(self.matrix.T))
        guesses = self._decode_indices(nearest)
        guesses = [i for i in guesses if i not in {movie1, movie2, movie3}]
        return list(reversed(guesses))[:top_n]

    def _decode_indices(self, indices):
        return map(self.idx_inv.get, indices)

    def _parse_embedding_file(self, embedding_filename):
        idx = {}
        matrix = []
        for i, line in enumerate(open(embedding_filename, 'r')):
            items = line.split()
            movie = ' '.join(items[0].split('_'))
            idx[movie] = i
            matrix.append(map(float, items[1:]))
        return idx, np.array(matrix)

    def _validate_key(self, key):
        if key not in self.idx:
            raise Exception('Invalid key: {}'.format(key))

    def __contains__(self, key):
        return key in self.idx

    def __getitem__(self, key):
        if isinstance(key, str):
            self._validate_key(key)
            return self.matrix[self.idx[key]]

        elif isinstance(key, list):
            [self._validate_key(k) for k in key]
            indices = map(self.idx.get, key)
            return self.matrix[indices]
