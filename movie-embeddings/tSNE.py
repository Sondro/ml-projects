import cPickle as pickle
import string
import re
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import imdb


class TSNEViewer(object):

    def __init__(self, model, perplexity=30, learning_rate=1000, n_iter=1000):
        self.tsne = TSNE(
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter
        )
        self.model = model

    def get_imdb_titles(self, movies, from_scratch=False):
        if os.path.isfile('netflix_to_imdb.pkl') and not from_scratch:
            return pickle.load(open('netflix_to_imdb.pkl', 'r'))

        netflix_to_imdb, movie_set = {}, []
        with open('/Users/seanvasquez/Downloads/movies.list', 'r') as f:
            for line in f:
                split = re.split('\t+', line)
                imdb = split[0].strip()
                imdb = re.sub('\([0-9\?][0-9\?][0-9\?][0-9\?]\)', '', imdb)
                movie_set.append(imdb.strip())
        movie_set = set(movie_set)

        for movie in movies:
            if movie in movie_set:
                closest = movie
            else:
                print movie
                closest = self.get_closest_movie(movie, movie_set)
                print closest
                print
            netflix_to_imdb[movie] = closest

        pickle.dump(netflix_to_imdb, open('netflix_to_imdb.pkl', 'w'))
        return netflix_to_imdb

    def query_imdb(self, movies, netflx_to_imdb, category, from_scratch=False):
        cache_file = category + '.pkl'
        if os.path.isfile(cache_file) and not from_scratch:
            return pickle.load(open(cache_file, 'r'))

        ia = imdb.IMDb()
        return_movies, return_info = [], []
        for i, movie in enumerate(movies):
            imdb_name = netflx_to_imdb[movie]
            results = ia.search_movie(imdb_name)
            for j, result in enumerate(results):
                name = result['long imdb title']
                name = re.sub('\([0-9\?][0-9\?][0-9\?][0-9\?]\)', '', name)
                if name.strip() == imdb_name:
                    movie_obj = ia.get_movie(result.movieID)
                    try:
                        info = movie_obj[category]
                        if info:
                            if category == 'mpaa':
                                info = str(info.split()[1])
                            elif category == 'rating':
                                info = float(info)
                            elif category == 'year':
                                info = int(info)
                            elif category == 'genres':
                                info = list(info)
                            return_movies.append(movie)
                            return_info.append(info)
                        break
                    except KeyError:
                        continue

        to_return = (return_movies, return_info)
        pickle.dump(to_return, open(cache_file, 'w'))
        return to_return

    def get_closest_movie(self, movie, all_movies):
        closest_movie, smallest_distance = None, 0

        movie = re.sub('\(.*\)', '', movie).strip()
        movie = movie.translate(None, string.punctuation)
        movie = movie.replace('Special Edition', '')
        movie = movie.replace('Extended Edition', '')
        movie = movie.replace('Anniversary Edition', '')
        movie = movie.replace("Collectors' Edition", '')
        movie = movie.replace('Anniversary', '')
        movie = movie.replace("Collector's Edition", '')
        movie = movie.replace("Platinum Edition", '')
        movie = movie.replace("Extreme Edition", '')
        movie = movie.replace("Edition", '')
        movie = re.sub('Season [0-9]', '', movie)
        movie = re.sub('Part [0-9]', '', movie)
        movie = movie.replace("Director's Cut", '')
        movie = movie.replace("The Movie", '')

        for m in all_movies:
            m2 = m.translate(None, string.punctuation)
            set1 = set(m2.split())
            set2 = set(movie.split())
            dist = len(set1 & set2) / float(len(set1 | set2))
            if dist > smallest_distance:
                smallest_distance = dist
                closest_movie = m
        return closest_movie

    def trim_range(self, range_str):
        start, end = range_str[1:-1].split(', ')
        start = str(int(round(float(start.strip()))))
        end = str(int(round(float(end.strip()))))
        new_str = '{} - {}'.format(start, end)
        return new_str

    def trim_year_range(self, range_str):
        start, end = range_str[1:-1].split(', ')
        start = str(int(round(float(start.strip())))).ljust(4, '0')
        end = str(int(round(float(end.strip())))).ljust(4, '0')
        new_str = '{} - {}'.format(start, end)
        return new_str

    def bucket_genres(self, genres):
        new = set()
        for i in genres:
            if i in {'Action', 'Thriller', 'Drama', 'Horror'}:
                new.add('Action')
            if i in {'Comedy', 'Family', 'Romance'}:
                new.add('Comedy')
            if i in {'Sci-Fi', 'Fantasy'}:
                new.add('Sci-Fi')
            if len(new) == 0:
                new.add('Other')
        return '/'.join(sorted(list(new)))

    def simple_plot(self, movies, show):
        plt.style.use('ggplot')
        fig, ax = plt.subplots()

        embeddings = self.tsne.fit_transform(self.model[movies])
        x, y = embeddings[:, 0], embeddings[:, 1]
        ax.scatter(x, y, alpha=.3, s=50)

        seen = set()
        x_seen = set()
        for i, txt in enumerate(movies):
            if len(txt) > 20:
                continue
            if int(round(y[i])) in x_seen:
                continue
            if (round(x[i], -1), round(y[i], -1)) not in seen:
                ax.annotate(txt, (x[i], y[i]), weight='bold', size=12)
                seen.add((round(x[i], -1), round(y[i], -1)))
                x_seen.add(int(round(y[i])))
                x_seen.add(int(round(y[i])) - 1)
                x_seen.add(int(round(y[i])) + 1)
                x_seen.add(int(round(y[i])) - 2)
                x_seen.add(int(round(y[i])) + 2)
        ax.set(yticklabels=[])
        ax.set(xticklabels=[])
        plt.show() if show else None
        return

    def plot(self, category=None, num_quantiles=None, show=True):
        movie_dist = pickle.load(open('SGNS/cache/movie_dist.pkl', 'r'))
        movie_index = pickle.load(open('SGNS/cache/movie_index.pkl', 'r'))
        movies = map(movie_index.get, np.argsort(movie_dist)[-500:][::-1])

        if category is None:
            self.simple_plot(movies, show)
            return
        else:
            if category not in ['mpaa', 'rating', 'year', 'genres']:
                assert False, \
                    "category must be 'mpaa', 'rating', 'year', or 'genres'"

        netflix_to_imdb = self.get_imdb_titles(movies)
        movies, categories = self.query_imdb(movies, netflix_to_imdb, category)

        embeddings = self.tsne.fit_transform(self.model[movies])
        x, y = embeddings[:, 0], embeddings[:, 1]
        df = pd.DataFrame({'x': x, 'y': y, category: categories})

        kwargs = {}
        if category == 'mpaa':
            df[' '] = df[category]
        if category == 'rating':
            df[' '] = pd.qcut(df[category], q=5).map(self.trim_range)
        if category == 'year':
            noise = np.random.normal(scale=.001, size=len(df))
            df[category] = df[category] + noise
            df[' '] = pd.qcut(df[category], q=16).map(self.trim_year_range)
        if category == 'genres':
            df[' '] = df[category].map(self.bucket_genres)
            kwargs.update({
                'hue_order': [
                    'Action', 'Action/Sci-Fi', 'Action/Comedy',
                    'Action/Comedy/Sci-Fi', 'Sci-Fi', 'Comedy/Sci-Fi',
                    'Comedy', 'Other'
                ]
            })

        ax = sns.lmplot('x', 'y', data=df, hue=' ', fit_reg=False,
                        scatter_kws={"s": 100}, **kwargs)
        ax.set(yticklabels=[])
        ax.set(xticklabels=[])
        plt.xlabel('')
        plt.ylabel('')
        plt.show() if show else None
