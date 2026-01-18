from flask import Flask, render_template, request, redirect, url_for, flash
from recommender import Recommender
from tf_model import NMFRecommender
import os

app = Flask(__name__)
app.secret_key = 'dev-secret'

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MOVIES = os.path.join(DATA_DIR, 'movies.csv')
RATINGS = os.path.join(DATA_DIR, 'ratings.csv')

# Initialize recommenders
classic_rec = Recommender(MOVIES, RATINGS)
try:
    nmf_rec = NMFRecommender(MOVIES, RATINGS)
except Exception:
    nmf_rec = None


@app.route('/')
def index():
    users = sorted(list(classic_rec.user_item.index))
    movies = classic_rec.movies[['movieId', 'title']] \
        .sort_values('title') \
        .to_dict(orient='records')
    return render_template('index.html', users=users, movies=movies)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_id = int(request.form.get('user_id'))
        topn = int(request.form.get('topn', 8))
        engine = request.form.get('engine', 'classic')

        if engine == 'nmf' and nmf_rec is not None:
            results = nmf_rec.recommend_for_user(user_id, topn)
            title = f'Recommended for you'
        else:
            results = classic_rec.recommend_for_user(user_id, topn)
            title = f'Recommended for you'

        items = results.to_dict(orient='records')

        return render_template(
            'recommend.html',
            title=title,
            items=items,
            hero_image=url_for('static', filename='posters/hero.jpg')
        )

    except Exception as e:
        flash(str(e))
        return redirect(url_for('index'))


@app.route('/similar', methods=['POST'])
def similar():
    try:
        movie_id = int(request.form.get('movie_id'))
        n = int(request.form.get('topn', 6))

        results = classic_rec.similar_items(movie_id, n)

        return render_template(
            'recommend.html',
            title='Similar Movies',
            items=results.to_dict(orient='records'),
            hero_image=url_for('static', filename='posters/hero.jpg')
        )

    except Exception as e:
        flash(str(e))
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
