from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash
)
import os

from ml import get_image_by_url, predict_comics


app = Flask(__name__)
app.config["SECRET_KEY"] = "super-secret-fast-ai"
BASE_DIR = os.path.abspath(os.path.dirname(__name__))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/comics", methods=["GET", "POST"])
def comics():
    if request.method == "POST" and request.form.get('image_url'):
        try:
            # Example: https://krot.info/uploads/posts/2020-01/1579518770_29-78.jpg
            url = request.form.get('image_url') # by name
            img = get_image_by_url(url)
            # img.save(os.path.join(BASE_DIR, 'static', 'img', 'img.jpg'))
            class_name, yhat_probas = predict_comics(img)

            return render_template("comics.html", class_name=class_name, yhat_probas=yhat_probas, url=url)

        except:
            flash("ERROR: Can't process this url or image. Please, try another one.", category='danger')

    return render_template("comics.html", class_name=None, yhat_probas=None, url=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)