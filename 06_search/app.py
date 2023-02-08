from search import SearchEngine
from files import get_article, load
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

doc_ids, terms, svd = load(['doc_ids', 'terms', 'svd'])
print("Loaded files.")
se = SearchEngine(doc_ids, terms, svd)
print("Loaded search engine.")


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    query = request.form["query"]
    try:
        search_results = se.search(query)
        for result in search_results:
            result.update(get_article(result['id']))
    except Exception as e:
        search_results = []
    return render_template("results.html", search_results=search_results)


@app.route("/article", methods=["GET", "POST"])
def article():
    id = request.args.to_dict()["id"]
    article = get_article(id)
    return render_template("article.html",
                           title=article['title'],
                           text=article['text'])


if __name__ == "__main__":
    app.run()
