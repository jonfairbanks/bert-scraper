import os
import sys
import argparse
import requests
from bs4 import BeautifulSoup
from bertopic import BERTopic
import pandas as pd
import plotly.express as px
from umap import UMAP

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    content = []
    for paragraph in soup.find_all("p"):
        text = paragraph.get_text(strip=True)
        if len(text) > 0 and isinstance(text, str):
            content.append(text)
    return content


def custom_visualize_topics(
    topic_model, topics, documents, n_neighbors=15, max_hover_text_length=350, **kwargs
):
    topic_model.umap_model.n_neighbors = n_neighbors
    embeddings = topic_model._extract_embeddings(documents)
    umap_embeddings = UMAP(
        n_neighbors=n_neighbors, n_components=2, metric="cosine", random_state=42
    ).fit_transform(embeddings)

    viz_df = pd.DataFrame(umap_embeddings, columns=["x", "y"])
    viz_df["Topic"] = topics
    viz_df["Document"] = documents

    viz_df["TruncatedDocument"] = viz_df["Document"].apply(
        lambda x: x[:max_hover_text_length] + "..."
        if len(x) > max_hover_text_length
        else x
    )

    fig = px.scatter(
        viz_df,
        x="x",
        y="y",
        color="Topic",
        hover_name="TruncatedDocument",
        color_continuous_scale="viridis",
    )

    fig.update_layout(
        {
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
            "margin": {
                "r": 0,
                "t": 0,
                "l": 0,
                "b": 0,
                "pad": 0,
            },
        }
    )

    fig.show()


def main(url):
    print(f"Scraping {url}...")
    scraped_content = scrape_website(url)

    if len(scraped_content) < 2:
        print("Not enough content for topic modeling.")
        return

    umap_model = UMAP(n_neighbors=5, min_dist=0.1, n_components=2, metric="cosine")
    model = BERTopic(
        language="english",
        umap_model=umap_model,
        min_topic_size=2,
    )

    topics, _ = model.fit_transform(scraped_content)

    # topic_words = model.get_topic_freq()
    # print("Topic Words\n")
    # print(topic_words)

    # N = 15
    # topic_representation = model.get_topic_info()
    # print("\nTop Tokens for Each Topic:")
    # for index, row in topic_representation.iterrows():
    #     if index < N:
    #         print(f"Topic {row['Topic']}: {row['Name']}")

    custom_visualize_topics(model, topics, scraped_content)

    # new_document = ["This is a new document"]
    # new_topic, _ = model.transform(new_document)
    # print("\nNew Document Topic:")
    # print(new_topic)

    # Print details for all topics
    all_topics = model.get_topic_freq()["Topic"].unique()

    print("Topic Details:")
    for topic in all_topics:
        if topic != -1:
            topic_words = model.get_topic(topic)
            print(f"\nTopic {topic}:")
            for word, probability in topic_words:
                print(f"{word} ({probability:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "url", help="URL of the webpage to be scraped for topic modeling."
    )
    args = parser.parse_args()
    main(args.url)
