from icrawler.builtin import BingImageCrawler
import os

keywords = [
    "tools",
    "kitchen utensils",
    "office desk objects",
    "metal objects",
    "bag contents",
    "scissors",
    "pliers",
    "screwdriver",
    "toy water gun",
    "plastic toy gun"
]

BASE_DIR = "negative_dataset"

for keyword in keywords:
    folder_name = keyword.replace(" ", "_")
    save_dir = os.path.join(BASE_DIR, folder_name)

    crawler = BingImageCrawler(
        storage={"root_dir": save_dir}
    )

    crawler.crawl(
        keyword=keyword,
        max_num=200
    )