import math
from matplotlib import pyplot as plt
import pandas as pd
import pymysql
import plotly.express as px

DB_CONFIG = dict(
    host='192.168.1.3',
    port=3306,
    user='root',
    password='root',
    db='content_ner',
    charset='utf8mb4',
    autocommit=True
)


def read_data():
    conn = pymysql.connect(**DB_CONFIG)
    with conn.cursor() as cursor:
        sql = "SELECT * FROM tbl_image_caption"
        cursor.execute(sql)
        data = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
    conn.close()
    df = pd.DataFrame(list(data), columns=columns)
    return df


def split_tags(field):
    tags = field.split('|')
    return [tag.strip() for tag in tags]


def count_tags(df):
    tags = []
    for field in df['tags'].unique():
        tags.extend(split_tags(field))
    tag_count = pd.Series(tags, name='count').value_counts()
    return tag_count


def write_to_csv(tag_count):
    tag_count.to_csv('tag_count.csv', index=True)


def read_csv():
    try:
        df = pd.read_csv('tag_count.csv')
    except FileNotFoundError:
        df = None
    return df


def plot_radar(df):
    categories = list(df.index)
    N = len(categories)
    values = list(df['count'])
    values += values[:1]

    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(16, 16), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, fontsize=14)
    ax.tick_params(axis='both', which='major', pad=15)

    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, alpha=0.5)

    plt.title('Tag Distribution', fontsize=22, color='black', y=1.08)
    plt.show()


if __name__ == '__main__':
    df = read_csv()
    if df is None:
        df = read_data()
        tag_count = count_tags(df)
        write_to_csv(tag_count)
    plot_radar(df)
