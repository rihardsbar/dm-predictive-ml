import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import TruncatedSVD, PCA, NMF, SparsePCA


def create_2d_plot(data_2d, labels):
    plt.figure()
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels)


def create_3d_plot(data_2d, target, labels):
    fig = plt.figure()
    plt.clf()
    ax = Axes3D(fig, elev=15, azim=125)
    plt.cla()
    ax.scatter(data_2d[:, 0], data_2d[:, 1], target, 'o', s=50, c=labels)
    plt.show()


def create_3d_plot(data_3d, labels):
    fig = plt.figure()
    plt.clf()
    ax = Axes3D(fig, elev=15, azim=125)
    plt.cla()
    ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], 'o', s=50, c=labels)
    plt.show()


def create_3d_plot_2(data_2d, target, labels):
    fig = plt.figure()
    plt.clf()
    ax = Axes3D(fig, elev=15, azim=125)
    plt.cla()
    ax.scatter(data_2d[:, 0], data_2d[:, 1], target, 'o', s=50, c=labels)
    plt.show()


def clean_na(ds):
    clean_ds = ds
    # remove the null values, that is fill NaN with there - FIXME: Rihards, naive implementation
    clean_ds = clean_ds.dropna()
    #clean_ds = clean_ds.fillna(value=0, axis=1)
    clean_ds = clean_ds.drop('Unnamed: 0', axis=1)
    return clean_ds


def discretize_gross(row):
    if row['worldwide_gross'] < 1e7:
        return 1
    elif row['worldwide_gross'] < 3e8:
        return 2
    else:
        return 3


def read_data():
    input_file = "../dataset_/no_imdb_names-count_cat-tf_184f.csv"
    ds = pd.read_csv(input_file)
    ds = clean_na(ds)

    ds['gross_class'] = ds.apply(discretize_gross, axis=1)

    return ds


def plot_3d():
    df = read_data()

    # 3D-plot
    data_3d = PCA(n_components=3).fit_transform(df)
    create_3d_plot(data_3d, df['gross_class'])

    # How many movies per gross class?
    grouped = df.groupby('gross_class')
    print("Movies per gross class")

    print(grouped.count())


def plot_3d_2():
    df = read_data()
    target = df['worldwide_gross']
    labels = df['gross_class']

    # 3D-plot
    data_2d = PCA(n_components=2).fit_transform(df)
    create_3d_plot_2(data_2d, target, labels)

    # How many movies per gross class?
    grouped = df.groupby('gross_class')
    print("Movies per gross class")
    print(grouped.count())


if __name__ == "__main__":
    plot_3d()
    plot_3d_2()
