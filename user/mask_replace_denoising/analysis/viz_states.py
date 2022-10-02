import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import bokeh
import faiss
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
import umap.plot
from bokeh.plotting import show, output_file
from matplotlib import rc
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from yellowbrick.cluster import KElbowVisualizer

sns.set_style("white")
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)

sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')


def flatten(deeplist):
    return np.array([item for sublist in deeplist for item in sublist])


def dim_reduce(points, algo='pca', remove_pcs=0, kwargs={}):
    if algo == 'pca':
        pca = PCA(n_components=2).fit_transform(points)
        c1, c2 = pca[:, 0], pca[:, 1]
    elif algo == 'umap':
        if kwargs.get('metric') == 'cosine':
            points -= points.mean(axis=0)
        mapper = umap.UMAP(**kwargs).fit(points)
        c1, c2 = mapper.embedding_[:, 0], mapper.embedding_[:, 1]
    elif algo == 'tsne':
        if kwargs.get('metric') == 'cosine':
            points -= points.mean(axis=0)
        if kwargs.get('pca', 0) > 0:
            points = PCA(n_components=kwargs["pca"]).fit_transform(points)
        tsne = TSNE(n_components=2, n_jobs=-1).fit_transform(points)
        c1, c2 = tsne[:, 0], tsne[:, 1]
    elif algo == 'tsne2':
        if kwargs.get('metric') == 'cosine':
            points -= points.mean(axis=0)
        if kwargs.get('pca', 0) > 0:
            points = PCA(n_components=kwargs["pca"]).fit_transform(points)
        tsne = TSNE(n_components=2, n_jobs=-1, init='pca',
                    learning_rate=max(500, points.shape[0] / 48)).fit_transform(
            points)
        c1, c2 = tsne[:, 0], tsne[:, 1]
    else:
        raise NotImplemented
    return c1, c2


def _prep_sent_data(data):
    non_special_mask = [[3 < x < len(data['vocab']) - 4 for x in s]
                        for s in data['targets']]
    langid_mask = [[x >= len(data['vocab']) - 3 for x in s]
                   for s in data['targets']]
    special_mask = [[x in [0, 2] for x in s] for s in data['targets']]
    labels = [data['vocab'][s[0]] for s in data['language']]
    inputs = [f"({len(s)}) " +
              "".join([data['vocab'][w] for w in s]).replace("▁", " ")
              for s in data['encoder']]
    targets = [f"({len(s)}) " +
               "".join([data['vocab'][w] for w in s]).replace("▁", " ")
               for s in data['targets']]

    return non_special_mask, langid_mask, special_mask, labels, inputs, targets


def _prep_tok_data(data, top, limit):
    # omit special and low-frequency words
    # IMPORTANT: The label is determined based on the true identity of a token
    # mask = [[3 < x < top for x in sent] for sent in data['targets']]
    mask = [[3 < x < top and len(data['vocab'][x]) > 2 for x in sent]
            for sent in data['targets']]
    # mask = [[3 < x < top for x in sent] for sent in data['encoder']]

    # create labels for sentences and tokens, respectively
    tok_lang = [[data['vocab'][w] for w in s] for s in data['language']]
    tok_type = [["Corrupted" if w else 'Real' for w in s] for s in
                data['noisy']]
    tok_inp = [[data['vocab'][w] for w in s] for s in data['encoder']]
    tok_trg = [[data['vocab'][w] for w in s] for s in data['targets']]

    tok_lang = flatten(tok_lang)[flatten(mask)]
    tok_type = flatten(tok_type)[flatten(mask)]
    tok_inp = flatten(tok_inp)[flatten(mask)]
    tok_trg = flatten(tok_trg)[flatten(mask)]

    lang_mean = {lang: [] for lang in set(tok_lang)}
    for i in tqdm(range(1, 7), desc=f'Language Centroids'):
        # concat all states into a single 2D array and apply the filter mask
        points = np.concatenate(data[f'layer_{i - 1}'], axis=0)[flatten(mask)]
        for lang in set(tok_lang):
            lang_mean[lang].append(points[tok_lang == lang].mean(axis=0))

    # sub-sample N tokens to improve speed
    ntokens = sum(flatten(mask))
    print("Token dataset size: ", ntokens)

    if ntokens > limit:
        np.random.seed(0)
        subsample_mask = np.random.choice(ntokens, limit)
        tok_lang = tok_lang[subsample_mask]
        tok_type = tok_type[subsample_mask]
        tok_inp = tok_inp[subsample_mask]
        tok_trg = tok_trg[subsample_mask]
    else:
        subsample_mask = None

    return mask, subsample_mask, tok_inp, tok_trg, tok_lang, tok_type, lang_mean


def plot_sentences(path, data, sent_data, algo, kwargs):
    name = '_'.join([algo] + [f'{key}={value}' for key, value
                              in kwargs.items()])
    ordinary_mask, langid_mask, special_mask, labels, _, _ = sent_data
    labels = [x.replace("[", "").replace("]", "") for x in labels]

    token_types = {"langid": langid_mask,
                   "bos-eos": special_mask,
                   "ordinary": ordinary_mask}

    points_2d = defaultdict(list)
    for subset, mask in token_types.items():
        for i in tqdm(range(1, 7), desc=f'Reducing dimensionality, {name}'):
            points = np.array([s[m].mean(0) for s, m
                               in zip(data[f'layer_{i - 1}'], mask)])
            points_2d[subset].append(dim_reduce(points, algo, 0, kwargs))

    for i in tqdm(range(1, 7), desc=f'Reducing dimensionality, {name}'):
        points = np.array([s.mean(axis=0) for s in data[f'layer_{i - 1}']])
        points_2d["generic"].append(dim_reduce(points, algo, 0, kwargs))
        points -= points.mean(axis=0)
        points_2d["generic-centered"].append(
            dim_reduce(points, algo, 0, kwargs))

    def _plot(width, height, key, pname,
              bbox_to_anchor, label_size, alpha, layer_label, palette):
        fig = plt.figure(figsize=(width, height))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0, hspace=0)

        for i in tqdm(range(1, 7), desc=f'Plotting sentences, {key}-{pname}'):
            comp1, comp2 = points_2d[key][i - 1]
            ax = plt.subplot(1, 6, i)
            ax = scatterplot_points(ax, comp1, comp2, labels, palette, alpha)
            if layer_label is not None:
                ax.set_title(f"Layer {i}", size=label_size)
            ax.get_legend().remove()

        handles, _labels = ax.get_legend_handles_labels()
        fig.legend(handles, _labels, loc='lower center',
                   prop={'size': label_size},
                   handlelength=1, borderpad=0,
                   columnspacing=3, borderaxespad=0,
                   bbox_to_anchor=bbox_to_anchor, ncol=2)

        out_path = os.path.join(path, 'sentence_representation')
        Path(out_path).mkdir(parents=True, exist_ok=True)

        fig.savefig(os.path.join(out_path, f'{pname}.png'),
                    bbox_inches='tight', pad_inches=0, dpi=300)
        fig.savefig(os.path.join(out_path, f'{pname}.pdf'),
                    bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig)

    for subset in points_2d.keys():
        model = path.split(os.sep)[-2]
        # for p in ["tab10", "Dark2", "Set1", "bright"]:
        for p in ["bright"]:
            if p == "bright":
                palette = sns.color_palette("bright").as_hex()[
                          :len(set(labels))]
            else:
                palette = p

            _plot(30, 5, subset, f"{model}_text_{subset}_{name}_palette={p}",
                  (0.5, 0.03), 24, 0.5,
                  layer_label="Layer ", palette=palette)
            _plot(15, 2.5, subset, f"{model}_col_{subset}_{name}_palette={p}",
                  (0.5, -0.03), 22, 0.3,
                  layer_label="L", palette=palette)


def plot_sentence_noise_comparison(path, data, sent_data, algo, kwargs):
    name = '_'.join([algo] + [f'{key}={value}' for key, value
                              in kwargs.items()])

    ordinary_mask, langid_mask, special_mask, labels, _, _ = sent_data
    labels = [x.replace("[", "").replace("]", "") for x in labels]

    token_types = {"langid": langid_mask,
                   "sos": special_mask,
                   "ordinary": ordinary_mask}

    labels = [x + y for x, y
              in zip(labels + labels,
                     [""] * len(labels) + [" w/o noise"] * len(labels))]

    points_2d = defaultdict(list)
    for subset, mask in token_types.items():
        for i in tqdm(range(1, 7), desc=f'Reducing dimensionality, {name}'):
            p_normal = np.array([s[m].mean(0) for s, m in
                                 zip(data[f'layer_{i - 1}'], mask)])
            p_pure = np.array([s[m].mean(0) for s, m in
                               zip(data[f'uncorrupted_layer_{i - 1}'], mask)])
            points = np.concatenate([p_normal, p_pure])
            points_2d[subset].append(dim_reduce(points, algo, 0, kwargs))

    def _plot(width, height, key, pname, bbox_to_anchor,
              label_size, alpha, layer_label, palette):
        fig = plt.figure(figsize=(width, height))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0, hspace=0)

        for i in tqdm(range(1, 7), desc=f'Plotting sentences, {subset}-{name}'):
            ax = plt.subplot(1, 6, i)
            comp1, comp2 = points_2d[key][i - 1]
            ax = scatterplot_points(ax, comp1, comp2, labels, palette, alpha)
            if layer_label is not None:
                ax.set_title(f"Layer {i}", size=label_size)
            ax.get_legend().remove()

        handles, _labels = ax.get_legend_handles_labels()
        fig.legend(handles, _labels, loc='lower center',
                   prop={'size': label_size},
                   handlelength=1, borderpad=0,
                   columnspacing=3, borderaxespad=0,
                   bbox_to_anchor=bbox_to_anchor, ncol=4)

        out_path = os.path.join(path,
                                'sentence_representation_noise_comparison')
        Path(out_path).mkdir(parents=True, exist_ok=True)

        fig.savefig(os.path.join(out_path, f'{pname}.png'),
                    bbox_inches='tight', pad_inches=0, dpi=300)
        fig.savefig(os.path.join(out_path, f'{pname}.pdf'),
                    bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig)

    for subset, mask in token_types.items():
        # for p in ["tab10", "Dark2", "Set1", "bright"]:
        for p in ["bright"]:
            palette = p
            if p == "bright":
                palette = sns.color_palette("bright").as_hex()[
                          :len(set(labels))]

            model = path.split(os.sep)[-2]
            _plot(30, 5, subset, f"{model}_text_{subset}_{name}_palette={p}",
                  (0.5, 0.03), 24, 0.5, layer_label="Layer ", palette=palette)
            _plot(15, 2.5, subset, f"{model}_col_{subset}_{name}_palette={p}",
                  (0.5, -0.03), 22, 0.3, layer_label="L", palette=palette)


def scatterplot_points(ax, x, y, labels, palette, alpha=0.6, point_size=None):
    if point_size is None:
        point_size = min(1, 100.0 / np.sqrt(len(labels)))
        # point_size = 1
    ax = sns.scatterplot(x, y, hue=labels,
                         hue_order=sorted(set(labels)),
                         ax=ax, palette=palette,
                         s=point_size, linewidth=0, alpha=alpha,
                         # kwargs=dict(rasterized=True),
                         rasterized=True)
    ax.tick_params(right=False, top=False, left=False, bottom=False)
    ax.tick_params(labelbottom=False, labeltop=False, labelleft=False,
                   labelright=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return ax


def scatterplot_labels(ax, x, y, classes, labels, palette):
    assert len(labels) == len(x) == len(y) == len(classes)
    point_size = 100.0 / np.sqrt(len(classes))
    classes_order = list(sorted(set(classes)))
    texts = []

    x += np.random.normal(0, .05, len(x))
    y += np.random.normal(0, .05, len(y))

    for i in tqdm(range(0, len(classes)), desc="scatterplot_labels"):
        texts.append(ax.text(x[i], y[i], labels[i],
                             horizontalalignment='center', alpha=0.66,
                             # fontstretch='ultra-expanded',
                             color=palette[classes_order.index(classes[i])],
                             fontsize=point_size / 4))
    return texts


def plot_tokens(path, data, algo, kwargs, top, limit, tag="", tok_data=None):
    name = '_'.join([algo] + [f'{k}={v}' for k, v in kwargs.items()])

    if tok_data is None:
        mask, subset_mask, src, trg, langs, noise, langmu = _prep_tok_data(data,
                                                                           top,
                                                                           limit)
    else:
        mask, subset_mask, src, trg, langs, noise, langmu = tok_data

    name += tag

    langs = [x.replace("[", "").replace("]", "") for x in langs]
    noise = [x.lower() for x in noise]

    points_2d = []
    for i in tqdm(range(1, 7), desc=f'Reducing dimensionality, {name}'):
        # concat all states into a single 2D array and apply the filter mask
        points = np.concatenate(data[f'layer_{i - 1}'], axis=0)[flatten(mask)]
        if subset_mask is not None:
            points = points[subset_mask]
        points_2d.append(dim_reduce(points, algo, 0, kwargs))

    # -------------------------------------------------------------------
    # Plot Points
    # -------------------------------------------------------------------
    def _plot(width, height, hspace, pname,
              bbox_to_anchors,
              label_size='large', alpha=0.3, layer_label=None,
              palette=['r', 'b']):
        fig = plt.figure(figsize=(width, height))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0, hspace=hspace)

        npoints = len(points_2d[0][0])
        # Color by language
        for i in tqdm(range(1, 7),
                      desc=f'Plotting tokens (n={npoints}), {pname}'):
            x, y = points_2d[i - 1]
            ax_lang = plt.subplot(2, 6, i)
            ax_lang = scatterplot_points(ax_lang, x, y, langs, palette, alpha)

            if layer_label is not None:
                ax_lang.set_title(f"{layer_label}{i}", size=label_size)

            ax_lang.get_legend().remove()

        handles, labels = ax_lang.get_legend_handles_labels()
        fig.legend(handles, labels,
                   # title="Language",
                   handlelength=0.5,
                   borderpad=0,
                   columnspacing=3.23,
                   borderaxespad=0,
                   loc='lower center', prop={'size': label_size},
                   bbox_to_anchor=bbox_to_anchors[0], ncol=2)

        # Color by noise
        for i in tqdm(range(1, 7), desc=f'Plotting tokens, {pname}'):
            x, y = points_2d[i - 1]
            ax_noise = plt.subplot(2, 6, i + 6)
            ax_noise = scatterplot_points(ax_noise, x, y, noise, palette, alpha)
            ax_noise.get_legend().remove()

        handles, labels = ax_noise.get_legend_handles_labels()
        fig.legend(handles, labels,
                   # title="Noise",
                   handlelength=0.5,
                   borderpad=0, labelspacing=0, borderaxespad=0,
                   loc='lower center', prop={'size': label_size},
                   bbox_to_anchor=bbox_to_anchors[1], ncol=2)

        out_path = os.path.join(path, 'token_representations')
        Path(out_path).mkdir(parents=True, exist_ok=True)

        fig.savefig(os.path.join(out_path, f'{pname}.png'),
                    bbox_inches='tight', pad_inches=0.02, dpi=300)
        fig.savefig(os.path.join(out_path, f'{pname}.pdf'),
                    bbox_inches='tight', pad_inches=0.02, dpi=300)
        plt.close(fig)

    model = path.split(os.sep)[-2]

    out_path = os.path.join(path, 'token_representations')
    Path(out_path).mkdir(parents=True, exist_ok=True)
    rep_data = {
        "points": points_2d,
        "tok_data": tok_data,
    }
    with open(os.path.join(out_path, f'{model}_{name}.pickle'), 'wb') as f:
        pickle.dump(rep_data, f)

    # for p in ["tab10", "Dark2", "Set1", "bright"]:
    for p in ["bright"]:
        if p == "bright":
            palette = sns.color_palette("bright").as_hex()[:2]
        elif p == "custom":
            palette = ["#386CB0", "#F0027F"]
        else:
            palette = p

        _plot(9, 4, 0.5, f"{model}_col_" + name + f"_palette={p}",
              [(0.4912, 0.49), (0.5, 0.03)],
              22, 0.3, layer_label="L", palette=palette)
        _plot(20, 6.4, 0.3, f"{model}_text_" + name + f"_palette={p}",
              [(0.496, 0.485), (0.5, 0.05)],
              24, 0.6, layer_label="Layer ", palette=palette)


def denoise(embeddings,
            unit_var=False,
            drop_first_n=0):
    pca = PCA()
    embs_pca = pca.fit_transform(embeddings)

    # zero-center the data
    print("Centering word embeddings (mean subtraction) ...")
    embeddings -= embeddings.mean(axis=0)

    if unit_var:
        print("Normalizing word embeddings (unit variance) ...")
        embeddings /= embeddings.std(axis=0)

    # get the data covariance matrix
    cov = np.dot(embeddings.T, embeddings) / embeddings.shape[0]

    # plt.imshow(cov, cmap='hot', interpolation='nearest')
    # plt.show()

    U, S, V = np.linalg.svd(cov)

    if drop_first_n > 0:
        U = U[:, drop_first_n:]

    embeddings = np.dot(embeddings, U)
    # return embeddings, U, S

    return embeddings


def homogeneity_sentences(path, data, ncentroids=512):
    ordinary_mask, _, _, labels, _, _ = _prep_sent_data(data)
    name = f'vmeasure_sentences={ncentroids}'

    scores = defaultdict(list)

    for i in tqdm(range(1, 7), desc=f'Sentence Homogeneity'):
        points = np.array([s[m].mean(0) for s, m
                           in zip(data[f'layer_{i - 1}'], ordinary_mask)])
        # print(points.shape)
        # kmeans = KMeans(nclasses)
        # kmeans.fit(points)

        ncentroids = 512
        niter = 25
        clustering = faiss.Kmeans(points.shape[1], ncentroids,
                                  niter=niter, verbose=True)
        clustering.train(points)
        centroids, labels = clustering.assign(points)

        ho, co, vm = metrics.homogeneity_completeness_v_measure(labels, labels)
        scores["homogeneity"].append(ho)
        scores["completeness"].append(co)
        scores["vmeasure"].append(vm)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    for score, values in scores.items():
        ax.plot(range(1, len(values) + 1), values, label=score)
        ax.set_xlabel("Layers")

    ax.legend()

    out_path = os.path.join(path, 'cluster_analysis')
    Path(out_path).mkdir(parents=True, exist_ok=True)

    fig.savefig(os.path.join(out_path, f'{name}.jpeg'),
                bbox_inches='tight', pad_inches=0, dpi=200)
    fig.savefig(os.path.join(out_path, f'{name}.pdf'),
                bbox_inches='tight', pad_inches=0)

    scores["size"] = points.shape[0]

    with open(os.path.join(out_path, f'{name}.json'),
              'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    return scores


def homogeneity_tokens(path, data, top, limit, ncentroids=512,
                       center_lang=False):
    scores = {"language": defaultdict(list), "noise": defaultdict(list)}
    mask, sample_mask, src, trg, langs, noise, langmu = _prep_tok_data(data,
                                                                       top,
                                                                       limit)

    name = f'vmeasure_tokens_top={top}_ncentroids={ncentroids}'

    if center_lang:
        name += "_lang-center"

    for i in tqdm(range(1, 7), desc=f'Token Homogeneity'):
        # concat all states into a single 2D array and apply the filter mask
        points = np.concatenate(data[f'layer_{i - 1}'], axis=0)[flatten(mask)]
        if sample_mask is not None:
            points = points[sample_mask]

        if center_lang:

            for lang, mu in langmu.items():
                lang_mean = np.expand_dims(mu[i - 1], axis=0)
                lang_offsets = np.expand_dims((langs == lang).astype("float"),
                                              axis=1)
                points -= lang_offsets.dot(lang_mean)

        niter = 25
        clustering = faiss.Kmeans(points.shape[1], ncentroids,
                                  niter=niter, verbose=True)
        scores["algorithm"] = f"{repr(clustering)}:{vars(clustering)}"
        scores["size"] = points.shape[0]
        clustering.train(points)
        centroids, labels = clustering.assign(points)

        h, c, v = metrics.homogeneity_completeness_v_measure(langs, labels)
        scores["language"]["homogeneity"].append(h)
        scores["language"]["completeness"].append(c)
        scores["language"]["vmeasure"].append(v)

        h, c, v = metrics.homogeneity_completeness_v_measure(noise, labels)
        scores["noise"]["homogeneity"].append(h)
        scores["noise"]["completeness"].append(c)
        scores["noise"]["vmeasure"].append(v)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0, hspace=0.3)

    for score, values in scores["language"].items():
        axes[0].plot(range(1, len(values) + 1), values, label=score)
        # axes[0].set_xlabel("Layers")
        axes[0].set_title('Language')
        axes[0].legend()

    for score, values in scores["noise"].items():
        axes[1].plot(range(1, len(values) + 1), values, label=score)
        axes[1].set_title('Real-Fake')
        axes[1].set_xlabel("Layers")
        axes[1].legend()

    out_path = os.path.join(path, 'cluster_analysis')
    Path(out_path).mkdir(parents=True, exist_ok=True)

    fig.savefig(os.path.join(out_path, f'{name}.jpeg'),
                bbox_inches='tight', pad_inches=0, dpi=200)
    fig.savefig(os.path.join(out_path, f'{name}.pdf'),
                bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)

    with open(os.path.join(out_path, f'{name}.json'),
              'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    return scores


def kmeans_elbow(path, data, top):
    mask = [[3 < x < top for x in sent] for sent in data['targets']]
    points = np.concatenate(data[f'layer_5'], axis=0)[flatten(mask)]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    visualizer = KElbowVisualizer(KMeans(), k=(1, 30), ax=ax)
    visualizer.fit(points)  # Fit the data to the visualizer
    visualizer.finalize()

    out_path = os.path.join(path, 'cluster_analysis')
    Path(out_path).mkdir(parents=True, exist_ok=True)

    fig.savefig(os.path.join(out_path, 'kmeans_elbow.jpeg'),
                bbox_inches='tight', pad_inches=0, dpi=200)
    fig.savefig(os.path.join(out_path, 'kmeans_elbow.pdf'),
                bbox_inches='tight', pad_inches=0)


def iplot_sentences(path, data):
    ordinary_mask, langid_mask, special_mask, labels, inputs, targets = _prep_sent_data(
        data)
    token_types = {"language_ids": langid_mask,
                   "bos_eos": special_mask,
                   "ordinary": ordinary_mask}

    inputs = [x.replace("<s>", "[s]")
                  .replace("<mask>", " [M] ")
                  .replace("</s>", "[/s]") for x in inputs]
    targets = [x.replace("<s>", "[s]")
                   .replace("<mask>", " [M] ")
                   .replace("</s>", "[/s]") for x in targets]

    for subset, mask in token_types.items():
        print(f"Plot of {subset} token representations")
        output_file(f"sentence_iplot{subset}.html")

        points = np.array([s[m].mean(0) for s, m
                           in zip(data[f'layer_5'], mask)])
        mapper = umap.UMAP().fit(points)

        _label_map = {lang: i for i, lang in enumerate(set(labels))}
        _labels = [_label_map[x] for x in labels]
        _hover_data = [{"index": i, "label": l, "item": text}
                       for i, (l, text) in enumerate(zip(_labels, targets))]
        _hover_data = pd.DataFrame().from_records(_hover_data)
        p = umap.plot.interactive(mapper,
                                  labels=_labels,
                                  hover_data=_hover_data,
                                  interactive_text_search=True,
                                  point_size=3,
                                  )

        out_path = os.path.join(path, 'interactive_plots')
        Path(out_path).mkdir(parents=True, exist_ok=True)
        file = os.path.join(out_path, f"sentences_{subset}.html")
        bokeh.plotting.save(p, file, title=f"Sentences ({subset})")


def iplot_tokens(path, data, top, limit):
    mask, subset_mask, src, trg, langs, noise, langmu = _prep_tok_data(data,
                                                                       top,
                                                                       limit)
    text = [f"{x} → {y}" if x != y else x for x, y in zip(src, trg)]
    text = [x.replace("<mask>", '[M]')
                .replace("▁", '')
                .replace("<", '[')
                .replace(">", ']')
            for x in text]

    hover_data = [{"index": i, "label": f"{l} - {n}".lower(), "item": t}
                  for i, (l, n, t) in enumerate(zip(langs, noise, text))]
    hover_data = pd.DataFrame().from_records(hover_data)
    # -------------------------------------------------------------------
    # Plot Points
    # -------------------------------------------------------------------
    for i in tqdm(range(1, 7), desc=f'Reducing dimensionality'):
        # concat all states into a single 2D array and apply the filter mask
        points = np.concatenate(data[f'layer_{i - 1}'], axis=0)[flatten(mask)]
        if subset_mask is not None:
            points = points[subset_mask]
        mapper = umap.UMAP().fit(points)

        for label_name, label_values in {"Language": langs,
                                         "Noise": noise}.items():
            _label_map = {lang: i for i, lang in enumerate(set(label_values))}
            _labels = [_label_map[x] for x in label_values]
            p = umap.plot.interactive(mapper,
                                      labels=_labels,
                                      hover_data=hover_data,
                                      width=900, height=900,
                                      point_size=3)

            out_path = os.path.join(path, 'interactive_plots')
            Path(out_path).mkdir(parents=True, exist_ok=True)
            file = os.path.join(out_path,
                                f"tokens_layer{i}_{label_name.lower()}.html")
            bokeh.plotting.save(p, file,
                                title=f"Token Layer:{i} ({label_name})")


def viz_encoder(path, data):
    print(f"Sentences={len(data['targets'])}, "
          f"Tokens={sum(len(x) for x in data['encoder'])}")

    try:
        iplot_sentences(path, data)
    except Exception as e:
        print("iplot_sentences Exception", e)
    try:
        iplot_tokens(path, data, top=6000, limit=80000)
    except Exception as e:
        print("iplot_tokens Exception", e)

    # --------------------------------------------------------------------
    # Sentence-level analysis
    # --------------------------------------------------------------------
    sent_data = _prep_sent_data(data)

    try:
        homogeneity_sentences(path, data)
        plot_sentences(path, data, sent_data, 'pca', {})
        plot_sentences(path, data, sent_data, 'tsne', {})
        plot_sentences(path, data, sent_data, 'tsne2', {})
        plot_sentence_noise_comparison(path, data, sent_data, 'pca', {})
        plot_sentence_noise_comparison(path, data, sent_data, 'tsne', {})
        plot_sentence_noise_comparison(path, data, sent_data, 'tsne2', {})
    except Exception as e:
        print(e)

    for nb, metric in list(product([15, 50], ['euclidean', 'cosine'])):
        try:
            plot_sentences(path, data, sent_data, 'umap', {'n_neighbors': nb,
                                                           'min_dist': 0.15,
                                                           'n_components': 2,
                                                           'metric': metric})
            plot_sentence_noise_comparison(path, data, sent_data, 'umap',
                                           {'n_neighbors': nb,
                                            'min_dist': 0.15,
                                            'n_components': 2,
                                            'metric': metric})

            plot_sentences(path, data, sent_data, 'tsne', {})
            plot_sentence_noise_comparison(path, data, sent_data, 'tsne', {})

            plot_sentences(path, data, sent_data, 'tsne2', {})
            plot_sentence_noise_comparison(path, data, sent_data, 'tsne2', {})

        except Exception as e:
            print(e)

    # --------------------------------------------------------------------
    # Token-level analysis
    # --------------------------------------------------------------------
    try:
        iplot_tokens(path, data, top=6000, limit=80000)
    except Exception as e:
        print("iplot_tokens Exception", e)

    try:
        kmeans_elbow(path, data, 5000)
    except:
        pass

    for topk, clusters in list(product([2000, 5000, 10000],
                                       [64, 128, 256, 512])):
        try:
            homogeneity_tokens(path, data, topk, 5000000, clusters)
            homogeneity_tokens(path, data, topk, 5000000, clusters, True)
        except Exception as e:
            print(e)

    top = 5000
    limit = 100000
    token_data = _prep_tok_data(data, top, limit)
    try:
        # plot_tokens(path, data, 'pca', {}, top, limit, tok_data=token_data)
        plot_tokens(path, data, 'tsne2', {}, top=2000, limit=limit, tag="_2K",
                    tok_data=token_data)
        plot_tokens(path, data, 'tsne2', {}, top=5000, limit=limit, tag="_5K",
                    tok_data=token_data)
        plot_tokens(path, data, 'tsne', {}, top=2000, limit=limit, tag="_2K",
                    tok_data=token_data)
        plot_tokens(path, data, 'tsne', {}, top=5000, limit=limit, tag="_5K",
                    tok_data=token_data)
    except Exception as e:
        print(e)

    for nb, metric in list(product([15, 50], ['euclidean', 'cosine'])):
        try:
            plot_tokens(path, data, 'umap', {'n_neighbors': nb,
                                             'min_dist': 0.15,
                                             'n_components': 2,
                                             'metric': metric},
                        top, limit, tok_data=token_data)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp', help="Path to the source file")
    opt = parser.parse_args()

    # from config import PATH_CP

    # experiment = 'ende_10M_marss_xlm_tune_word_tied_lr=0.0005_decay=0.0_clip=0.0_eps=1e-06_beta2=0.999'
    # experiment = 'ende10M_marss_word_035'
    # experiment = 'marss.deen.analysis_tied_replace=35_ertd=4'
    # experiment = 'marss.deen.analysis_mask=50'
    # opt.exp = os.path.join(PATH_CP, experiment)

    fname = os.path.join(opt.exp, 'analysis', 'states.bin')
    print(f"Loading {fname}...", end=" ")
    states_data = joblib.load(open(fname, 'rb'))
    print(f"done!")
    viz_encoder(os.path.join(opt.exp, 'analysis'), states_data)
