from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set_style("darkgrid")
from w2v_model import model

# create T-SNE visualization for all vocabulary of embedding model
def common_tsne_plot(model):
    labels = []
    tokens = []
    for word in model.wv.index2entity[:100]:
        tokens.append(model[word])
        labels.append(word)
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])       
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

common_tsne_plot(model)