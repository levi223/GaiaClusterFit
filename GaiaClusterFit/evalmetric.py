import sklearn as sk
import numpy as np

def homogeneityscore(knownregions, predictedregions,data=None):
    """Cross-match-scores 2 sets of clustered data on a homogeneity score
    Args:
        dataselection (astropy.Table): Astropy Table that includes all imported Gaia data of the Queried region.
        regiondata (astropy.Table): Astropy Table that includes all imported luster data .

    Returns:
        Float: The return value. True for success, False otherwise.
    """
    score = sk.metrics.homogeneity_score(knownregions,predictedregions)
    return score

def completenessscore(knownregions, predictedregions,data=None):
    score = sk.metrics.completeness_score(knownregions,predictedregions)
    return score


def randscore(knownregions, predictedregions,data=None):
    """Cross-match-scores 2 sets of clustered data on a homogeneity score
    Args:
        dataselection (astropy.Table): Astropy Table that includes all imported Gaia data of the Queried region.
        regiondata (astropy.Table): Astropy Table that includes all imported luster data .

    Returns:
        Float: The return value. True for success, False otherwise.
    """
    score = sk.metrics.rand_score(knownregions,predictedregions)
    return score

def calinskiharabaszscore(knownregions, predictedregions,data=None):
    """Cross-match-scores 2 sets of clustered data on a homogeneity score
    Args:
        dataselection (astropy.Table): Astropy Table that includes all imported Gaia data of the Queried region.
        regiondata (astropy.Table): Astropy Table that includes all imported luster data .

    Returns:
        Float: The return value. True for success, False otherwise.
    """
    score = sk.metrics.calinski_harabasz_score(knownregions,predictedregions)
    return score

def mutualinfoscore(knownregions, predictedregions,data=None):
    """Cross-match-scores 2 sets of clustered data on a homogeneity score
    Args:
        dataselection (astropy.Table): Astropy Table that includes all imported Gaia data of the Queried region.
        regiondata (astropy.Table): Astropy Table that includes all imported luster data .

    Returns:
        Float: The return value. True for success, False otherwise.
    """
    score = sk.metrics.mutual_info_score(knownregions,predictedregions)
    return score

def daviesbouldinscore(knownregions, predictedregions,data=None):
    """Cross-match-scores 2 sets of clustered data on a homogeneity score
    Args:
        dataselection (astropy.Table): Astropy Table that includes all imported Gaia data of the Queried region.
        regiondata (astropy.Table): Astropy Table that includes all imported luster data .

    Returns:
        Float: The return value. True for success, False otherwise.
    """
    score = sk.metrics.davies_bouldin_score(knownregions,predictedregions)
    return score

def vmeasurescore(knownregions, predictedregions,data=None):
    """Cross-match-scores 2 sets of clustered data on a v-measure score
    Args:
        dataselection (astropy.Table): Astropy Table that includes all imported Gaia data of the Queried region.
        regiondata (astropy.Table): Astropy Table that includes all imported luster data .

    Returns:
        Float: Score between 0 and 1
    """

    score = sk.metrics.v_measure_score(knownregions,predictedregions)
    return score


def silhouettescore(knownregions, predictedregions,data=None):

    #print(np.unique(knownregions))
    #print(len(predictedregions), len(knownregions))
    try:
        
        score = sk.metrics.silhouette_score(np.array(data),knownregions)
        print("Shilhouette score :",score)
        return score

    except Exception as e:
        print("Could not compute silhouette score")
        print(f"Error message:{e}")
        return float("nan")

