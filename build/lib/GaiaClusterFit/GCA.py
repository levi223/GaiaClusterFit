import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sciopt

#makeup
from tqdm import tqdm

#astropy packages
import astropy.io 
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astropy.table import Table

#Clustering packages
 
from joblib import Memory
mem = Memory(location='/tmp/')
from hdbscan import HDBSCAN as HDBSCAN
#from cuml.cluster import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.preprocessing import StandardScaler

#cross-match packages
from sklearn import metrics
import itertools


def test():
  print("test")
#gaia cluster analysis
def scoringfunction(dataselection, regiondata):
    """Cross-match-scores 2 sets of clustered data on a homogeneity score
    Args:
        dataselection (astropy.Table): Astropy Table that includes all imported Gaia data of the Queried region.
        regiondata (astropy.Table): Astropy Table that includes all imported luster data .

    Returns:
        Float: The return value. True for success, False otherwise.
    """

    common_elements_data = np.isin(dataselection["source_id"],regiondata["source_id"])
    common_elements_region = np.isin(regiondata["source_id"],dataselection["source_id"])
    predicted_common_elements = dataselection[common_elements_data].group_by("source_id")
    true_common_elements = regiondata[common_elements_region].group_by("source_id")
        
    score = metrics.homogeneity_score(true_common_elements["population"], predicted_common_elements["population"])
    return score


class GCAinstance():
  """Main instance used for cluster fitting and clusterer fitting.


    Args:
        data (Astropy.Table): An astropy table containing all data for a selected set of stars
        the ImportDataTable() function can also import gaia.fits tables to an Astropy.Table

        regiondata (Astropy.Table): An astropy table containing all data for clusterdata
        the ImportRegion() function can also import .fits tables to an Astropy.Table

        Regionname (str): Internal region name (default = No Region Name)
    """

  def __init__(self, data =None,regiondata =None, RegionName = "No region Name"):
    self.regionname = RegionName  #Region name
    self.datatable = data #complete table containing all data
    self.regiondata =regiondata

  #data functions
  def GaiaLogin(self, username, password):
    """Gaia Login function used to connect GCA instance to Gaia database
       such that asynchronous querys can be passed. 

    Args:
        username (str): Gaia account username .
        password (str): Gaia account password

    Returns:
        Nothing
    """

    Gaia.login(user=str(username), password=str(password))

  def FetchQueryAsync(self, query, **kwargs):
    """Fetches Gaia database data based on Query. Returned data is saved in self.datatable

    Args:
        query (str): example - SELECT TOP 1000  source_id, b, l, parallax,phot_g_mean_mag,pmra,pmdec, RUWE, bp_rp,phot_g_mean_mag+5*log10(parallax)-10 as mg 
        FROM gaiadr3.gaia_source
        WHERE l < 275 AND l > 240
        

    Returns:
        Nothing
    """

    job = Gaia.launch_job_async(query, **kwargs)
    self.datatable = job.get_results()

  def ImportDataTable(self,path): #import a fits datatable comming from Gaia or whatever
      self.datatable =Table(fits.open(path)[1].data)
  
  def ExportDataTable(self, path, **kwargs): #export the self.datatable to any format(for importing measures i would recommend .fits)
      self.datatable.write(f'{path}',**kwargs)
  
  def ImportRegion(self, path, **kwargs):
      self.regiondata =Table(fits.open(path)[1].data)

  def ExportRegion(self, path, **kwargs):
      self.regiondata.write(f'{path}',**kwargs)

  def RenameCol(self, table, newnames):
    for i in newnames:
      table.rename_column(i[0],i[1])

  #plotting functions
  def Plot(self, xaxis = "b", yaxis = "l", **kwargs):
    plt.title(f"{self.regionname}")
    plt.scatter(self.datatable[xaxis],self.datatable[yaxis], **kwargs)
    plt.ylabel(yaxis)
    plt.xlabel(xaxis)
    plt.xlim(max(self.datatable[xaxis]),min(self.datatable[yaxis]))
    plt.show()
  
  def PlotCluster(self, xaxis="b", yaxis ="l", clusterer="HDBSCAN", remove_outliers = False , **kwargs): #modified plot function with outlier filtration and Cluster selection
    try:
      fig, ax = plt.subplots(figsize=(10,10))
      if remove_outliers == False:
        ax.set_title(f"{clusterer} clusters in \n {self.regionname}")
        ax.scatter(self.datatable[xaxis],self.datatable[yaxis], c=self.datatable[clusterer], **kwargs)
        ax.set_ylabel(yaxis)
        ax.set_xlabel(xaxis)
        plt.show()
      if remove_outliers != False:
        ax.set_title(f"{clusterer} clusters in \n {self.regionname} \n {remove_outliers} outliers removed")
        ax.scatter(self.datatable[xaxis][self.datatable[f"{remove_outliers}_outlier"]],self.datatable[yaxis][self.datatable[f"{remove_outliers}_outlier"]], c=self.datatable[clusterer][self.datatable[f"{remove_outliers}_outlier"]], **kwargs)
        ax.set_ylabel(yaxis)
        ax.set_xlabel(xaxis)
        plt.show()
    except:
      if clusterer not in self.datatable.columns:
        print(f"Error: You did not perform the{clusterer} clustering yet. No {clusterer} column found in self.Datatable")
      return fig,ax

  def cluster(self, clusterer = HDBSCAN, dimensions = ["b","l","parallax","pmdec","pmra"],**kwargs):
        print(f"Running {clusterer.__class__.__name__} on {self.regionname} over {dimensions}\n")
        dataselection = [self.datatable[param] for param in dimensions] #N dimensional HDBscan
        data =StandardScaler().fit_transform(np.array(dataselection).T)
        clusterer = clusterer(**kwargs)
        clusterer.fit(data)
        clusterer.fit_predict(data) #in case of artificial of unknown stars we can use fit_predict to predict the cluster they would belong to
        labels = clusterer.labels_ #list of all stars in which a number encodes to what cluster it is assigned
        self.datatable[f"{clusterer.__class__.__name__}"] = labels #append all labels to the designated "clustername "self.datatable table
        self.clusterer = clusterer  
        return clusterer 


  def optimize_grid(self, dimensions= ["b","l","parallax","pmdec","pmra"], clusterer=HDBSCAN, fit_params=None, scoring_function=scoringfunction, **kwargs):     
        dataselection = [self.datatable[param] for param in dimensions] #N dimensional HDBscan
        
        data = StandardScaler().fit_transform(np.array(dataselection).T)
        scores= []
        param_values = []
        point_variable_names = [i["variable"]for i in fit_params]
        point_variable_list = [list(range(i["min"], i["max"])) for i in fit_params]
        combination = [p for p in itertools.product(*point_variable_list)]
        combination = [dict(zip(point_variable_names, i)) for i in combination]
        for i in tqdm(combination):
          cluster = clusterer(**i, **kwargs)
          cluster.fit(data)
          cluster.fit_predict(data) #in case of artificial of unknown stars we can use fit_predict to predict the cluster they would belong to
          labels = cluster.labels_
          self.datatable["population"] = labels
          scores.append(scoring_function(self.datatable, self.regiondata))
          param_values.append(i)
        max_score_index, max_score = np.argmax(scores) , np.max(scores)
        return param_values[max_score_index]
