import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sciopt
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.cm as cm

#makeup
from tqdm import tqdm

#astropy packages
import evalmetric as eval
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
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.preprocessing import StandardScaler

#cross-match packages
from sklearn import metrics
import itertools


#gaia cluster analysis
def scoringfunction(dataselection, regiondata):
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
    

#Data prep/Manipulation functions
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
  def PlotGAIA(self, xaxis = "b", yaxis = "l",plotclose=True, **kwargs):
    
    plt.scatter(self.datatable[xaxis],self.datatable[yaxis], **kwargs)

    if plotclose:
      plt.title(f"{self.regionname}")
      plt.ylabel(yaxis)
      plt.xlabel(xaxis)
      plt.xlim(max(self.datatable[xaxis]),min(self.datatable[xaxis]))
      plt.show()

  def PlotRegion(self, xaxis = "b", yaxis = "l",plotclose=True, **kwargs):
    regionnames = np.unique(self.regiondata["population"])
    colors = [np.where(regionnames == i) for i in self.regiondata["population"]]
    plt.scatter(self.regiondata[xaxis],self.regiondata[yaxis],c=colors, **kwargs)

    if plotclose:
      plt.title(f"{self.regionname} known region")
      plt.ylabel(yaxis)
      plt.xlabel(xaxis)
      plt.xlim(max(self.regiondata[xaxis]),min(self.regiondata[xaxis]))
      plt.show()

  
  def PlotCluster(self, xaxis="b", yaxis ="l", clusterer="HDBSCAN", remove_outliers =False , plotclose=True ,**kwargs): #modified plot function with outlier filtration and Cluster selection
    try:
    
      plotdata = [self.datatable[xaxis], self.datatable[yaxis]]
      labels = self.datatable["population"]
      
      if remove_outliers != False: 
        threshold = pd.Series(self.datatable["probabilities"]).quantile(remove_outliers)
        out1 = np.where((self.datatable["probabilities"] > threshold) & (self.datatable["population"] != -1))[0]
        plt.scatter(np.take(plotdata[0],out1),np.take(plotdata[1],out1), c=np.take(labels,out1), **kwargs)
      
      if remove_outliers ==False:
        plt.scatter(*plotdata, c=labels, **kwargs)
      
      if plotclose:
        plt.ylabel(yaxis)
        plt.xlabel(xaxis)
        plt.title(f"{clusterer} clusters in \n {self.regionname} \n Outliers removed = {remove_outliers} quantile ")
        plt.show()
        
      
    except:
      if clusterer not in self.datatable.columns:
        print(f"Error: You did not perform the{clusterer} clustering yet. No {clusterer} column found in self.Datatable")
      


#Clustering functions
  def cluster(self, clusterer = HDBSCAN, dimensions = ["b","l","parallax","pmdec","pmra"],**kwargs):
        print(f"Running {clusterer.__class__.__name__} on {self.regionname} over {dimensions}\n")
        dataselection = [self.datatable[param] for param in dimensions] #N dimensional HDBscan
        data =StandardScaler().fit_transform(np.array(dataselection).T)
        clusterer = clusterer(**kwargs)
        clusterer.fit(data)
        #clusterer.fit_predict(data) #in case of artificial of unknown stars we can use fit_predict to predict the cluster they would belong to
        labels = clusterer.labels_ #list of all stars in which a number encodes to what cluster it is assigned
        try: 
          probabilities = clusterer.outlier_scores_ 
          self.datatable["probabilities"] = probabilities
        except:
          return
        self.datatable[f"{clusterer.__class__.__name__}"] = labels
        self.datatable["population"] = labels #append all labels to the designated "clustername "self.datatable table
        self.clusterer = clusterer  
        return clusterer 

  def silhouette_cluster(self, threshold=0,xaxis="l",yaxis="b", dimensions=["b","l","parallax","pmdec","pmra"], **kwargs):

    dataselection = [self.datatable[param] for param in dimensions] #N dimensional HDBscan
    data = StandardScaler().fit_transform(np.array(dataselection).T)
    #calculating silhouette    
    #index selection of common stars by source id


     #seperate out the noise regions (-1 region in HDBSCAN)
    #final_datatable_selection = common_elements_datatable[np.where(common_elements_datatable != -1)]
    thold = pd.Series(self.datatable["probabilities"]).quantile(threshold)
    mask = np.where((self.datatable["population"] != -1) & (self.datatable["probabilities"] > thold))
    final_datatable_selection = self.datatable[mask]["population"] #df format
    final_regiondata_selection = 0 #unneccesary
    final_dataselection_selection = data[mask]# dataselection[common_elements_datatable["population"] != -1] #np.array format
    sample_silhouette_values = eval.silhouettesample(final_regiondata_selection, final_datatable_selection, final_dataselection_selection)
    avg_silhouette_score = eval.silhouettescore(final_regiondata_selection, final_datatable_selection, final_dataselection_selection)
    n_clusters = len(np.unique(final_datatable_selection))


    #creating figure
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    # The 1st subplot is the silhouette plot
    ax1.set_title("The silhouette plot for the various clusters.")
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_xlim([-0.1, 1])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylabel("Cluster label")
    ax1.set_ylim([0, len(final_dataselection_selection) + (n_clusters + 1) * 10]) #change variables
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    #calculating individual silhouette scores
    y_lower = 10  # starting position on the y-axis of the next cluster to be rendered


    for i in np.unique(final_datatable_selection): # Here we make the colored shape for each cluster
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[final_datatable_selection == i]
        print(len(ith_cluster_silhouette_values))
        ith_cluster_silhouette_values.sort()

        # Figure out how much room on the y-axis to reserve for this cluster
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        y_range = np.arange(y_lower, y_upper)

        # Use matplotlib color maps to make each cluster a different color, based on the total number of clusters.
        # We use this to make sure the colors in the right plot will match those on the left.
        color = cm.nipy_spectral(float(i) / n_clusters)

        # Draw the cluster's overall silhouette by drawing one horizontal stripe for each datapoint in it
        ax1.fill_betweenx(y=y_range,                            # y-coordinates of the stripes
                          x1=0,                                 # all stripes start touching the y-axis
                          x2=ith_cluster_silhouette_values,     # ... and they run as far as the silhouette values
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples  
    ax1.axvline(x=avg_silhouette_score, color="red", linestyle="--")
    

    #PLOT RIGHT PLOT
    #
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel(xaxis)
    ax2.set_ylabel(yaxis)

    plt.suptitle(f"Silhouette clustering analysis clustering on sample data with n_clusters = {n_clusters}",
                 fontsize=14, fontweight='bold')
    colors = cm.nipy_spectral(self.datatable[mask]["population"].astype(float) / n_clusters)  # make the colors match with the other plot
    ax2.scatter(self.datatable[mask][xaxis], self.datatable[mask][yaxis], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

    # Labeling the clusters
    centers = np.array([[np.average(self.datatable[self.datatable["population"] == i][xaxis]),np.average(self.datatable[self.datatable["population"] == i][yaxis])] for i in np.unique(final_datatable_selection)])
   
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')
    # Put numbers in those circles
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % (i+min(np.unique(final_datatable_selection))), alpha=1, s=50, edgecolor='k')
    plt.show()
    return 

  def optimize_grid(self, dimensions= ["b","l","parallax","pmdec","pmra"], clusterer=HDBSCAN, fit_params=None, scoring_function=scoringfunction, write_results=False, **kwargs):     
        dataselection = [self.datatable[param] for param in dimensions] #N dimensional HDBscan
        data = StandardScaler().fit_transform(np.array(dataselection).T)

        #logs
        scores= []
        param_values = []
        point_variable_names = [i["variable"]for i in fit_params]
        point_variable_list = [list(range(i["min"], i["max"])) for i in fit_params]
        combination = [p for p in itertools.product(*point_variable_list)]
        combination = [dict(zip(point_variable_names, i)) for i in combination]
        for i in tqdm(combination):
          print("Parameters :",i)
          cluster = clusterer(**i, **kwargs)
          cluster.fit(data)
          cluster.fit_predict(data) #in case of artificial of unknown stars we can use fit_predict to predict the cluster they would belong to
          labels = cluster.labels_
          self.datatable["population"] = labels

          #make selection of populations in both self.regiondata and self.dataselection. Only stars present in both tables are taken into account

          #index selection of common stars by source id
          redundant,spots1, spots2 = np.intersect1d(self.datatable["source_id"],self.regiondata["source_id"],return_indices=True)
          #select common stars in both data and region tables and sort
          common_elements_datatable = self.datatable[spots1].group_by("source_id")
          common_elements_regiondata = self.regiondata[spots2].group_by("source_id")

          #rename named regions to numbers for easier comparison
          common_elements_regiondata_numbered = np.array([np.where(np.unique(common_elements_regiondata["population"]) == i)[0][0] for i in common_elements_regiondata["population"]]) #convert named clusters to numbers
          
          #seperate out the noise regions (-1 region in HDBSCAN)
          #final_datatable_selection = common_elements_datatable[np.where(common_elements_datatable != -1)]
          final_datatable_selection = common_elements_datatable[common_elements_datatable["population"] != -1]["population"] #df format
          final_regiondata_selection = common_elements_regiondata_numbered[common_elements_datatable["population"] != -1] #np.array format

          final_dataselection_selection = data[spots1][common_elements_datatable["population"] != -1]# dataselection[common_elements_datatable["population"] != -1] #np.array format
          #print(len(final_datatable_selection), len(final_regiondata_selection),len(final_dataselection_selection))

          #scoring and adding to log
          scores.append(scoring_function(final_regiondata_selection, final_datatable_selection, final_dataselection_selection)) # score the regions
          param_values.append(i)

        max_score_index, max_score = np.argmax(scores) , np.max(scores)
        
        if write_results:
          with open(f"optimized results.txt","w") as f:
            combinated= [param_values,scores]
            for row in zip(*combinated):
              f.write((str(row))+'\n')
        return param_values[max_score_index], np.max(scores)
