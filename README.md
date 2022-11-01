# GaiaClusterFit

GaiaClusterFit is a Python library for optuimizing GAIA clustering

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install GaiaClusterFit.

```bash
pip install GaiaClusterFit
```

## Basic Usage

Import library
```python
from  GaiaClusterFit import GCA
from  GaiaClusterFit import evalmetric
```
Specify Gaia query
```python
#GAIA database query
query ="""SELECT TOP 1000  source_id, b, l, parallax,phot_g_mean_mag,pmra,pmdec, RUWE, bp_rp,phot_g_mean_mag+5*log10(parallax)-10 as mg
FROM gaiadr3.gaia_source
WHERE l < 275 AND l > 240 
AND b < 5 AND b > -15
AND phot_g_mean_mag < 18
AND RUWE < 1.4
AND parallax < 4 AND parallax > 1.8
AND parallax_error/parallax < 0.02""" 

````
Create an instance and import data
```python
#Create instance
job = GCA.GCAinstance(RegionName = "Char")

#Login and fetch GAIA Data
job.GaiaLogin(username='username', password='password')
job.FetchQueryAsync(query)

#Import known cluster
job.ImportRegion("G:/path/known_cluster.fits")

```
Setting up basic cluster fit function to clustered GAIA data to known clusters

```python
#Parameters to optimize Cluster function over (HDBscan by default)
parameters = [{"variable": "min_cluster_size", "min":10, "max":100}]

```
Renaming cluster table columns to match GAIA column names
```python
job.RenameCol(job.regiondata, [["Source", "source_id"],["Pop", "population"]])

```
Optimizing cluster function(HDBscan) over GAIA data to match known clusters
```python
optimal = job.optimize_grid(fit_params=parameters, evalmetric.homogeneityscore)
```
Scoring function returns a score for the fit based by default on homogeneity self-made score functions can be passed and recieve an astropy gaia table and an astropy region table. optimize_grid returns parameters for the highest score

## Code Discriptions



### GCA.GCAinstance
```python
GCAinstance(data =None, regiondata =None, RegionName = "No region Name")
```
Creates an instance object class used for clusteringa and cluster match scoreing later on.
`(data =None, regiondata =None, RegionName = "No region Name")`  are optional. 
Later `instance.Datatable` and `instance.Regiondata` can be populated by querying the GAIA database (`GCAinstance.GaiaLogin`  and `GCAinstance.FetchQueryAsync`) or by uploading a Gaia FITs table through `instance.ImportDataTable` and `instance.ImportRegion` 

* `data` : an astropy.table table containing star data 
* `regiondata`: an astropy.table table containing known cluster data

### GCAinstance.ImportDataTable()
```python
def ImportDataTable(self,path): #import a fits datatable comming from Gaia or whatever
  self.datatable =Table(fits.open(path)[1].data)
``` 
Imports a GAIA table from the .fits format and stores it to self.datatable

* `path`: a string specifying the path to the .fits table file containing star data 

### GCAinstance.ExportDataTable()
```python
def ExportDataTable(self, path, **kwargs): #export the self.datatable to any format(for importing measures i would recommend .fits)
     self.datatable.write(f'{path}',**kwargs)ta)
``` 
Exports self.datatable to a .fits file at a specified path. Kwargs translate over from `astropy.io.ascii.write(**kwargs)` function 
* `path`: a string specifying the path where the .fits table file containing star data will be stored


### GCAinstance.ImportRegion()
```python
def ImportDataTable(self,path): #import a fits datatable comming from Gaia or whatever
  self.regiondata =Table(fits.open(path)[1].data)
``` 
Imports a GAIA table from the .fits format and stores it to self.regiondata

* `path`: a string specifying the path to the .fits table file containing cluster region data 

### GCAinstance.ExportRegion()
```python
def ExportDataTable(self, path, **kwargs): #export the self.datatable to any format(for importing measures i would recommend .fits)
     self.regiondata.write(f'{path}',**kwargs)
``` 
Exports self.regiondata to a .fits file at a specified path. Kwargs translate over from `astropy.io.ascii.write(**kwargs)` function 
* `path`: a string specifying the path where the .fits table file containing cluster region data will be stored

### GCAinstance.GaiaLogin()

```python
def GaiaLogin(self, username, password):
  Gaia.login(user=str(username), password=str(password))
```
The `GCAinstance.GaiaLogin()` initiates a GAIA database session based on personal credentials (`username="username", password="password"`). This allows for asynchronous data queries (`GCAinstance.FetchQueryAsync()`) from the GAIA database. This session is constrained within the instance allowing multiple instances to initiate different sessions.

* `username`: a string specifying your GAIA username credential
* `password`: a string specifying your GAIA password credential
### GCAinstance.FetchQueryAsync()

```python
def FetchQueryAsync(self, query, **kwargs):
  job = Gaia.launch_job_async(query, **kwargs)
  self.datatable = job.get_results()
```

The `CAinstance.FetchQueryAsync(query, **kwargs)` function accepts a ADQL formatted query to fetch GAIA data. It writes this data to `GCAinstance.datatable` .
* `query`: a string containing the to be queried ADQL query
* `kwargs`: all keword arguments that the `Astroquery.Gaia.launch_job_async also accepts`

### GCAinstance.Renamecol()

```python
def RenameCol(self, table, newnames):
    for i in newnames:
      table.rename_column(i[0],i[1])
``` 
The Renamecol function converts the columnnames of an `astropy.table` object to a set of new names. Within GaiaClusterFit we require that the columns of the regions and GAIA data match column names. Therefore it is standard practice to convert the GCAinstance.regiondata columns to match that of the GAIA columns. I.E `GCAinstance.RenameCol(GCAinstance.regiondata, [["Source","Source_id"],["Pop",population]])`. The default columnname for labeled clusterdata in GCAinstance.datatable is `"population"` 

* `table`: astropy.table table object 
* `newnames`: 2D python list as such [["old column name 1","new column name 1"],["old column name 2","new column name 2"]]

### GCAinstance.Plot()

```python
def Plot(self, xaxis = "b", yaxis = "l", **kwargs):
    plt.title(f"{self.regionname}")
    plt.scatter(self.datatable[xaxis],self.datatable[yaxis], **kwargs)
    plt.ylabel(yaxis)
    plt.xlabel(xaxis)
    plt.xlim(max(self.datatable[xaxis]),min(self.datatable[yaxis]))
    plt.show()
``` 
`GCAinstance.Plot()` plots GCAinstance.datatable using matplotlib.pyplot. `x` and `y` dimensions of the plot can be controlled using `xaxis = "GAIA parameter" , yaxis = "GAIA parameter"'` where the GAIA parameter can be the string name of any column in GCAinstance.datatable. `**kwargs` takes any keywordargument `matplotlib.pyplot` accepts.

* `xaxis`: column name of column in `GCAinstance.datatable` to display on the x-axis
* `yaxis`: column name of column in `GCAinstance.datatable` to display on the y-axis
* `kwargs`: general keyword arguments accepted by matplotlib.pyplot.plot()


### GCAinstance.PlotCluster()
```python
  def PlotCluster(self, xaxis="b", yaxis ="l", clusterer="HDBSCAN", remove_outliers =False , **kwargs): #modified plot function with outlier filtration and Cluster selection
    try:
      fig, ax = plt.subplots(figsize=(10,10))

      plotdata = (self.datatable[xaxis], self.datatable[yaxis])
      labels = self.datatable[clusterer]

      if remove_outliers == True : 
        plotdata = self.datatable[xaxis][self.datatable[f"{remove_outliers}_outlier"]],self.datatable[yaxis][self.datatable[f"{remove_outliers}_outlier"]]
        labels = self.datatable[clusterer][self.datatable[f"{remove_outliers}_outlier"]]
      ax.set_title(f"{clusterer} clusters in \n {self.regionname} \n Outliers removed = {remove_outliers} ")
      ax.scatter(*plotdata, c=labels, **kwargs)
      ax.set_ylabel(yaxis)
      ax.set_xlabel(xaxis)
      plt.show()
      return fig,ax
    except:
      if clusterer not in self.datatable.columns:
        print(f"Error: You did not perform the{clusterer} clustering yet. No {clusterer} column found in self.Datatable")
      return fig,ax
```
The `GCAinstance.PlotCluster()`function plots the clusterdata alongside the `GCAinstance.datatable`  data. This requires GCAinstance.datatable` to be clustered before by GCAinstance.cluster()` function. The `GCAinstance.Plotcluster()` plots clusterlabels alongside GCAinstance.datatable using matplotlib.pyplot. `x` and `y` dimensions of the plot can be controlled using `xaxis = "GAIA parameter" , yaxis = "GAIA parameter"` where the GAIA parameter can be the string name of any column in GCAinstance.datatable. `**kwargs` takes any keywordargument `matplotlib.pyplot` accepts. 

* `xaxis`: column name of column in `GCAinstance.datatable` to display on the x-axis
* `yaxis`: column name of column in `GCAinstance.datatable` to display on the y-axis
* `clusterer`: cluster function name of which to display latest formed clusters

### GCAinstance.cluster()
```python
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
``` 
The `cluster(self, clusterer = HDBSCAN, dimensions = ["b","l","parallax","pmdec","pmra"],**kwargs)` clusters the `GCAinstance.datatable` data based on a specified cluster algorithm. The funnction returns the clusterer instance. Resulting Cluster labels are written to `GCAinstance.datatable["cluster algorithm name"]`

* `dimensions = ["GCAinstance.datatable column names"]`  determines which columns of GCA.datatable are used to cluster the data over
* `clusterer = cluster_algorithm` passes a clustering function that is used to cluster the data. By default this cluster function should only accept the to-be-clustered-data. i.e `clusterer = GCA.HDBSCAN` , `clusterer ='GCA.OPTICS', `clusterer = sklearn.cluster.DBSCAN`etc
* `**kwargs` accepts keywords arguments that are passed on to the cluster algorithms(HDBSCAN,DBSCAN etc)


### GCAinstance.optimize_grid()

```python
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
```
`GCAinstance.optimize_grid(self, dimensions= ["b","l","parallax","pmdec","pmra"], clusterer=HDBSCAN, fit_params=None, scoring_function=scoringfunction, **kwargs)'  fits cluster function `clusterer` based on a given set of parameter intervals `fit_params` to optimize a `scoring_unction`. This scoring function compares the predicted clusters to the true clusters. The highest score results in the best fit (according to the scoring_function).

The function returns a list of dictionaries with the optimized parameter values

* `dimensions` : the dimensions/datacolumns of GCAinstance.datatable we will cluster over
* `clusterer` : a clustering function that is used to cluster the data. By default this cluster function should only accept the to-be-clustered-data. i.e `clusterer = GCA.HDBSCAN` , `clusterer ='GCA.OPTICS', `clusterer = sklearn.cluster.DBSCAN`etc
* `fit_params`: Is a python-list containing dicts formatted as follows `[{"variable" :"cluster argument", "min":10, "max":20},{"variable" :"cluster argument", "min":5, "max":40}]`
*  `scoring_function`scoring function accepts a different function that takes `GCAinstance.datatable and GCAinstance.regiondata` A set of properly out of the box formatted scoring functions is included in `GaiaClusterFit.evalmetric`. 



## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
