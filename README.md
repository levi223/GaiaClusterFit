# GaiaClusterFit

GaiaClusterFit is a Python library for dealing cluster

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install GaiaClusterFit
```

## Usage

Import library
```python
from  GaiaClusterFit import GCA
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
optimal = job.optimize_grid(fit_params=parameters, scoring_function)
```
Scoring function returns a score for the fit based by default on homogeneity self-made score functions can be passed and recieve an astropy gaia table and an astropy region table. optimize_grid returns parameters for the highest score

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)