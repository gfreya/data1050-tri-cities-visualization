# Tri-Cities Load Interconnection capability Visualization
> DATA1050 Project Fall 2021

- Dataset: 
	- https://transmission.bpa.gov/Business/Operations/Charts/ashe.txt
	- https://transmission.bpa.gov/Business/Operations/Charts/triCities.txt

- Libraries and Packages
	- Python3
	- requests
	- numpy
	- pandas
	- io 
	- tensorflow
	- pymongo
	- plotly

- This is a web application to retrieve real-time time from web server and then join the two tables together. The newly merged dataframe will be uploaded to mongodb backend. Then we can load the new data from backend.

- This web application has following functions: 
	- demonstrate the time-series waveform of "voltage value", "load", "import" and "generation".
	- demonstrate the  interactive plot for users to compare the utility in tri-cities and the recent project: Ashe main bus voltage.
	- predict the future values according to past values of "voltage value", "load" and "import".
	- detect anomaly voltage values in a time period.
	
- Team Member: Yijing Gao and Qingyan Guo

Reference: [https://github.com/BrownDSI/data1050-demo-project-f20.git](https://github.com/BrownDSI/data1050-demo-project-f20.git)
