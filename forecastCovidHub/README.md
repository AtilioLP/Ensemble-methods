This folder contains the code to replicate the ensemble methodology proposed in the paper [Comparing trained and untrained probabilistic ensemble forecasts of COVID-19 cases and deaths in the United States](https://www.sciencedirect.com/science/article/pii/S0169207022000966?via%3Dihub).

* `download_Data.R`: code to load the data used in the article;
* `Data_CovidHub.ipynb`: code to replicate the non-robust ensemble methodology described in the paper;
* `data/covid_truth_data.parquet`: contains the ground-truth case data;
* `data/fullforecasts.parquet`: contains the forecasts from all models over the period;
* `data/models.parquet`: list of models, indicating whether they are primary or not;
* `data/scores_covidhubutils`: contains the WIS values obtained using `covidhubutils`. These matched the results from the `compute_wis` function in the notebook (`Data_CovidHub.ipynb`) and were used to validate our WIS formula.
