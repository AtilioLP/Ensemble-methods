library(covidHubUtils)
library(arrow)

# # Download truth data
truth_data <- load_truth(
  truth_source = "JHU",
  target_variable = "inc death",
  locations = "US"
)

start_date <- as.Date("2020-05-18")  # Ensemble first appearance
end_date <- as.Date("2022-03-14")
all_dates <- seq.Date(start_date, end_date, by = "day")

# Here we get only the target for 'inc death' with forecasts
# for 1,2,3 and 4 weeks ahead.

inc_case_targets <- paste(1:4, "wk ahead inc death")

forecasts <- load_forecasts(
  dates            = all_dates,
  targets = inc_case_targets,
  date_window_size = 0,
  locations        = "US",       # US national level
  types            = c("point", "quantile"),
  source           = "zoltar",   # default source
  verbose          = TRUE
)

scores <- score_forecasts(
  forecasts = forecasts,
  return_format = "wide",
  truth = truth_data
)

designations <- get_model_designations(source = "zoltar",hub="US")

write_parquet(forecasts, "data/fullforecasts_death.parquet")
write_parquet(scores, "data/scores_covidhubutils_death.parquet")
write_parquet(designations, "data/models_death.parquet")
write_parquet(truth_data, "data/covid_truth_data_death.parquet")
