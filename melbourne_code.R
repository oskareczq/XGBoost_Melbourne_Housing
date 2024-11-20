# Loading packages
basic_packages <- c("tidyverse", "tidymodels", "ggthemes", "vip",
                    "dlookr", "scales", "doParallel", "xgboost")
invisible(sapply(basic_packages, library, character.only = TRUE))

# Data import
data <- read_csv("/Users/oskareczqu/Melbourne_dataset.csv", show_col_types = FALSE)
glimpse(data)

# Conversion of date type and categorical variables
data <- data %>% 
  mutate(Date = dmy(Date)) %>% 
  mutate(across(is.character, factor))

# Plot of missing data
data %>%
  plot_na_pareto(only_na = TRUE) 

# Property sales in Melbourne over the months
data %>%
  mutate(year = factor(year(Date)), month = lubridate::month(Date, label = TRUE, abbr = FALSE)) %>%
  group_by(month, year) %>%
  summarise(total_sales = n(), .groups = 'drop') %>%
  ggplot(aes(x = month, y = total_sales, fill = year)) +
  geom_col(position = "stack") +
  geom_text(aes(y = total_sales, label = total_sales), 
            position = position_stack(vjust = 0.5), color = "brown") +
  scale_fill_discrete(name = "Year") +
  theme_fivethirtyeight() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Property Sales in Melbourne", subtitle = "by Month",
       x = "Month", y = "Number of Sales")

# Number of properties by distance from city center
data %>%
  mutate(DistanceGroup = case_when(
    Distance < 5 ~ "< 5 km",
    Distance < 10 ~ "5-10 km",
    Distance < 15 ~ "10-15 km",
    Distance < 20 ~ "15-20 km",
    TRUE ~ "20+ km"
  )) %>%
  group_by(DistanceGroup) %>%
  summarise(Count = n()) %>%
  ggplot(aes(x = fct_reorder(DistanceGroup, Count), y = Count)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = Count), vjust = -0.2) +
  labs(title = "Number of Properties by Distance",
       subtitle = "from Melbourne City Center") +
  theme_fivethirtyeight()

# Distribution of property prices
data %>%
  ggplot(aes(x = Price)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  labs(title = "Distribution of Property Prices in Melbourne") +
  theme_fivethirtyeight() +
  scale_x_continuous(labels = label_currency(prefix = "$"))

# Average property prices over time
data %>%
  mutate(Month = floor_date(Date, "month")) %>%
  group_by(Month) %>%
  summarize(Avg_Price = mean(Price, na.rm = TRUE)) %>%
  ggplot(aes(x = Month, y = Avg_Price)) +
  geom_col() +
  labs(title = "Average Property Prices Over Time", x = "Date", y = "Average Price") +
  theme_fivethirtyeight() +
  scale_y_continuous(labels = label_currency(prefix = "$")) +
  theme(legend.position = "bottom")

# Property prices by housing type
data %>%
  ggplot(aes(x = factor(Type), y = Price, color = factor(Type))) +
  geom_boxplot(outlier.colour = NA, size = 1) +
  labs(title = "Prices by Housing Type") +
  theme_fivethirtyeight() +
  scale_y_continuous(labels = label_currency(prefix = "$")) +
  scale_x_discrete(labels = c("h" = "House", "t" = "Townhouse", "u" = "Unit")) +
  coord_cartesian(ylim = c(0, 4000000)) +
  theme(legend.position = "none")

# Property prices by room count
data %>%
  ggplot(aes(x = factor(Rooms), y = Price, color = factor(Rooms))) +
  geom_boxplot(outlier.colour = NA) +
  labs(title = "Property Prices by Number of Rooms") +
  theme_fivethirtyeight() +
  scale_y_continuous(labels = label_currency(prefix = "$")) +
  coord_cartesian(ylim = c(0, 4000000)) +
  theme(legend.position = "none")
 
# Prices by Housing type
data %>%
  ggplot(aes(x = factor(Type), y = Price, color = factor(Type))) +
  geom_boxplot(outlier.colour = NA) +
  labs(title = "Prices by Housing Type") +
  theme_fivethirtyeight() +
  scale_y_continuous(labels = label_currency(prefix = "$")) +
  scale_x_discrete(labels = c("h" = "House", "t" = "Townhouse", "u" = "Unit")) +
  coord_cartesian(ylim = c(0, 4000000)) +
  theme(legend.position = "none")

# Average property prices by location
data %>%
  mutate(Longtitude = round(Longtitude, 2),
         Lattitude = round(Lattitude, 2)) %>%
  group_by(Longtitude, Lattitude) %>%
  summarise(AveragePrice = mean(Price), .groups = 'drop') %>%
  ggplot(aes(x = Longtitude, y = Lattitude)) +
  geom_tile(aes(fill = AveragePrice), color = "white") +
  scale_fill_gradient(low = "blue", high = "red", labels = label_currency(prefix = "$")) + 
  labs(title = "Average Property Prices in Melbourne by Location",
       x = "Longitude", y = "Latitude", fill = "Average Price") +
  theme_minimal()

# Price outliers analysis
data %>% 
  filter(Price > 7000000) %>% 
  select(Price, Suburb, Rooms, Distance, Bedroom2, Bathroom,
         YearBuilt, Landsize) %>% 
  arrange(-Price)

# Data filtering for modeling
data_filter <- data %>% 
  select(Suburb, Rooms:SellerG, Distance, Bedroom2:Landsize)

### Modeling

# Splitting data into training and testing sets
set.seed(123)
split <- initial_split(data_filter, strata = Price, prop = 0.8)
train <- training(split)
test <- testing(split)

# Cross-validation
set.seed(123)
house_folds <- vfold_cv(train, strata = Price)

# XGBoost model specification
xgb_spec <- boost_tree(
  trees = 1000,
  tree_depth = tune(), min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune(), mtry = tune(),
  learn_rate = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

# Hyperparameter grid
xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), train),
  learn_rate(),
  size = 30
)

# Workflow
xgb_wf <- workflow() %>%
  add_formula(Price ~ .) %>%
  add_model(xgb_spec) 

# Tuning the model with cross-validation
registerDoParallel()
set.seed(123)
xgb_res <- tune_grid(
  xgb_wf,
  resamples = house_folds,
  grid = xgb_grid,
  control = control_grid(save_pred = TRUE)
)

# Best result and parameter selection
show_best(xgb_res, metric = "rmse")
best_rmse <- select_best(xgb_res, metric = "rmse")

# Finalizing workflow
best_rmse <- select_best(xgb_res, metric = "rmse")
final_xgb <- finalize_workflow(
  xgb_wf,
  best_rmse)

# Fitting the final model
final_xgb %>%
  fit(data = train) %>%
  extract_fit_parsnip() %>%
  vip(geom = "point")

# Results on the test set
final_res <- last_fit(final_xgb, split)
collect_predictions(final_res) %>% metrics(truth = Price, estimate = .pred)

# Scatter plot of actual vs predicted prices
test_predictions <- collect_predictions(final_res)
ggplot(test_predictions, aes(x = Price, y = .pred)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red", size = 1) +
  labs(title = "Actual vs. Predicted Prices",
       x = "Actual Price",
       y = "Predicted Price") +
  theme_fivethirtyeight() +
  scale_x_continuous(labels = label_currency(prefix = '$')) +
  scale_y_continuous(labels = label_currency(prefix = '$')) +
  geom_density_2d()

