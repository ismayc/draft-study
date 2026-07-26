# Draft model — independent R implementation (tidyverse).
# Same definitions as ../python/02_model.py; 03_reconcile.py compares the
# outputs numerically. Run: Rscript R/02_model.R (after python/01_harvest.py)

suppressPackageStartupMessages(library(tidyverse))

root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=",
  commandArgs(trailingOnly = FALSE), value = TRUE))), ".."))
source(file.path(root, "R", "functions.R"))

tv <- readr::read_csv(file.path(root, "data", "torvik_2011_2022.csv"),
                      col_types = readr::cols(.default = "c"))

final <- tv |>
  filter(pick != "", !is.na(pick)) |>
  mutate(year = as.integer(year)) |>
  arrange(year) |>
  group_by(pid) |>
  slice_tail(n = 1) |>
  ungroup() |>
  mutate(name_n = apply_aliases(norm_name(player_name)))

dh <- readr::read_csv(file.path(root, "data", "draft_history.csv"),
                      show_col_types = FALSE) |>
  filter(between(SEASON, 2011, 2022)) |>
  mutate(name_n = norm_name(PLAYER_NAME))

by_name <- final |>
  # many-to-many is expected: cross-class name collisions (two players, same
  # normalized name) are resolved by the year == SEASON filter below.
  inner_join(dh |> select(name_n, PERSON_ID, SEASON, OVERALL_PICK),
             by = "name_n", relationship = "many-to-many")
joined <- by_name |> filter(year == SEASON)

min_files <- list.files(file.path(root, "data", "nba_min"),
                        full.names = TRUE, pattern = "^min_")
mins <- purrr::map_dfr(min_files, function(p) {
  readr::read_csv(p, show_col_types = FALSE) |>
    mutate(start_year = as.integer(substr(sub("min_", "", basename(p)), 1, 4)))
})

first3 <- joined |>
  select(PERSON_ID, SEASON) |>
  left_join(mins |> rename(PERSON_ID = PLAYER_ID), by = "PERSON_ID") |>
  filter(start_year >= SEASON, start_year <= SEASON + 2) |>
  group_by(PERSON_ID) |>
  summarise(min3 = sum(MIN), .groups = "drop")

df <- joined |>
  left_join(first3, by = "PERSON_ID") |>
  mutate(min3 = replace_na(min3, 0),
         rotation = as.numeric(min3 >= ROTATION_MIN),
         log_pick = log(as.numeric(OVERALL_PICK)),
         ht_in = ht_inches(ht),
         age = age_at_draft(dob, SEASON),
         class_num = as.numeric(CLASS_NUM[yr]),
         three_a_pg = as.numeric(three_a) / as.numeric(g),
         across(c(obpm, dbpm, usg, ts_per, orb_per, drb_per, ast_per,
                  to_per, blk_per, stl_per, ftr), as.numeric)) |>
  drop_na(all_of(FEATURES))

cat(sprintf("%d drafted college players modeled (base rate %.3f)\n",
            nrow(df), mean(df$rotation)))

fit_and_score <- function(df, features) {
  train <- df |> filter(SEASON %in% TRAIN_YEARS)
  test <- df |> filter(SEASON %in% TEST_YEARS)
  Xtr <- cbind(1, as.matrix(train[features]))
  Xte <- cbind(1, as.matrix(test[features]))
  beta <- logistic_irls(Xtr, train$rotation)
  ptr <- predict_prob(Xtr, beta)
  pte <- predict_prob(Xte, beta)
  list(beta = beta, names = c("intercept", features),
       train_auc = auc(ptr, train$rotation), test_auc = auc(pte, test$rotation),
       train_logloss = log_loss(ptr, train$rotation),
       test_logloss = log_loss(pte, test$rotation),
       pred_test = pte, y_test = test$rotation)
}

m0 <- fit_and_score(df, "log_pick")
m1 <- fit_and_score(df, FEATURES)

out <- file.path(root, "output")
tibble(term = m0$names, coef = m0$beta) |>
  readr::write_csv(file.path(out, "coefficients_m0_r.csv"))
tibble(term = m1$names, coef = m1$beta) |>
  readr::write_csv(file.path(out, "coefficients_m1_r.csv"))
tibble(model = c("pick_only", "pick_plus_college"),
       train_auc = c(m0$train_auc, m1$train_auc),
       test_auc = c(m0$test_auc, m1$test_auc),
       train_logloss = c(m0$train_logloss, m1$train_logloss),
       test_logloss = c(m0$test_logloss, m1$test_logloss)) |>
  readr::write_csv(file.path(out, "metrics_r.csv"))
quintile_calibration(m1$pred_test, m1$y_test) |>
  readr::write_csv(file.path(out, "calibration_test_r.csv"))

# ggplot mirror of the pick curve
picks <- 1:60
curve <- predict_prob(cbind(1, log(picks)), m0$beta)
ggplot(tibble(pick = picks, p = curve), aes(pick, p)) +
  geom_line(colour = "#1E3A5F", linewidth = 1) +
  labs(title = "What a draft slot is worth: P(2,000+ min in 3 seasons) by pick",
       x = "overall pick", y = "P(rotation)") +
  theme_minimal()
ggsave(file.path(root, "figures", "fig1_pick_curve_r.png"),
       width = 7, height = 5, dpi = 150)

cat("R outputs written\n")
