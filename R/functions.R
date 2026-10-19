# Pure functions for the draft model, extracted so both the pipeline
# (02_model.R) and the test suite (../../tests/R) can source them.
# Mirrors ../python/02_model.py definition-for-definition.

TRAIN_YEARS <- 2011:2018
TEST_YEARS <- 2019:2022
ROTATION_MIN <- 2000
CLASS_NUM <- c(Fr = 1, So = 2, Jr = 3, Sr = 4)

FEATURES <- c("log_pick", "age", "ht_in", "obpm", "dbpm", "usg", "ts_per",
              "orb_per", "drb_per", "ast_per", "to_per", "blk_per", "stl_per",
              "three_a_pg", "ftr", "class_num")

# Same committed alias table as the Python side (data/draft_name_aliases.csv,
# normalized names, one reason per row). A row with a torvik_year also admits
# that one Torvik (name, year) row despite a blank pick column.
# Resolved from this file's own location (source() records it as `ofile`),
# so sourcing from another repo, as basketball-analysis-tools' cross-study
# test does, still finds draft-study's table; `root` or `REPO` is the
# fallback when the file is not sourced from disk.
.this_dir <- local({
  d <- NULL
  for (i in rev(seq_len(sys.nframe()))) {
    f <- sys.frame(i)$ofile
    if (!is.null(f)) { d <- dirname(normalizePath(f)); break }
  }
  d
})
.repo_root <- function() {
  if (!is.null(.this_dir)) dirname(.this_dir) else if (exists("root")) root else REPO
}
.alias_tbl <- readr::read_csv(
  file.path(.repo_root(), "data", "draft_name_aliases.csv"),
  col_types = readr::cols(.default = "c"))
NICKNAME_ALIASES <- setNames(.alias_tbl$dh_name_n, .alias_tbl$torvik_name_n)
ADMIT_BLANK_PICK <- with(.alias_tbl[!is.na(.alias_tbl$torvik_year), ],
                         paste(torvik_name_n, torvik_year, sep = "|"))

#' Fold case/diacritics, drop punctuation and generational suffixes.
norm_name <- function(name) {
  s <- tolower(iconv(name, from = "UTF-8", to = "ASCII//TRANSLIT"))
  s <- gsub("[^a-z ]", "", s)
  vapply(strsplit(s, " +"), function(parts) {
    parts <- parts[!parts %in% c("jr", "sr", "ii", "iii", "iv", "v", "")]
    paste(parts, collapse = " ")
  }, character(1))
}

apply_aliases <- function(name_n) {
  hit <- name_n %in% names(NICKNAME_ALIASES)
  name_n[hit] <- NICKNAME_ALIASES[name_n[hit]]
  name_n
}

#' '6-10' -> 82; NA when unparseable.
ht_inches <- function(ht) {
  parts <- strsplit(ht, "-", fixed = TRUE)
  vapply(parts, function(p) {
    if (length(p) != 2) return(NA_real_)
    v <- suppressWarnings(as.numeric(p))
    if (anyNA(v)) NA_real_ else v[1] * 12 + v[2]
  }, numeric(1))
}

#' Age in years on draft night (approximated as June 25 of draft year).
age_at_draft <- function(dob, draft_year) {
  parts <- strsplit(dob, "-", fixed = TRUE)
  vapply(seq_along(parts), function(i) {
    p <- suppressWarnings(as.numeric(parts[[i]]))
    if (length(p) != 3 || anyNA(p)) return(NA_real_)
    (draft_year[i] - p[1]) + (6 - p[2]) / 12 + (25 - p[3]) / 365
  }, numeric(1))
}

#' Newton/IRLS logistic regression, identical to the Python implementation.
logistic_irls <- function(X, y, ridge = 1e-8, tol = 1e-10, max_iter = 50) {
  beta <- rep(0, ncol(X))
  for (i in seq_len(max_iter)) {
    p <- 1 / (1 + exp(-as.vector(X %*% beta)))
    w <- p * (1 - p)
    H <- crossprod(X, X * w) + diag(ridge, ncol(X))
    step <- solve(H, crossprod(X, y - p))
    beta <- beta + as.vector(step)
    if (max(abs(step)) < tol) break
  }
  beta
}

predict_prob <- function(X, beta) 1 / (1 + exp(-as.vector(X %*% beta)))

#' Mann-Whitney AUC with average ranks for ties (rank() default).
auc <- function(pred, y) {
  r <- rank(pred)
  n1 <- sum(y)
  n0 <- length(y) - n1
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n0 * n1)
}

log_loss <- function(pred, y) {
  p <- pmin(pmax(pred, 1e-12), 1 - 1e-12)
  -mean(y * log(p) + (1 - y) * log(1 - p))
}

#' Mean predicted vs actual by prediction quintile (stable ties as in Python).
quintile_calibration <- function(pred, y) {
  n <- length(pred)
  ord <- order(round(pred, 10))
  q <- integer(n)
  q[ord] <- ((seq_len(n) - 1L) * 5L) %/% n
  tibble::tibble(quintile = q + 1L, pred = pred, y = y) |>
    dplyr::group_by(quintile) |>
    dplyr::summarise(n = dplyr::n(), mean_pred = mean(pred),
                     actual = mean(y), .groups = "drop") |>
    dplyr::arrange(quintile)
}
