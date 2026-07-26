# Unit tests for draft-study/R/functions.R

source(file.path(REPO, "R", "functions.R"))

test_that("name normalization folds case, diacritics, and suffixes", {
  expect_equal(norm_name("Gary Trent Jr."), "gary trent")
  expect_equal(norm_name("Nah'Shon Hyland"), "nahshon hyland")
  expect_equal(norm_name("Wendell Carter III"), "wendell carter")
  expect_equal(norm_name("Dario Šarić"), "dario saric")
})

test_that("alias table maps formal names to NBA names", {
  expect_equal(apply_aliases(c("edrice adebayo", "someone else")),
               c("bam adebayo", "someone else"))
})

test_that("height parsing", {
  expect_equal(ht_inches(c("6-10", "7-4")), c(82, 88))
  expect_true(is.na(ht_inches("tall")))
})

test_that("age at draft", {
  expect_equal(age_at_draft("2000-06-25", 2019), 19)
  expect_true(is.na(age_at_draft("", 2019)))
})

test_that("AUC known values, including all-tied = 0.5", {
  y <- c(0, 0, 1, 1)
  expect_equal(auc(c(0.1, 0.2, 0.8, 0.9), y), 1)
  expect_equal(auc(c(0.9, 0.8, 0.2, 0.1), y), 0)
  expect_equal(auc(rep(0.5, 4), y), 0.5)
})

test_that("log loss matches the closed form", {
  expect_equal(log_loss(c(0.9, 0.1), c(1, 0)), -log(0.9))
})

