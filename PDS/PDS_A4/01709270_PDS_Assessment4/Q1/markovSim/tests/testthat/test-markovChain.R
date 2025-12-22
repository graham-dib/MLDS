# test-markovChain.R

library(testthat)

#' MarkovChain class
#'
#' @description
#' Tests that the `MarkovChain` R6 class can be instantiated and that its
#' core fields are correctly set on construction.
test_that("MarkovChain class exists and can be instantiated", {
  P <- matrix(c(0.8, 0.2,
                0.1, 0.9), nrow = 2, byrow = TRUE)
  mc <- MarkovChain$new(P = P, initial_state = 1)
  
  expect_true(R6::is.R6(mc))
  expect_true(inherits(mc, "MarkovChain"))
  expect_equal(mc$initial_state, 1L)
  expect_equal(mc$P, P)
})

#' initialize()
#'
#' @description
#' Tests that `initialize()` correctly validates `P` and `initial_state`.
test_that("initialise validates P and initial_state", {
  P_good <- matrix(c(0.5, 0.5,
                     0.2, 0.8), nrow = 2, byrow = TRUE)
  
  P_nonsq <- matrix(c(0.5, 0.5, 0.0,
                      0.2, 0.8, 0.0), nrow = 2, byrow = TRUE)
  expect_error(MarkovChain$new(P = P_nonsq, initial_state = 1))
  
  P_bad_rowsum <- matrix(c(0.6, 0.6,
                           0.2, 0.8), nrow = 2, byrow = TRUE)
  expect_error(MarkovChain$new(P = P_bad_rowsum, initial_state = 1))
  
  P_bad_prob <- matrix(c(1.2, -0.2,
                         0.1, 0.9), nrow = 2, byrow = TRUE)
  expect_error(MarkovChain$new(P = P_bad_prob, initial_state = 1))
  
  expect_error(MarkovChain$new(P = P_good, initial_state = 0))
  expect_error(MarkovChain$new(P = P_good, initial_state = 3))
})

#' simulate()
#'
#' @description
#' Tests that `simulate()` returns a list containing a correctly sized
#' matrix, with all paths starting at `initial_state` and valid state values.
test_that("simulate returns correctly shaped paths and valid states", {
  P <- matrix(c(0.7, 0.3,
                0.4, 0.6), nrow = 2, byrow = TRUE)
  mc <- MarkovChain$new(P = P, initial_state = 2)
  
  sim <- mc$simulate(n_steps = 5, n_paths = 10, seed = 123)
  
  expect_true(is.list(sim))
  expect_true(is.matrix(sim$paths))
  expect_equal(dim(sim$paths), c(6L, 10L))
  expect_equal(sim$paths[1, ], rep(2L, 10L))
  expect_true(all(sim$paths %in% 1:2))
})

#' marginal_distribution()
#'
#' @description
#' Tests that `marginal_distribution()` returns the correct marginal
#' distributions for a simple deterministic Markov chain.
test_that("marginal_distribution returns correct distribution for simple cases", {
  P <- matrix(c(0, 1,
                1, 0), nrow = 2, byrow = TRUE)
  mc <- MarkovChain$new(P = P, initial_state = 1)
  
  expect_equal(mc$marginal_distribution(0), c(1, 0))
  expect_equal(mc$marginal_distribution(1), c(0, 1))
  expect_equal(mc$marginal_distribution(2), c(1, 0))
  
  expect_error(mc$marginal_distribution(-1))
})

#' empirical_marginal_distribution()
#'
#' @description
#' Tests that `empirical_marginal_distribution()` returns a valid empirical
#' distribution matrix whose rows sum to one and whose first row corresponds
#' to the initial state.
test_that("empirical_marginal_distribution returns a valid empirical distribution matrix", {
  P <- matrix(c(0.8, 0.2,
                0.1, 0.9), nrow = 2, byrow = TRUE)
  mc <- MarkovChain$new(P = P, initial_state = 1)
  
  sim <- mc$simulate(n_steps = 4, n_paths = 2000, seed = 42)
  emp <- mc$empirical_marginal_distribution(sim$paths)
  
  expect_true(is.matrix(emp))
  expect_equal(dim(emp), c(5L, 2L))
  expect_true(all(abs(rowSums(emp) - 1) < 1e-12))
  expect_equal(unname(emp[1, ]), c(1, 0), tolerance = 1e-12)
  
  expect_error(mc$empirical_marginal_distribution(list()))
})