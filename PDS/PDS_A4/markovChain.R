# markovChain.R
#' MarkovChain R6 class
#'
#' Discrete-time, time-homogeneous Markov chain on a finite state space
#' \eqn{S = \{1,\dots,K\}} with transition matrix \eqn{P}.
#'
#' @import R6
#' @import igraph
#' @export
MarkovChain <- R6::R6Class(
  classname = "MarkovChain",
  public = list(
    #' @field initial_state Index in 1..K with the initial state.
    initial_state = NULL,
    
    #' @field P Quadratic transition matrix (K x K).
    P = NULL,
    
    #' @description
    #' Create a MarkovChain object.
    #'
    #'
    #' @param initial_state Integer in 1,...,K.
    #' @param P Quadratic transition matrix with positive entries; entries in [0,1]; row sums to 1.
    #' @return A new `MarkovChain` object.
    initialize = function(P, initial_state) {
      private$validate_P(P)
      K <- nrow(P)
      
      private$validate_positive_int(initial_state, "initial_state")
      if (initial_state > K) {
        stop(sprintf("initial_state must be in 1,...,%d.", K))
      }
      
      self$P <- P
      self$initial_state <- as.integer(initial_state)
      invisible(self) #Avoids verbose printing when creating object
    },
    
    #' @description
    #' Simulate - takes as inputs a number of steps n_steps and a number of distinct 
    #'  simulation paths n_paths, simulates n_path independent trajectories of 
    #'  length n_steps all starting from initial_state, returns a convenient 
    #'  R object with the simulation results
    #'
    #' @param n_steps Integer >= 1. Number of transitions steps to simulate.
    #' @param n_paths Integer >= 1. Number of distinct simulation paths.
    #' @param seed Optional integer - seed for reproducability.
    #' @return A list with elements:
    #' \itemize{
    #'   \item `paths`: Integer matrix of dimension (n_steps + 1) x n_paths; rows are time 0..n_steps.
    #'   \item `initial_state`: Initial state used.
    #'   \item `P`: Transition matrix used.
    #' }
    simulate = function(n_steps, n_paths, seed = NULL) {
      private$validate_positive_int(n_steps, "n_steps")
      private$validate_positive_int(n_paths, "n_paths")
      if (!is.null(seed)) {
        if (length(seed) != 1 || is.na(seed) || seed != as.integer(seed)) {
          stop("seed must be a single integer or ommited.")
        }
        set.seed(as.integer(seed))
      }
      
      P <- self$P
      K <- nrow(P)
      
      paths <- matrix(NA_integer_, nrow = n_steps + 1L, ncol = n_paths) #Rows are time index, columns are independent simulation path (states)
      paths[1L, ] <- self$initial_state #Set first row to initial_state (all paths start from same initial state)
      
      # Partially vectorised, loop through time and use vectorisation to calculate paths for each current state.
      for (t in 2L:(n_steps + 1L)) {
        prev <- paths[t - 1L, ]
        curr <- integer(n_paths)
        for (s in 1:K) {
          idx <- which(prev == s)
          if (length(idx) > 0L) {
            curr[idx] <- sample.int(K, size = length(idx), replace = TRUE, prob = P[s, ]) #Advance all paths in state s by one time step.
          }
        }
        paths[t, ] <- curr
      }
      
      list(paths = paths, initial_state = self$initial_state, P = self$P)
    },
    
    #' @description
    #' Marginal_distribution - takes as input a time t, returns the marginal 
    #' distribution at time t when starting from the initial distribution with 
    #' point mass at initial_state
    #'
    #' Uses \eqn{\pi_t = \pi_0 P^t} with \eqn{\pi_0} the vector with point mass at `initial_state`.
    #'
    #' @param t Integer >= 0.
    #' @return Numeric vector of length K summing to 1.
    marginal_distribution = function(t) {
      private$validate_nonneg_int(t, "t")
      
      t <- as.integer(t)
      K <- nrow(self$P)
      
      # Construct initial distribution vector - point mass at initial state
      pi <- rep(0, K)
      pi[self$initial_state] <- 1
      
      if (t == 0L) return(pi)
      
      #advance prob distributions using matrix multiplication
      for (i in seq_len(t)) {
        pi <- as.numeric(pi %*% self$P)
      }
      
      # Return the marginal distribution at time t.
      
      pi
    },
    
    #' @description
    #' empirical_marginal_distribution - Takes as inputs paths of a Markov chain 
    #' simulation (obtained from simulate) and returns the empirical 
    #' distribution at each of the time points (i.e. for each state at this 
    #' time point the relative frequency along the paths)
    
    
    #'
    #' 
    #'
    #' @param input_sim Takes as inputs paths of a Markov chain 
    #' simulation (obtained from simulate) - sim_res$paths
    #' @return Numeric matrix of dimension (T x K), where T = number of time points,
    #' and each row sums to 1.
    empirical_marginal_distribution = function(input_sim) {
      
      #defensive check for input_sim
      paths <- private$validate_sim_paths(input_sim)
      
      
      K <- nrow(self$P)
      T_n <- nrow(paths)
      
      #create the emperical matrix
      emp_res <- matrix(0, nrow = T_n, ncol = K)
      
      #loop over time, count state occurrences and calculate and save freq
      for (t in seq_len(T_n)) {
        tab <- tabulate(paths[t, ], nbins = K)
        emp_res[t, ] <- tab / ncol(paths)
      }
      colnames(emp_res) <- paste0("state_", seq_len(K))
      rownames(emp_res) <- paste0("t_", 0:(T_n - 1L))
      
      #return empirical distribution
      emp_res
    },
    
    #' @description
    #' stationary_distribution - takes as inputs a tolerance value tol and a 
    #' maximum number of iterations max_iter and returns an estimate of the 
    #' stationary distribution (assuming there exists one) by the power method 
    #' starting from the initial distribution with point mass at initial_state
    #'
    #'
    #' @param tol Numeric > 0. Defines the tolerance value.
    #' @param max_iter Integer >= 1. Max no of iterations.
    #' @return Numeric vector of length K summing to 1. Estimate of the stationary distribution.
    stationary_distribution = function(tol = 1e-10, max_iter = 10000L) {
      
      private$validate_nonneg_numeric(tol, "tol")
      private$validate_positive_int(max_iter, "max_iter")
      
      K <- nrow(self$P)
      pi_res <- rep(0, K)
      pi_res[self$initial_state] <- 1
      
      for (iter in seq_len(as.integer(max_iter))) {
        pi_next <- as.numeric(pi_res %*% self$P)
        
        # Return if stationary distribution converges.
        if (max(abs(pi_next - pi_res)) < tol) return(pi_next)
        
        
        pi_res <- pi_next
      }
      
      
      # Return last iteration if stationary distribution does not converge.
      warning("Stationary distribution did not converge - returning last iteration. Consider adjusting tolerance or max iterations.")
      pi_res
    },
    
    #' @description
    #' visualise - takes no inputs and prints the directed weighted graph 
    #' corresponding to the transition matrix using 
    #' igraph::graph_from_adjacency_matrix() from the igraph package and plots 
    #' it; edge labels reflect transition probabilities.
    #'
    #' @return No return value. Prints object and plots graph.
    visualise = function() {
      P <- self$P
      
      g <- igraph::graph_from_adjacency_matrix(
        P,
        mode = "directed",
        weighted = TRUE,
        diag = TRUE
      )
      
      igraph::E(g)$label <- formatC(igraph::E(g)$weight, format = "f", digits = 3)
      
      print(g)
      
      plot(
        g,
        edge.arrow.size = 0.4,
        edge.label = igraph::E(g)$label,
        vertex.label = igraph::V(g)$name
      )
      
      # Return nothing.
      invisible(NULL)
    }
  ),
  
  
  #Private helper functions for defensive checks.
  private = list(
    
    #Validates a Positive Integer >= 1
    validate_positive_int = function(x, name) {
      if (length(x) != 1 || is.na(x) || x != as.integer(x) || as.integer(x) < 1L) {
        stop(sprintf("%s must be a single integer >= 1.", name))
      }
      invisible(TRUE)
    },
    
    #Validates a non-negative integer >= 0.
    validate_nonneg_int = function(x, name) {
      if (length(x) != 1 || is.na(x) || x != as.integer(x) || x < 0L) {
        stop(sprintf("%s must be a single integer >= 0.", name))
      }
      invisible(TRUE)
    },
    
    #Validates a non-negative numeric value > 0.
    validate_nonneg_numeric = function(x, name) {
      if (length(x) != 1 || is.na(x) || !is.numeric(x) || name <= 0) {
        stop("%s must be a single numeric value > 0.", name)
      }
      invisible(TRUE)
    },
    
    #Validates P - the transition matrix.
    validate_P = function(P) {
      if (!is.matrix(P)) stop("P must be a matrix.")
      if (!is.numeric(P)) stop("P must be numeric.")
      if (anyNA(P)) stop("P must not contain NA values.")
      if (nrow(P) < 1L || ncol(P) < 1L) stop("P must have positive dimensions.")
      if (nrow(P) != ncol(P)) stop("P must be square.")
      if (any(P < 0) || any(P > 1)) stop("P entries must be in [0,1].")
      
      rs <- rowSums(P)
      if (any(!is.finite(rs))) stop("Row sums of P must be finite.")
      if (max(abs(rs - 1)) > 1e-12) stop("Each row of P must sum to 1 (within tolerance).")
      
      invisible(TRUE)
    },
    
    
    # Validates that input_sim is of the form of a simulation return objects path matrix.
    validate_sim_paths = function(input_sim) {
      
      # Must be a matrix
      if (!is.matrix(input_sim)) {
        stop("input_sim must be a matrix of simulated paths.")
      }
      
      # Must be numeric / integer-valued
      if (!is.numeric(input_sim) || anyNA(input_sim)) {
        stop("input_sim must contain numeric, non-missing values.")
      }
      if (any(input_sim != as.integer(input_sim))) {
        stop("input_sim must contain integer-valued state indices.")
      }
      
      paths <- matrix(as.integer(input_sim),
                      nrow = nrow(input_sim),
                      ncol = ncol(input_sim))
      
      # Dimensions
      if (nrow(paths) < 1L || ncol(paths) < 1L) {
        stop("input_sim must have positive dimensions.")
      }
      
      # State-space validity
      K <- nrow(self$P)
      if (any(paths < 1L) || any(paths > K)) {
        stop("input_sim contains state indices outside 1..K.")
      }
      
      paths
    }
  )
)
