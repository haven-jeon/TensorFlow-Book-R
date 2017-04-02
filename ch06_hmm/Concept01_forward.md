Ch 06: Concept 01
================

Hidden Markov model forward algorithm
=====================================

Oof this code's a bit complicated if you don't already know how HMMs work. Please see the book chapter for step-by-step explanations. I'll try to improve the documentation, or feel free to send a pull request with your own documentation!

First, let's import TensorFlow and assertthat:

``` r
library(tensorflow)
library(assertthat)
```

Define the HMM model:

``` r
HMM <- setRefClass("HMM", 
          fields=c('N', 'initial_prob',
                   'trans_prob', 'emission', 'obs_idx', 'fwd'),
          methods=list(
            initialize=function(initial_prob, trans_prob, obs_prob){
              .self$N <- nrow(initial_prob)
              .self$initial_prob <- initial_prob
              .self$trans_prob <- trans_prob
              .self$emission <- tf$constant(obs_prob)
              
              assert_that(all(dim(.self$initial_prob) == c(.self$N, 1)))
              assert_that(all(dim(.self$trans_prob) == c(.self$N, .self$N)))
              assert_that(dim(obs_prob)[1] == .self$N)
              
              .self$obs_idx <- tf$placeholder(tf$int32)
              .self$fwd     <- tf$placeholder(tf$float64)
              
            }, 
            get_emission=function(obs_idx){
              slice_location <- list(0L, obs_idx)
              num_rows <- tf$shape(.self$emission)[0]
              slice_shape <- list(num_rows, 1L)
              return(tf$slice(.self$emission, slice_location, slice_shape))
            },
            forward_init_op=function(){
              obs_prob <- get_emission(.self$obs_idx)
              fwd_l <- tf$multiply(.self$initial_prob, obs_prob)
              return(fwd_l)
            },
            forward_op=function(){
              transitions <- tf$matmul(.self$fwd, tf$transpose(get_emission(.self$obs_idx)))
              weighted_transitions <- transitions * .self$trans_prob
              fwd_l <- tf$reduce_sum(weighted_transitions, 0L)
              return(tf$reshape(fwd_l, tf$shape(.self$fwd)))
            }
        )
)
```

Define the forward algorithm:

``` r
forward_algorithm <- function(sess, hmm, observations){
  obs_idx_hmm <- hmm$obs_idx
  fwd_hmm<- hmm$fwd 
  fwd <- sess$run(hmm$forward_init_op(), feed_dict=dict(obs_idx_hmm= observations[1]))
  for(t in 2:length(observations)){
    fwd <- sess$run(hmm$forward_op(), feed_dict=dict(obs_idx_hmm= observations[t], fwd_hmm= fwd))
  }
  prob <- sess$run(tf$reduce_sum(fwd))
  return(prob)
}
```

Let's try it out:

``` r
initial_prob <- matrix(c(0.6, 0.4),nrow = 2)
trans_prob <- matrix(c(0.7,0.3,0.4,0.6), nrow = 2, byrow = T)
obs_prob <- matrix(c(0.5, 0.4, 0.1, 0.1, 0.3, 0.6), nrow=2, byrow = T)

hmm <- HMM$new(initial_prob= initial_prob, trans_prob=trans_prob, obs_prob=obs_prob)

observations = c(0, 1, 1, 2, 1)
with(tf$Session() %as% sess, {
  prob <- forward_algorithm(sess, hmm, observations)
  print(sprintf('Probability of observing %s is %s',paste(observations, collapse = ", "), prob))
})
```

    ## [1] "Probability of observing 0, 1, 1, 2, 1 is 0.0046421488"
