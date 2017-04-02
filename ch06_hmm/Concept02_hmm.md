Ch 06: Concept 02
================

Viterbi parse of a Hidden Markov model
======================================

import TensorFlow and Numpy

``` r
library(tensorflow)
library(assertthat)
```

Create the same HMM model as before. This time, we'll include a couple additional functions.

``` r
# initial parameters can be learned on training data
# theory reference https://web.stanford.edu/~jurafsky/slp3/8.pdf
# code reference https://phvu.net/2013/12/06/sweet-implementation-of-viterbi-in-python/

HMM <- setRefClass("HMM", 
          fields=c('N', 'initial_prob',
                   'trans_prob', 'emission', 'obs', 'fwd', 'viterbi'),
          methods=list(
            initialize=function(initial_prob, trans_prob, obs_prob){
              .self$N <- nrow(initial_prob)
              .self$initial_prob <- initial_prob
              .self$trans_prob <- trans_prob
              .self$emission <- tf$constant(obs_prob)
              
              assert_that(all(dim(.self$initial_prob) == c(.self$N, 1)))
              assert_that(all(dim(.self$trans_prob) == c(.self$N, .self$N)))
              assert_that(dim(obs_prob)[1] == .self$N)
              
              .self$obs     <- tf$placeholder(tf$int32, name='obs')
              .self$fwd     <- tf$placeholder(tf$float64,name='fwd')
              .self$viterbi <- tf$placeholder(tf$float64, name='viterbi')
              
            }, 
            get_emission=function(obs_idx){
              slice_location <- list(0L, obs_idx)
              num_rows <- tf$shape(.self$emission)[0]
              slice_shape <- list(num_rows, 1L)
              return(tf$slice(.self$emission, slice_location, slice_shape))
            },
            forward_init_op=function(){
              obs_prob <- get_emission(.self$obs)
              fwd_l <- tf$multiply(.self$initial_prob, obs_prob)
              return(fwd_l)
            },
            forward_op=function(){
              transitions <- tf$matmul(.self$fwd, tf$transpose(get_emission(.self$obs)))
              weighted_transitions <- transitions * .self$trans_prob
              fwd_l <- tf$reduce_sum(weighted_transitions, 0L)
              return(tf$reshape(fwd_l, tf$shape(.self$fwd)))
            },
            decode_op=function(){
              transitions <- tf$matmul(.self$viterbi, tf$transpose(get_emission(.self$obs)))
              weighted_transitions <- transitions * .self$trans_prob
              viterbi_l <- tf$reduce_max(weighted_transitions, 0L)
              return(tf$reshape(viterbi_l, tf$shape(.self$viterbi)))
            },
            backpt_op=function(){
              back_transitions <- tf$matmul(.self$viterbi, tf$ones(list(1L, .self$N),dtype = tf$float64))
              weighted_back_transitions <- back_transitions * .self$trans_prob
              return(tf$argmax(weighted_back_transitions, 0L))
            }
        )
)
```

Define the forward algorithm from Concept01.

``` r
forward_algorithm <- function(sess, hmm, observations){
  obs_hmm <- hmm$obs
  fwd_hmm<- hmm$fwd 
  fwd <- sess$run(hmm$forward_init_op(), feed_dict=dict(obs_hmm= observations[1]))
  for(t in 2:length(observations)){
    fwd <- sess$run(hmm$forward_op(), feed_dict=dict(obs_hmm= observations[t], fwd_hmm= fwd))
  }
  prob <- sess$run(tf$reduce_sum(fwd))
  return(prob)
}
```

Now, let's compute the Viterbi likelihood of the observed sequence:

``` r
viterbi_decode <- function(sess, hmm, observations){
    obs_hmm <- hmm$obs
    viterbi_hmm <- hmm$viterbi
    viterbi <- sess$run(hmm$forward_init_op(), feed_dict=dict(obs_hmm= observations[1]))
    backpts <- matrix(rep(1L, hmm$N * length(observations)), nrow=hmm$N) * -1
    for(t in 2:length(observations)){
        viterbi_backpt <- sess$run(list(hmm$decode_op(), hmm$backpt_op()),
                                    feed_dict=dict(obs_hmm= observations[t],
                                               viterbi_hmm= viterbi))
        viterbi <- viterbi_backpt[[1]]
        backpt <- viterbi_backpt[[2]]
        backpts[,t] <-backpt + 1
        
    }
    tokens  <- which.max(viterbi[,ncol(viterbi)])
    for(i in length(observations):2){
        tokens <- c(tokens, backpts[tokens[length(tokens)], i])
    }
    return(rev(tokens))
}
```

Now, let's compute the Viterbi likelihood of the observed sequence:

``` r
initial_prob <- matrix(c(0.6, 0.4),nrow = 2)
trans_prob <- matrix(c(0.7,0.3,0.4,0.6), nrow = 2, byrow = T)
obs_prob <- matrix(c(0.5, 0.4, 0.1, 0.1, 0.3, 0.6), nrow=2, byrow = T)

hmm <- HMM$new(initial_prob= initial_prob, trans_prob=trans_prob, obs_prob=obs_prob)

observations = c(0, 1, 1, 2, 1)
with(tf$Session() %as% sess, {
  prob <- forward_algorithm(sess, hmm, observations)
  print(sprintf('Probability of observing %s is %s',paste(observations, collapse = ", "), prob))
  
  seq <- viterbi_decode(sess, hmm, observations)
  print(sprintf('Most likely hidden states are %s',paste(seq - 1, collapse = ", ")))
})
```

    ## [1] "Probability of observing 0, 1, 1, 2, 1 is 0.0046421488"
    ## [1] "Most likely hidden states are 0, 0, 0, 1, 1"
