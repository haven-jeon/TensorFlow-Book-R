Ch 07: Concept 01
================

Autoencoder
===========

All we'll need is TensorFlow and NumPy:

``` r
library(tensorflow)
```

Instead of feeding all the training data to the training op, we will feed data in small batches:

``` r
get_batch <- function(X, size){
  a <- sample(length(X), size)
  return(X[a])
}
```

Define the autoencoder class:

``` r
Autoencoder <- setRefClass("Autoencoder",
  fields = c('input_dim', 'hidden_dim', 'epoch', 'batch_size', 
             'learning_rate', 'x', 'encoded', 'decoded', 'loss', 'all_loss',
             'train_op','saver'),
  methods=list(
    initialize=function(input_dim, hidden_dim, epoch=500, batch_size=10, learning_rate=0.001){
      .self$epoch <- as.integer(epoch)
      .self$batch_size <- as.integer(batch_size)
      .self$learning_rate <- learning_rate
      .self$input_dim <- as.integer(input_dim)
      .self$hidden_dim <- as.integer(hidden_dim)
      
      .self$x <- tf$placeholder(dtype=tf$float32, shape=list(NULL, .self$input_dim))
      
      with(tf$name_scope('encode'), {
            weights <- tf$Variable(tf$random_normal(list(.self$input_dim, .self$hidden_dim),
                                                    dtype=tf$float32), name='weights')
            biases <- tf$Variable(tf$zeros(list(.self$hidden_dim)), name='biases')
            .self$encoded <- tf$nn$sigmoid(tf$matmul(x, weights) + biases)
      })
      
      with(tf$name_scope('decode'),{
            weights <- tf$Variable(tf$random_normal(list(.self$hidden_dim, .self$input_dim),
                                                    dtype=tf$float32), name='weights')
            biases <- tf$Variable(tf$zeros(list(.self$input_dim)), name='biases')
            .self$decoded <- tf$matmul(encoded, weights) + biases
      })
      
      .self$x <- x
      .self$encoded <- encoded
      .self$decoded <- decoded

      # Define cost function and training op
      .self$loss <- tf$sqrt(tf$reduce_mean(tf$square(tf$subtract(.self$x, .self$decoded))))

      .self$all_loss <- tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.x, self.decoded)), 1))
      .self$train_op <- tf$train$AdamOptimizer(.self$learning_rate)$minimize(.self$loss)
        
      # Define a saver op
      .self$saver = tf$train.Saver()
      },
    train=function(){
      with(tf$Session() %as% sess,{
        sess$run(tf$global_variables_initializer())
        for(i in 1:.self$epoch){
          for(j in 1:500){
            batch_data <- get_batch(data, .self$batch_size)
            l_ <- sess$run(list(.self$loss, .self$train_op), feed_dict=dict(x=batch_data))
          }
        }
        })
      },
    test=function(data){
      with(tf$Session() %as% sess, {
        .self$saver$restore(sess, './model.chkpt')
        hidden_reconstructed <- sess$run(list(.self$encoded, .self$decoded), feed_dict=dict(x=data))
        })
      print(paste('input', data))
      print(paste('compressed', hidden_reconstructed[[1]]))
      print(paste('reconstructed', hidden_reconstructed[[2]]))
      return(hidden_reconstructed[[2]])
      },
    get_params=function(){
      with(tf$Session() %as% sess, {
        .self$saver$restore(sess, './model.chkpt')
        weight_biases <- sess$run(list(.self$weight1, .self$biases1))
      })
      return(list(weight=weight_biases[[1]], biases=weight_biases[[2]]))
      },
    classify=function(data, labels){
      with(tf$Session() %as% sess, {
        sess$run(tf$global_variables_initializer())
        .self$saver$restore(sess, './model.chkpt')
        hidden_reconstructed <- sess$run(list(.self$encoded, .self$decoded), feed_dict=dict(x=data))
        reconstructed <- hidden_reconstructed[[2]][1]
        # loss <- sess$run(.self$all_loss, feed_dict=dict(x: data))
        print(paste('data', dim(data)))
        print(paste('reconstructed', dim(hidden_reconstructed[[2]])))
        .self$loss <- sqrt(mean((data - hidden_reconstructed[[2]])^2))
        print(paste('loss', dim(loss)))
        horse_indices <- where(labels == 7)
        not_horse_indices <- where(labels != 7)
        horse_loss <- mean(loss[horse_indices])
        not_horse_loss <- mean(loss[not_horse_indices])
        print(paste('horse', horse_loss))
        print(paste('not horse', not_horse_loss))
      })
      return(hidden_reconstructed[[1]][7,])
      },
    decode=function(encoding){
      with(tf$Session() %as% sess, {
            sess$run(tf$global_variables_initializer())
            .self$saver$restore(sess, './model.ckpt')
            reconstructed <- sess.run(self.decoded, feed_dict=dict(encoded= encoding))
      })
      img <- matrix(reconstructed,byrow = T, nrow = 32)
      }
    ))
```
