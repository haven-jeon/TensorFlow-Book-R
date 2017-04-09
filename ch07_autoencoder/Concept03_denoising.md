Ch 07: Concept 03
================

Denoising autoencoder
=====================

A denoising autoencoder is pretty much the same architecture as a normal autoencoder. The input is noised up, and cost function tries to denoise it by minimizing the construction error from denoised input to clean output.

``` r
library(stringr)
library(reticulate)
py <- import_builtins()


get_batch_noise <- function(X,Xn,size){
  a <- sample(1:nrow(X), size)
  return(list(X[a,], Xn[a,]))
}



Denoiser <- setRefClass("Denoiser",
  fields = c('input_dim', 'hidden_dim', 'epoch', 'batch_size', 
             'learning_rate', 'x', 'encoded', 'decoded', 'loss', 'all_loss',
             'train_op','saver', 'denoiser_graph', 'x_noised', 'weights1', 'biases1'),
  methods=list(
    initialize=function(input_dim, hidden_dim, epoch=10000, batch_size=10, learning_rate=0.001){
      .self$epoch <- as.integer(epoch)
      .self$batch_size <- as.integer(batch_size)
      .self$learning_rate <- learning_rate
      .self$input_dim <- as.integer(input_dim)
      .self$hidden_dim <- as.integer(hidden_dim)
      
      # make graph for avoiding restore error 
      .self$denoiser_graph <- tf$Graph()
      
      with(.self$denoiser_graph$as_default(), {
        .self$x <- tf$placeholder(dtype=tf$float32, shape=list(NULL, .self$input_dim), name='x')
        .self$x_noised <- tf$placeholder(dtype=tf$float32, 
                                         shape=list(NULL, .self$input_dim), name='x_noised')
        
        with(tf$name_scope('encode'), {
              .self$weights1  <- tf$Variable(tf$random_normal(list(.self$input_dim, .self$hidden_dim),
                                                      dtype=tf$float32), name='weights')
              .self$biases1  <- tf$Variable(tf$zeros(list(.self$hidden_dim)), name='biases')
              .self$encoded  <- tf$nn$sigmoid(tf$matmul(x_noised, encode_weights) + encode_biases)
        })
        
        with(tf$name_scope('decode'),{
              decode_weights <- tf$Variable(tf$random_normal(list(.self$hidden_dim, .self$input_dim),
                                                      dtype=tf$float32), name='weights')
              decode_biases <- tf$Variable(tf$zeros(list(.self$input_dim)), name='biases')
              .self$decoded <- tf$matmul(encoded, decode_weights) + decode_biases
        })
        .self$loss <- tf$sqrt(tf$reduce_mean(tf$square(tf$subtract(.self$x, .self$decoded))))
        .self$train_op <- tf$train$AdamOptimizer(.self$learning_rate)$minimize(.self$loss)
        .self$saver = tf$train$Saver()
      })
      },
    add_noise=function(data){
      noise_type <- 'mask-0.2'
      if(noise_type == 'gaussian'){
        n <- array(rnorm(prod(dim(data)), 0, 0.1) , dim = dim(data))
        return(data + n)
      }
      if(str_detect(noise_type, "mask")){
        frac <- as.numeric(str_split("mask-0.2", '-')[[1]][2])
        temp <- data
        for(i in 1:nrow(temp)){
          n <- sample(1:dim(temp)[2], round(frac * dim(temp)[2]),replace = FALSE)
          temp[i,n] <- 0
        }
        return(temp)
      }
    },
    
    train=function(data){
      data_noised <- add_noise(data)
      with(py$open("log.csv", "w") %as% writer, {
        with(tf$Session(graph=.self$denoiser_graph) %as% sess, {
          sess$run(tf$global_variables_initializer())
                for(i in 1:.self$epoch){
                    for(j in 1:50){
                        batch_data_batch_data_noised <- get_batch_noise(data, data_noised, self.batch_size)
                        l_ <- sess$run(list(.self$loss, .self$train_op), 
                                       feed_dict=dict(x= batch_data, x_noised= batch_data_noised))
                    }
                    if(i %% 10 == 0){
                        print(sprintf('epoch %d: loss = %f', i, l_[[1]]))
                        .self$saver$save(sess, './model.ckpt')
                        epoch_time <- date()
                        row_str <- paste(epoch_time , ',',  i ,  ',' ,l_[[1]] , '\n')
                        writer$write(row_str)
                        writer$flush()
                    }
                }
                .self$saver$save(sess, './model.ckpt')
        })
      })
    },
    test=function(data){
        with(tf$Session(graph=.self$denoiser_graph) %as% sess,{
            .self$saver$restore(sess, './model.ckpt')
            hidden_reconstructed <- sess$run(list(.self$encoded, .self$decoded), feed_dict=dict(x= data))
        })
        print(paste('input', data))
        print(paste('compressed', hidden_reconstructed[[1]]))
        print(paste('reconstructed', hidden_reconstructed[[2]]))
        return(hidden_reconstructed[[2]])
    },
    
    get_params=function(){
      with(tf$Session(graph=.self$denoiser_graph) %as% sess,{
        .self$saver$restore(sess, './model.ckpt')
        weights_biases <- sess.run(list(.self$weights1, .self$biases1))
      })
      return(weights_biases)
    }
    ))
```
