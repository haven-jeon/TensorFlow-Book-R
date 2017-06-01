## ------------------------------------------------------------------------
library(tensorflow)

## ------------------------------------------------------------------------
get_batch <- function(X, size){
  a <- sample(1:nrow(X), size)
  return(X[a,])
}


## ------------------------------------------------------------------------

library(R6)
Autoencoder <- R6Class("Autoencoder",
  public = list(input_dim=NULL, hidden_dim=NULL, epoch=NULL, batch_size=NULL, 
             learning_rate=NULL, x=NULL, encoded=NULL, decoded=NULL, loss=NULL, all_loss=NULL,
             train_op=NULL,saver=NULL, autoencoder_graph=NULL,
             
    initialize=function(input_dim, hidden_dim, epoch=500, batch_size=10, learning_rate=0.001){
      self$epoch <- as.integer(epoch)
      self$batch_size <- as.integer(batch_size)
      self$learning_rate <- learning_rate
      self$input_dim <- as.integer(input_dim)
      self$hidden_dim <- as.integer(hidden_dim)
      
      # make graph for avoiding restore error 
      self$autoencoder_graph <- tf$Graph()
      
      with(self$autoencoder_graph$as_default(), {
        self$x <- tf$placeholder(dtype=tf$float32, shape=list(NULL, self$input_dim))
        with(tf$name_scope('encode'), {
              encode_weights <- tf$Variable(tf$random_normal(list(self$input_dim, self$hidden_dim),
                                                      dtype=tf$float32), name='weights')
              encode_biases <- tf$Variable(tf$zeros(list(self$hidden_dim)), name='biases')
              self$encoded  <- tf$nn$sigmoid(tf$matmul(self$x, encode_weights) + encode_biases)
        })
        
        with(tf$name_scope('decode'),{
              decode_weights <- tf$Variable(tf$random_normal(list(self$hidden_dim, self$input_dim),
                                                      dtype=tf$float32), name='weights')
              decode_biases <- tf$Variable(tf$zeros(list(self$input_dim)), name='biases')
              self$decoded <- tf$matmul(self$encoded, decode_weights) + decode_biases
        })
  
        # Define cost function and training op
        self$loss <- tf$sqrt(tf$reduce_mean(tf$square(tf$subtract(self$x, self$decoded))))
  
        self$all_loss <- tf$sqrt(tf$reduce_mean(tf$square(tf$subtract(self$x, self$decoded)), 1L))
        self$train_op <- tf$train$AdamOptimizer(self$learning_rate)$minimize(self$loss)
          
        # Define a saver op
        self$saver <- tf$train$Saver()
      })
      },
    train=function(data){
      with(tf$Session(graph=self$autoencoder_graph) %as% sess,{
        sess$run(tf$global_variables_initializer())
        for(i in 1:self$epoch){
          for(j in 1:500){
            batch_data <- get_batch(data, self$batch_size)
            x <- self$x
            l_ <- sess$run(list(self$loss, self$train_op), feed_dict=dict(x=batch_data))
          }
          if(i %% 50 == 0){
            print(sprintf('epoch %d: loss = %f', i, l_[[1]]))
            self$saver$save(sess, './model.ckpt')
          }
        }
        self$saver$save(sess, './model.ckpt')
        })
      },
    test=function(data){
      with(tf$Session(graph=self$autoencoder_graph) %as% sess, {
        self$saver$restore(sess, './model.ckpt')
        x <- self$x
        hidden_reconstructed <- sess$run(list(self$encoded, self$decoded), feed_dict=dict(x=data))
        })
      print(paste('input', data))
      print(paste('compressed', hidden_reconstructed[[1]]))
      print(paste('reconstructed', hidden_reconstructed[[2]]))
      return(hidden_reconstructed[[2]])
      },
    get_params=function(){
      with(tf$Session(graph=self$autoencoder_graph) %as% sess, {
        self$saver$restore(sess, './model.ckpt')
        weight_biases <- sess$run(list(self$weight1, self$biases1))
      })
      return(list(weight=weight_biases[[1]], biases=weight_biases[[2]]))
      },
    classify=function(data, labels){
      with(tf$Session(graph=self$autoencoder_graph) %as% sess, {
        sess$run(tf$global_variables_initializer())
        self$saver$restore(sess, './model.ckpt')
        x <- self$x
        hidden_reconstructed <- sess$run(list(self$encoded, self$decoded), feed_dict=dict(x=data))
        reconstructed <- hidden_reconstructed[[2]]
        print(dim(reconstructed))
        # loss <- sess$run(self$all_loss, feed_dict=dict(x: data))
        print(paste('data', dim(data)))
        print(paste('reconstructed', dim(hidden_reconstructed[[2]])))
        loss_test <- apply((data - hidden_reconstructed[[2]])^2, 1, function(x){sqrt(mean(x))})
        print(paste('loss length', length(loss_test)))
        horse_indices <- which(labels == 7)
        not_horse_indices <- which(labels != 7)
        horse_loss <- mean(loss_test[horse_indices])
        not_horse_loss <- mean(loss_test[not_horse_indices])
        print(paste('horse', horse_loss))
        print(paste('not horse', not_horse_loss))
      })
      return(hidden_reconstructed[[1]])
      },
    decode=function(encoding){
      with(tf$Session(graph=self$autoencoder_graph) %as% sess, {
            sess$run(tf$global_variables_initializer())
            self$saver$restore(sess, './model.ckpt')
            encoded <- self$encoded
            reconstructed <- sess$run(self$decoded, feed_dict=dict(encoded= encoding))
      })
      img <- matrix(reconstructed,byrow = T, nrow = 32)
      return(img)
      }
    ))

