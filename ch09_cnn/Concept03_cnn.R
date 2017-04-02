## ---- cache=T------------------------------------------------------------
# source("Concept01_cifar.R")

learning_rate <- 0.001

# names_data_labels <- read_data('./cifar-10-batches-py')

## ------------------------------------------------------------------------
library(tensorflow)
# library(data.table)

x <- tf$placeholder(tf$float32, list(NULL, as.integer(24 * 24)))
y <- tf$placeholder(tf$float32, list(NULL, 10L))

W1 <- tf$Variable(tf$random_normal(list(5L, 5L, 1L, 64L)))
b1 <- tf$Variable(tf$random_normal(list(64L)))

W2 <- tf$Variable(tf$random_normal(list(5L, 5L, 64L, 64L)))
b2 <- tf$Variable(tf$random_normal(list(64L)))

W3 <- tf$Variable(tf$random_normal(list(6L*6L*64L, 1024L)))
b3 <- tf$Variable(tf$random_normal(list(1024L)))

W_out <- tf$Variable(tf$random_normal(list(1024L, 10L)))
b_out <- tf$Variable(tf$random_normal(list(10L)))


## ------------------------------------------------------------------------
conv_layer <- function(x, W, b){
  conv <- tf$nn$conv2d(x, W, strides = list(1L, 1L, 1L, 1L), padding = 'SAME')
  conv_with_b <- tf$nn$bias_add(conv, b)
  conv_out <- tf$nn$relu(conv_with_b)
  return(conv_out)
}

maxpool_layer <- function(conv, k=2L){
  return(tf$nn$max_pool(conv, ksize=list(1L, k, k, 1L), strides=list(1L, k, k, 1L), padding='SAME'))
}


## ------------------------------------------------------------------------

model <- function(){
  #R의 방식과 다르다. 주의!
  x_reshaped <- tf$reshape(x, shape=list(-1L, 24L, 24L, 1L))
  conv_out1 <<- conv_layer(x_reshaped, W1, b1)
  maxpool_out1 <- maxpool_layer(conv_out1)
  norm1 <- tf$nn$lrn(maxpool_out1, 4L, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
  conv_out2 <<- conv_layer(norm1, W2, b2)
  norm2 <- tf$nn$lrn(conv_out2, 4L, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
  maxpool_out2 <- maxpool_layer(norm2)
  
  maxpool_reshaped <- tf$reshape(maxpool_out2, list(-1L, W3$get_shape()$as_list()[1]))
  local <- tf$add(tf$matmul(maxpool_reshaped, W3), b3)
  local_out <- tf$nn$relu(local)
  out <- tf$add(tf$matmul(local_out, W_out), b_out)
  return(out)  
}



## ------------------------------------------------------------------------

model_op <- model()

cost <- tf$reduce_mean(
    tf$nn$softmax_cross_entropy_with_logits(logits=model_op, labels=y)
)
train_op <- tf$train$AdamOptimizer(learning_rate=learning_rate)$minimize(cost)

correct_pred <- tf$equal(tf$argmax(model_op, 1L), tf$argmax(y, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_pred, tf$float32))
saver <- tf$train$Saver(max_to_keep=0L)


## ------------------------------------------------------------------------
# 
# accuracy_list <- list()
# 
# 
# fileConn<-file("output.txt",open = 'wt')
# 
# with(tf$Session() %as% sess,{
#     sess$run(tf$global_variables_initializer())
#     onehot_labels <- tf$one_hot(names_data_labels$labels, length(names_data_labels$names), 
#                                 on_value=1., off_value=0., axis=-1)
#     onehot_vals <- sess$run(onehot_labels)
#     
#     print(head(names_data_labels$labels))
#     print(head(onehot_vals))
#     
#     batch_size <- nrow(names_data_labels$data) %/% 200
#     print(sprintf('batch size %d', batch_size))
#     for(j in 1:1000){
#         avg_accuracy_val <- 0.
#         batch_count <- 0.
#         for(i in seq(1, nrow(names_data_labels$data), batch_size)){
#           if(i+batch_size > nrow(names_data_labels$data)){
#             max_idx <- nrow(names_data_labels$data)
#           }else{
#             max_idx <- i+batch_size 
#           }
#           batch_data <- names_data_labels$data[i:max_idx, ]
#           batch_onehot_vals <- onehot_vals[i:max_idx,]
#           accuracy_val_ = sess$run(list(train_op, accuracy), 
#                                    feed_dict=dict(x=batch_data, y=batch_onehot_vals))
#           avg_accuracy_val <- avg_accuracy_val + accuracy_val_[[2]]
#           batch_count <- batch_count +  1.
#         }
#         avg_accuracy_val <- avg_accuracy_val/batch_count
#         save_path <- saver$save(sess=sess, 
#                                 save_path = sprintf("saver/model_%s.chkp", j), global_step =1L)
#         accuracy_list[[j]] <- data.table(epoch=j, accu=avg_accuracy_val)
#         txt <- sprintf('%s Epoch %d. Avg accuracy %f',date(), j, avg_accuracy_val)
#         print(txt)
#         cat(paste(txt, "\n"), file = fileConn, append=T)
#     }
# })
# 
# save(accuracy_list, file="accuracy_list.RData")
# close(fileConn)
# 
