
show_conv_results <- function(data, filename=NULL){
  if(!is.null(filename)){
    png(filename)
  }
  par(mfrow=c(4,8), mar = rep(1, 4))
  for(i in 1:dim(data)[4]){
    img <- data[,,,i]
    image(img, axes = F, col = grey(seq(0, 1, length = 256)))
  }
  if(!is.null(filename)){
    dev.off()
  }
}

show_weights <- function(W, filename=NULL){
  if(!is.null(filename)){
    png(filename)
  }
  par(mfrow=c(4,8), mar = rep(1, 4))
  for(i in 1:dim(W)[4]){
    img <- W[,,,i]
    image(img, axes = F, col = grey(seq(0, 1, length = 256)))
  }
  if(!is.null(filename)){
    dev.off()
  }
}

