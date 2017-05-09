## ------------------------------------------------------------------------
load_series <- function(filename){
  data <- read.csv(filename, header=F)[,2]
  normalized_data <- scale(data)
  return(normalized_data)
}

## ------------------------------------------------------------------------
split_data <- function(data, percent_train=0.8){
  num_rows <- length(data)
  train_data <- c()
  test_data <- c()
  for(idx in 1:num_rows){
    if(idx <= num_rows * percent_train){
      train_data <- append(train_data, data[idx])
    }else{
      test_data <- append(test_data, data[idx])
    }
  }
  return(list(train_data, test_data))
}


## ------------------------------------------------------------------------
# https://datamarket.com/data/set/22u3/international-airline-passengers-monthly-totals-in-thousands-jan-49-dec-60#!ds=22u3&display=line
#timeseries <- load_series('international-airline-passengers.csv')
#print(dim(timeseries))
    
#plot(ts(timeseries),col='blue')

