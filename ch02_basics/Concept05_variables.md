Ch 02: Concept 05
================

Using variables
===============

Here we go, here we go, here we go! Moving on from those simple examples, let's get a better understanding of variables. Start with a session:

``` r
library(tensorflow)

sess <- tf$InteractiveSession()
```

Below is a series of numbers. Don't worry what they mean. Just for fun, let's think of them as neural activations.

``` r
raw_data <- c(1., 2., 8., -1., 0., 5.5, 6., 13)
```

Create a boolean variable called spike to detect a sudden increase in the values.

All variables must be initialized. Go ahead and initialize the variable by calling run() on its initializer:

``` r
spike <- tf$Variable(FALSE)
spike$initializer$run()
```

Loop through the data and update the spike variable when there is a significant increase:

``` r
for(i in 2:length(raw_data)){
    if(raw_data[i] - raw_data[i-1] > 5){
        updater <- tf$assign(spike, tf$constant(TRUE))
        updater$eval()
    }else{
        tf$assign(spike, FALSE)$eval()
    }
  print(paste("Spike", spike$eval()))
}
```

    ## [1] "Spike FALSE"
    ## [1] "Spike TRUE"
    ## [1] "Spike FALSE"
    ## [1] "Spike FALSE"
    ## [1] "Spike TRUE"
    ## [1] "Spike FALSE"
    ## [1] "Spike TRUE"

``` r
sess$close()
```
