Ch 02: Concept 04
================

Session logging
===============

Define an op on a tensor. Here's an example:

``` r
library(tensorflow)

x <- tf$constant(matrix(c(1.0, 2.0), nrow=1, byrow=T))
neg_op <- tf$negative(x)
```

Now let's use a session with a special argument passed in.

``` r
with(tf$Session(config=tf$ConfigProto(log_device_placement=TRUE)) %as% sess, {
    result <- sess$run(neg_op)
    print(result)
})
```

    ##      [,1] [,2]
    ## [1,]   -1   -2
