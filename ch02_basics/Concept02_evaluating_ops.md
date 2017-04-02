Ch 02: Concept 02
================

Evaluating ops
==============

Import TensorFlow:

``` r
library(tensorflow)
```

Start with a 1x2 matrix:

``` r
x <- tf$constant(matrix(c(1, 2), nrow=1))
```

Let's negate it. Define the negation op to be run on the matrix:

``` r
neg_x <- tf$negative(x)
```

It's nothing special if you print it out. In fact, it doesn't even perform the negation computation. Check out what happens when you simply print it:

``` r
print(neg_x)
```

    ## Tensor("Neg:0", shape=(1, 2), dtype=float64)

You need to summon a session so you can launch the negation op:

``` r
with(tf$Session() %as% sess, {
    result <- sess$run(neg_x)
    print(result)
})
```

    ##      [,1] [,2]
    ## [1,]   -1   -2
