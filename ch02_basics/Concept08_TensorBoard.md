Ch 02: Concept 08
================

Using TensorBoard
=================

TensorBoard is a great way to visualize what's happening behind the code.

In this example, we'll loop through some numbers to improve our guess of the average value. Then we can visualize the results on TensorBoard.

Let's just set ourselves up with some data to work with:

``` r
library(tensorflow)

raw_data <- rnorm(n = 100, mean = 10, sd = 1)
```

The moving average is defined as follows:

``` r
alpha <- tf$constant(0.05)
curr_value <- tf$placeholder(tf$float32)
prev_avg <- tf$Variable(0.0)

update_avg <- tf$multiply(alpha , curr_value) + tf$multiply((1 - alpha),  prev_avg)
```

Here's what we care to visualize:

``` r
avg_hist <- tf$summary$scalar("running_average", update_avg)
value_hist <- tf$summary$scalar("incoming_values", curr_value)

merged <- tf$summary$merge_all()
writer <- tf$summary$FileWriter("./logs")
```

Time to compute the moving averages. We'll also run the merged op to track how the values change:

``` r
init <- tf$global_variables_initializer()

with(tf$Session() %as% sess,{
    sess$run(init)
    for(i in 1:length(raw_data)){
        summary_str_curr_avg <- sess$run(list(merged, update_avg), feed_dict=dict(curr_value= raw_data[i]))
        sess$run(tf$assign(prev_avg, summary_str_curr_avg[[2]]))
        print(raw_data[i], summary_str_curr_avg[[2]])
        writer$add_summary(summary_str_curr_avg[[1]], i)
    }
})
```

    ## [1] 9
    ## [1] 12
    ## [1] 10
    ## [1] 11
    ## [1] 10
    ## [1] 10
    ## [1] 10.9
    ## [1] 10.5
    ## [1] 8.52
    ## [1] 9.086
    ## [1] 10.67
    ## [1] 10.77
    ## [1] 8.544
    ## [1] 9.05
    ## [1] 10.343
    ## [1] 8.1328
    ## [1] 8.8041
    ## [1] 10.09
    ## [1] 9.67421
    ## [1] 12.6141
    ## [1] 10.616
    ## [1] 9.58634
    ## [1] 10.6577
    ## [1] 8.144914
    ## [1] 10.49725
    ## [1] 8.671988
    ## [1] 10.14806
    ## [1] 12.0437
    ## [1] 11.62103
    ## [1] 9.880391
    ## [1] 10.066897
    ## [1] 11.15887
    ## [1] 9.0824674
    ## [1] 9.9634875
    ## [1] 9.9664862
    ## [1] 11.12229
    ## [1] 10.406994
    ## [1] 11.800467
    ## [1] 10.600642
    ## [1] 10.489026
    ## [1] 8.7676998
    ## [1] 11.1706194
    ## [1] 9.98861603
    ## [1] 11.0496917
    ## [1] 9.24225221
    ## [1] 9.88258022
    ## [1] 11.3398253
    ## [1] 10.8301641
    ## [1] 9.74208505
    ## [1] 8.32976201
    ## [1] 9.51207759
    ## [1] 9.96914536
    ## [1] 10.1359746
    ## [1] 9.69864071
    ## [1] 10.5692185
    ## [1] 9.01525484
    ## [1] 10.2298739
    ## [1] 8.66523742
    ## [1] 9.65314452
    ## [1] 9.64959423
    ## [1] 9.30210023
    ## [1] 10.0472626
    ## [1] 10.2028971
    ## [1] 9.42797585
    ## [1] 8.2648293
    ## [1] 10.3392788
    ## [1] 9.26113712
    ## [1] 11.112187
    ## [1] 9.85396252
    ## [1] 11.1730307
    ## [1] 10.0892227
    ## [1] 10.5185161
    ## [1] 10.7929991
    ## [1] 9.40467202
    ## [1] 9.26379603
    ## [1] 9.11675988
    ## [1] 10.5740041
    ## [1] 10.9558968
    ## [1] 10.9652205
    ## [1] 11.6082927
    ## [1] 9.47933576
    ## [1] 9.57941513
    ## [1] 9.50520077
    ## [1] 9.78511161
    ## [1] 11.393274
    ## [1] 10.80100475
    ## [1] 11.02137763
    ## [1] 9.207636037
    ## [1] 12.51502792
    ## [1] 10.99695912
    ## [1] 10.26385406
    ## [1] 10.3101392
    ## [1] 10.38742683
    ## [1] 9.891132156
    ## [1] 9.572757928
    ## [1] 9.686651927
    ## [1] 9.291348564
    ## [1] 8.728775071
    ## [1] 8.3661731
    ## [1] 8.33598726

Check out the visualization by running TensorBoard from the terminal:

$ tensorboard --logdir=path/to/logs
