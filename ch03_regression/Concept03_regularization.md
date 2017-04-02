Ch 03: Concept 03
================

Regularization
==============

Import the relevant libraries and initialize the hyper-parameters

``` r
library(tensorflow)

learning_rate <- 0.001
training_epochs <- 1000
reg_lambda <- 0.
```

create a helper method to split the dataset

``` r
split_dataset <- function(x_dataset, y_dataset, ratio){
  samp_idx <- sample(1:length(x_dataset), length(x_dataset) * ratio)
  x_train <- x_dataset[samp_idx]
  y_train <- y_dataset[samp_idx]
  x_test <- x_dataset[-samp_idx]
  y_test <- y_dataset[-samp_idx]
  return(list(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test))
}
```

Create a fake dataset. y = x^2

``` r
x_dataset <- seq.int(-1, 1, length.out = 101)

num_coeffs <- 9
y_dataset_params <- rep(0., num_coeffs) 
y_dataset_params[3] <- 1
y_dataset <- 0
for(i in 1:num_coeffs){
    y_dataset <- y_dataset +  y_dataset_params[i] * x_dataset^(i - 1)
}
y_dataset <-  y_dataset  + rnorm(length(y_dataset), mean = 0, sd=0.5) * 0.3
```

Split the dataset into 70% training and testing 30%

``` r
train_test_sets <- split_dataset(x_dataset, y_dataset, 0.7)
```

Set up the input/output placeholders

``` r
X <- tf$placeholder("float")
Y <- tf$placeholder("float")
```

Define our model

``` r
model <- function(X, w){
    terms <- list()
    for(i in 0:(num_coeffs-1)){
        term <- tf$multiply(w[i], tf$pow(X, i))
        terms[[i+1]] <- term 
    }
    return(tf$add_n(terms))
}
```

Define the regularized cost function

``` r
w <- tf$Variable(rep(0, num_coeffs), name="parameters")
y_model <- model(X, w)
cost <- tf$div(tf$add(tf$reduce_sum(tf$square(Y-y_model)),
                     tf$multiply(reg_lambda, tf$reduce_sum(tf$square(w)))),
              2*length(train_test_sets$x_train))
train_op <- tf$train$GradientDescentOptimizer(learning_rate)$minimize(cost)
```

Set up the session

``` r
sess <- tf$Session()
init <- tf$global_variables_initializer()
sess$run(init)
```

Try out various regularization parameters

``` r
for(reg_lambda in seq.int(0,1,length.out = 100)){
    for(epoch in 1:training_epochs){
        sess$run(train_op, feed_dict=dict(X= train_test_sets$x_train, Y= train_test_sets$y_train))
    }
    final_cost <- sess$run(cost, feed_dict=dict(X= train_test_sets$x_test, Y=train_test_sets$y_test))
    print(paste('reg lambda', reg_lambda))
    print(paste('final cost', final_cost))
}
```

    ## [1] "reg lambda 0"
    ## [1] "final cost 0.0153459683060646"
    ## [1] "reg lambda 0.0101010101010101"
    ## [1] "final cost 0.0113882794976234"
    ## [1] "reg lambda 0.0202020202020202"
    ## [1] "final cost 0.00975883193314075"
    ## [1] "reg lambda 0.0303030303030303"
    ## [1] "final cost 0.00873591564595699"
    ## [1] "reg lambda 0.0404040404040404"
    ## [1] "final cost 0.00806906726211309"
    ## [1] "reg lambda 0.0505050505050505"
    ## [1] "final cost 0.00763418339192867"
    ## [1] "reg lambda 0.0606060606060606"
    ## [1] "final cost 0.00734846061095595"
    ## [1] "reg lambda 0.0707070707070707"
    ## [1] "final cost 0.00715737091377378"
    ## [1] "reg lambda 0.0808080808080808"
    ## [1] "final cost 0.00702570145949721"
    ## [1] "reg lambda 0.0909090909090909"
    ## [1] "final cost 0.00693095894530416"
    ## [1] "reg lambda 0.101010101010101"
    ## [1] "final cost 0.00685887457802892"
    ## [1] "reg lambda 0.111111111111111"
    ## [1] "final cost 0.00680046668276191"
    ## [1] "reg lambda 0.121212121212121"
    ## [1] "final cost 0.00675013661384583"
    ## [1] "reg lambda 0.131313131313131"
    ## [1] "final cost 0.00670443149283528"
    ## [1] "reg lambda 0.141414141414141"
    ## [1] "final cost 0.00666126562282443"
    ## [1] "reg lambda 0.151515151515152"
    ## [1] "final cost 0.00661941012367606"
    ## [1] "reg lambda 0.161616161616162"
    ## [1] "final cost 0.00657816464081407"
    ## [1] "reg lambda 0.171717171717172"
    ## [1] "final cost 0.00653715571388602"
    ## [1] "reg lambda 0.181818181818182"
    ## [1] "final cost 0.00649621244519949"
    ## [1] "reg lambda 0.191919191919192"
    ## [1] "final cost 0.00645526917651296"
    ## [1] "reg lambda 0.202020202020202"
    ## [1] "final cost 0.00641433335840702"
    ## [1] "reg lambda 0.212121212121212"
    ## [1] "final cost 0.00637344643473625"
    ## [1] "reg lambda 0.222222222222222"
    ## [1] "final cost 0.00633264845237136"
    ## [1] "reg lambda 0.232323232323232"
    ## [1] "final cost 0.00629201298579574"
    ## [1] "reg lambda 0.242424242424242"
    ## [1] "final cost 0.00625158566981554"
    ## [1] "reg lambda 0.252525252525253"
    ## [1] "final cost 0.00621140794828534"
    ## [1] "reg lambda 0.262626262626263"
    ## [1] "final cost 0.00617152312770486"
    ## [1] "reg lambda 0.272727272727273"
    ## [1] "final cost 0.00613201875239611"
    ## [1] "reg lambda 0.282828282828283"
    ## [1] "final cost 0.00609286827966571"
    ## [1] "reg lambda 0.292929292929293"
    ## [1] "final cost 0.00605411734431982"
    ## [1] "reg lambda 0.303030303030303"
    ## [1] "final cost 0.00601579807698727"
    ## [1] "reg lambda 0.313131313131313"
    ## [1] "final cost 0.00597789790481329"
    ## [1] "reg lambda 0.323232323232323"
    ## [1] "final cost 0.00594046525657177"
    ## [1] "reg lambda 0.333333333333333"
    ## [1] "final cost 0.00590347405523062"
    ## [1] "reg lambda 0.343434343434343"
    ## [1] "final cost 0.0058669694699347"
    ## [1] "reg lambda 0.353535353535354"
    ## [1] "final cost 0.00583092728629708"
    ## [1] "reg lambda 0.363636363636364"
    ## [1] "final cost 0.00579536193981767"
    ## [1] "reg lambda 0.373737373737374"
    ## [1] "final cost 0.00576030788943172"
    ## [1] "reg lambda 0.383838383838384"
    ## [1] "final cost 0.00572572834789753"
    ## [1] "reg lambda 0.393939393939394"
    ## [1] "final cost 0.00569160981103778"
    ## [1] "reg lambda 0.404040404040404"
    ## [1] "final cost 0.00565799092873931"
    ## [1] "reg lambda 0.414141414141414"
    ## [1] "final cost 0.00562488101422787"
    ## [1] "reg lambda 0.424242424242424"
    ## [1] "final cost 0.00559226190671325"
    ## [1] "reg lambda 0.434343434343434"
    ## [1] "final cost 0.00556008657440543"
    ## [1] "reg lambda 0.444444444444444"
    ## [1] "final cost 0.00552839040756226"
    ## [1] "reg lambda 0.454545454545455"
    ## [1] "final cost 0.0054971850477159"
    ## [1] "reg lambda 0.464646464646465"
    ## [1] "final cost 0.0054664658382535"
    ## [1] "reg lambda 0.474747474747475"
    ## [1] "final cost 0.00543622951954603"
    ## [1] "reg lambda 0.484848484848485"
    ## [1] "final cost 0.00540646817535162"
    ## [1] "reg lambda 0.494949494949495"
    ## [1] "final cost 0.00537718087434769"
    ## [1] "reg lambda 0.505050505050505"
    ## [1] "final cost 0.00534834759309888"
    ## [1] "reg lambda 0.515151515151515"
    ## [1] "final cost 0.00531994039192796"
    ## [1] "reg lambda 0.525252525252525"
    ## [1] "final cost 0.00529199512675405"
    ## [1] "reg lambda 0.535353535353535"
    ## [1] "final cost 0.00526450946927071"
    ## [1] "reg lambda 0.545454545454546"
    ## [1] "final cost 0.00523747643455863"
    ## [1] "reg lambda 0.555555555555556"
    ## [1] "final cost 0.00521089322865009"
    ## [1] "reg lambda 0.565656565656566"
    ## [1] "final cost 0.00518475007265806"
    ## [1] "reg lambda 0.575757575757576"
    ## [1] "final cost 0.00515904789790511"
    ## [1] "reg lambda 0.585858585858586"
    ## [1] "final cost 0.00513378391042352"
    ## [1] "reg lambda 0.595959595959596"
    ## [1] "final cost 0.00510894414037466"
    ## [1] "reg lambda 0.606060606060606"
    ## [1] "final cost 0.00508452346548438"
    ## [1] "reg lambda 0.616161616161616"
    ## [1] "final cost 0.00506051676347852"
    ## [1] "reg lambda 0.626262626262626"
    ## [1] "final cost 0.00503692636266351"
    ## [1] "reg lambda 0.636363636363636"
    ## [1] "final cost 0.00501374388113618"
    ## [1] "reg lambda 0.646464646464647"
    ## [1] "final cost 0.00499091809615493"
    ## [1] "reg lambda 0.656565656565657"
    ## [1] "final cost 0.00496848206967115"
    ## [1] "reg lambda 0.666666666666667"
    ## [1] "final cost 0.00494644325226545"
    ## [1] "reg lambda 0.676767676767677"
    ## [1] "final cost 0.00492478813976049"
    ## [1] "reg lambda 0.686868686868687"
    ## [1] "final cost 0.00490351347252727"
    ## [1] "reg lambda 0.696969696969697"
    ## [1] "final cost 0.00488262949511409"
    ## [1] "reg lambda 0.707070707070707"
    ## [1] "final cost 0.00486210687085986"
    ## [1] "reg lambda 0.717171717171717"
    ## [1] "final cost 0.00484194420278072"
    ## [1] "reg lambda 0.727272727272727"
    ## [1] "final cost 0.00482216151431203"
    ## [1] "reg lambda 0.737373737373737"
    ## [1] "final cost 0.00480271922424436"
    ## [1] "reg lambda 0.747474747474748"
    ## [1] "final cost 0.00478362431749701"
    ## [1] "reg lambda 0.757575757575758"
    ## [1] "final cost 0.00476488238200545"
    ## [1] "reg lambda 0.767676767676768"
    ## [1] "final cost 0.00474646734073758"
    ## [1] "reg lambda 0.777777777777778"
    ## [1] "final cost 0.00472838943824172"
    ## [1] "reg lambda 0.787878787878788"
    ## [1] "final cost 0.00471063703298569"
    ## [1] "reg lambda 0.797979797979798"
    ## [1] "final cost 0.00469319662079215"
    ## [1] "reg lambda 0.808080808080808"
    ## [1] "final cost 0.00467608822509646"
    ## [1] "reg lambda 0.818181818181818"
    ## [1] "final cost 0.00465927552431822"
    ## [1] "reg lambda 0.828282828282828"
    ## [1] "final cost 0.00464278226718307"
    ## [1] "reg lambda 0.838383838383838"
    ## [1] "final cost 0.00462657725438476"
    ## [1] "reg lambda 0.848484848484849"
    ## [1] "final cost 0.00461067399010062"
    ## [1] "reg lambda 0.858585858585859"
    ## [1] "final cost 0.00459505803883076"
    ## [1] "reg lambda 0.868686868686869"
    ## [1] "final cost 0.00457971729338169"
    ## [1] "reg lambda 0.878787878787879"
    ## [1] "final cost 0.00456467177718878"
    ## [1] "reg lambda 0.888888888888889"
    ## [1] "final cost 0.00454988423734903"
    ## [1] "reg lambda 0.898989898989899"
    ## [1] "final cost 0.00453539378941059"
    ## [1] "reg lambda 0.909090909090909"
    ## [1] "final cost 0.00452114222571254"
    ## [1] "reg lambda 0.919191919191919"
    ## [1] "final cost 0.00450717145577073"
    ## [1] "reg lambda 0.929292929292929"
    ## [1] "final cost 0.00449343258515"
    ## [1] "reg lambda 0.939393939393939"
    ## [1] "final cost 0.00447996705770493"
    ## [1] "reg lambda 0.94949494949495"
    ## [1] "final cost 0.00446673016995192"
    ## [1] "reg lambda 0.95959595959596"
    ## [1] "final cost 0.00445376383140683"
    ## [1] "reg lambda 0.96969696969697"
    ## [1] "final cost 0.0044410121627152"
    ## [1] "reg lambda 0.97979797979798"
    ## [1] "final cost 0.0044285194016993"
    ## [1] "reg lambda 0.98989898989899"
    ## [1] "final cost 0.00441622780635953"
    ## [1] "reg lambda 1"
    ## [1] "final cost 0.00440418114885688"

``` r
sess$close()
```
