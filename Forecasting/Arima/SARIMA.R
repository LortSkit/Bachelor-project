setwd("C:/Users/vidis/OneDrive/Desktop/Spring23/BachelorProject")

library(forecast)
library(ggplot2)

df <- read.csv('pf_filled.csv')

arr <- df$prod_k28

i = 0
j = 500
mses <- c()
method <- c()
test <- c()
forecasted <- c()
f_upper_bounds <- c()
f_lower_bounds <- c()
p_value <-c()

#while (j<length(arr)-24) {
while (j<548) {
  train <- arr[i:j]
  test_ <- arr[j:(j+23)]
  
  test <- append(test, test_)
  
  train.ts <- ts(train, frequency=24)

  fit <- auto.arima(train.ts, seasonal=TRUE, trace=FALSE)
  fc <- forecast(fit, h=24, level=c(95))
  
  dummy <- checkresiduals(fit, plot=F, lag=24)
  p_value <- append(p_value, dummy$p.value)
  mses <- append(mses, mean(test_^2-fc$mean^2))
  
  
  i = i+24
  j = j+24
}

