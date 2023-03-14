setwd("C:/Users/vidis/OneDrive/Desktop/Spring23/BachelorProject")

library(forecast)
library(ggplot2)

df <- read.csv('cf_filled.csv')

arr <- df$cons_k28

i = 0
j = 3000
mses <- c()
method <- c()
train <- c()
test <- c()
forecasted <- c()
f_upper_bounds <- c()
f_lower_bounds <- c()
p_value <-c()

while (j<(length(arr)-24)) {
  train_ <- arr[i:j]
  test_ <- arr[j:(j+23)]
  
  print(c(i,j))
  print(c(j,(j+23)))
  
  train <- append(train, train_)
  test <- append(test, test_)
  
  train.ts <- ts(train_, frequency=24)

  fit <- auto.arima(train.ts, seasonal=TRUE, trace=FALSE)
  fc <- forecast(fit, h=24, level=c(95))
  
  forecasted <- append(forecasted, fc$mean)
  f_upper_bounds <- append(f_upper_bounds, fc$upper)
  f_lower_bounds <- append(f_lower_bounds, fc$lower) 
  
  dummy <- checkresiduals(fit, plot=F, lag=24)
  
  p_value <- append(p_value, dummy$p.value)
  mse <- mean((test_-fc$mean)^2)
  mses <- append(mses, mse)
  print(sqrt(mse))
  
  i = i+24
  j = j+24
}

write.csv(mses,'C:/Users/vidis/OneDrive/Desktop/Spring23/BachelorProject/mses.csv',row.names = FALSE)
write.csv(train,'C:/Users/vidis/OneDrive/Desktop/Spring23/BachelorProject/train.csv',row.names = FALSE)
write.csv(test,'C:/Users/vidis/OneDrive/Desktop/Spring23/BachelorProject/test.csv',row.names = FALSE)
write.csv(forecasted,'C:/Users/vidis/OneDrive/Desktop/Spring23/BachelorProject/forecasted.csv',row.names = FALSE)
write.csv(f_upper_bounds,'C:/Users/vidis/OneDrive/Desktop/Spring23/BachelorProject/f_upper_bounds.csv',row.names = FALSE)
write.csv(f_lower_bounds,'C:/Users/vidis/OneDrive/Desktop/Spring23/BachelorProject/f_lower_bounds.csv',row.names = FALSE)
write.csv(p_value,'C:/Users/vidis/OneDrive/Desktop/Spring23/BachelorProject/p_value.csv',row.names = FALSE)
