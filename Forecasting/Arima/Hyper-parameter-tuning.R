setwd("C:/Users/vidis/OneDrive/Desktop/Spring23/BachelorProject")

library(forecast)
library(ggplot2)
set.seed(0)

df <- read.csv('pf_filled.csv')
ind = 24*365
hf = df[0:ind,]

sample_sizes = c(3000,2000,1000,500,100)
start_inds = list(sample(c(4000:(ind-24)),size=5))

mses <- c()
method <- c()
p_value <-c()

x <- data.frame(sample_size = numeric(), building = numeric(), test_start_ind = numeric(), rmse = numeric(),method = character(),p_value = numeric())

for (sample_size in sample_sizes) {
  
  print(sample_size)
  
  for (col in c('prod_k28','prod_h16','prod_h22','prod_h28','prod_h32')) {
    i = 1
    print(col)
    
    while (i < (length(start_inds[[1]])+1)) {
      
      tsi = start_inds[[1]][i]
      print(tsi)
      
      train = hf[(tsi-sample_size):(tsi-1),col]
      print(length(train))
      test = hf[tsi:(tsi+23),col]
      
      train.ts <- ts(train, frequency=24)
      fit <- auto.arima(train.ts, seasonal=TRUE, trace=FALSE)
      fc <- forecast(fit, h=24, level=c(95))
      
      method <- append(method, fc$method)
      print(fc$method)
      
      dummy <- checkresiduals(fit, plot=F, lag=24)
      p_value <- append(p_value, dummy$p.value)
      
      mse = mean((test-fc$mean)^2)
      mses <- append(mses, mse)
      print(mse)
      
      x[(nrow(x)+1),] = c(sample_size,col,tsi, sqrt(mse), fc$method, dummy$p.value)
      print(x)
      i = i+1
      
      
    }
  }
}

write.csv(x,'C:/Users/vidis/OneDrive/Desktop/Spring23/BachelorProject/tuning.csv',row.names = FALSE)

  

