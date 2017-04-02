d<-read.csv("/Users/syz/Documents/Semester2/DataMining/groupcoursework/moviedata/movie_metadata.csv", head=TRUE, sep=",")
library(ggplot2)
library(reshape2)
library(corrgram)
library(dplyr)
dataset <- na.omit(d)
d<-subset(dataset, dataset$country == 'USA')
d<-subset(d, d$gross != 0)
d<-subset(d, d$budget != 0)

sample_size <- floor(0.75*nrow(d))
sample_size
set.seed(123)
train_ind <- sample(seq_len(nrow(d)), size = sample_size)
train <- d[train_ind, ]
test <-  d[-train_ind, ]

summary(train$gross)
summary(test$gross)

plot(train$budget, train$gross)
model = lm(gross~budget, data = train)
abline(model)

new <- data.frame(budget = test$budget)

prediction <- predict(model, newdata = new, interval="prediction")

residuals <- prediction[,1] - test$gross
per <- abs(residuals)/test$gross*100

b <-0:8
b <-b*100

plot(test$gross/1000000, per, xlim = c(0,800), ylim=c(0,5000))
plot(test$budget, test$gross)

test_data <- data.frame(test$gross, prediction, abs(residuals))
mean(test_data$abs.residuals.)
ggplot(test_data, aes(x = abs.residuals.)) + geom_histogram(binwidth = 1000000)


gross <- predict(model)
lines(gross)
summary(gross)
lines(data.frame(budget = c(10000000, 100000000)), gross)

c <-table(train$title_year)
c <-data.frame(c)

b <-table(test$title_year)
b <-data.frame(b)
