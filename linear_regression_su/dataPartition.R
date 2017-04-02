d<-read.csv("/Users/syz/Documents/Semester2/DataMining/groupcoursework/moviedata/movie_metadata.csv", head=TRUE, sep=",")
dataset <- na.omit(d)
d<-subset(dataset, dataset$country == 'USA')
d<-subset(d, d$movie_facebook_likes != 0)
d<-subset(d, d$gross != 0)
d<-subset(d, d$budget != 0)
library(caret)
names(d)
nrow(d)
sample_size <- floor(0.75*nrow(d))
sample_size
set.seed(123)
train_ind <- sample(seq_len(nrow(d)), size = sample_size)
train <- d[train_ind, ]
test <-  d[-train_ind, ]

write.csv(train, "/Users/syz/Documents/Semester2/DataMining/groupcoursework/moviedata/train.csv", 
          col.names = TRUE, sep=",")
write.csv(test, "/Users/syz/Documents/Semester2/DataMining/groupcoursework/moviedata/test.csv", 
          col.names = TRUE, sep=",")

nrow(train)
nrow(dataset)
nrow(d)
dataset<-subset(dataset, dataset$country == 'USA')
dataset<-subset(dataset, dataset$gross > 0)
dataset<-subset(dataset, dataset$budget > 0)
df <- dataset[, c(3, 4, 5, 6, 8, 9, 13, 14, 16, 19, 23, 24, 25, 26, 27, 28)]
plot(log(df$gross), df$budget/1000000)
df1 <- subset(df, df$gross >= 8800000)
plot(df1$gross, (df1$budget/1000000)^2)
abline(lm(df1$budget~log(df1$gross)))
df2 <- subset(df2, df2$gross >= 2000000)
plot(log(df2$gross), df2$budget/1000000)


#sample_size <- floor(0.75*nrow(df))
#sample_size
#set.seed(123)
#train_ind <- sample(seq_len(nrow(df)), size = sample_size)
#train <- df[train_ind, ]
#test <-  df[-train_ind, ]

set.seed(7)
df <- df[sample(nrow(df)), ]
train <- df[1:2000,]
validation <- df[2001:2801,]
test <- df[2001:nrow(df), ]

set.seed(5)
r <- randomForest(gross~., data = train, mtry = 5)
p <- predict(r, validation)

#RMSE
sqrt((sum((validation$gross - p)^2))/ nrow(validation))

#MSE
mean((p - validation$gross)^2)

#we predict on test dataset.
ptest <- predict(r, test)
#RMSE
sqrt((sum((test$gross - ptest)^2))/ nrow(test))

mean((ptest - test$gross)^2)

varImpPlot(r)
summary(df$num_voted_users)
plot((train$num_voted_users)^(1/8), (train$gross)^(1/5))
plot((test$num_voted_users)^(1/8), (test$gross)^(1/5))
prd <- lm((I(train$gross)^(1/5))~I((train$num_voted_users)^(1/8)), data = train)
abline(prd)


prd
p <- predict(prd, test)
summary(p)
x <- (train$num_voted_users)^(1/8)
y <- (train$gross)^(1/5)
tr <- c(x, y)
fit<-lm(y ~ x) #use name b here
x <- (test$num_voted_users)^(1/8)
y1 <- (test$gross)^(1/5)





p <- predict(fit,data.frame(x), interval="confidence") #use name x as previously
length(p)
actuals_preds <- data.frame(cbind(actuals=y1, predicteds=p))  # make actuals_predicteds dataframe.
correlation_accuracy <- cor(actuals_preds)  # 82.7%
head(actuals_preds)

min_max_accuracy <- mean(apply(actuals_preds, 1, min) / apply(actuals_preds, 1, max))  
# => 58.42%, min_max accuracy
mape <- mean(abs((actuals_preds$predicteds - actuals_preds$actuals))/actuals_preds$actuals) 
