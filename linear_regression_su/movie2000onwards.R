d<-read.csv("/Users/syz/Documents/Semester2/DataMining/groupcoursework/moviedata/movie_metadata.csv", head=TRUE, sep=",")
dataset <- na.omit(d)
d<-subset(dataset, dataset$country == 'USA')
#d<-subset(dataset, dataset$title_year >= 2000)
d<-subset(d, d$gross != 0)
d<-subset(d, d$budget != 0)
d<-subset(d, d$movie_facebook_likes != 0)
d<-subset(d, d$gross != 0)
d<-subset(d, d$budget != 0)

sample_size <- floor(0.75*nrow(d))
sample_size
set.seed(123)
train_ind <- sample(seq_len(nrow(d)), size = sample_size)
train <- d[train_ind, ]
test <-  d[-train_ind, ]

plot(train$budget, train$gross)
model = lm(gross~budget, data = train)
abline(model)

new <- data.frame(budget = test$budget)

prediction <- predict(model, newdata = new, interval="prediction")

residuals <- prediction[,1] - test$gross
per <- abs(residuals)/test$gross*100


plot(test$gross/1000000, per, ylim=c(0,5000))
