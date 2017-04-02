d1<-read.csv("/Users/syz/Documents/Semester2/DataMining/groupcoursework/moviedata/movie_metadata.csv", head=TRUE, sep=",")
dataset <- na.omit(d1)
d<-subset(dataset, dataset$country == 'USA')
#d<-subset(dataset, dataset$title_year >= 2000)
d<-subset(d, d$gross >= 50000000)
d<-subset(d, d$gross != 0)
d<-subset(d, d$budget != 0)
d<-subset(d, d$movie_facebook_likes != 0)
d<-subset(d, d$gross != 0)
d<-subset(d, d$budget != 0)
d<-subset(d, d$actor_1_facebook_likes != 0)
d<-subset(d, d$actor_2_facebook_likes != 0)
d<-subset(d, d$actor_3_facebook_likes != 0)
d<-subset(d, d$cast_total_facebook_likes != 0)
#d<-subset(d, d$imdb_score != 0)
d<-subset(d, d$movie_facebook_likes != 0)
d<-subset(d, d$title_year != 0)


sample_size <- floor(0.75*nrow(d))
sample_size
set.seed(123)
train_ind <- sample(seq_len(nrow(d)), size = sample_size)
train <- d[train_ind, ]
test <-  d[-train_ind, ]

plot(train$budget 
     + train$movie_facebook_likes 
     + train$actor_1_facebook_likes
     + train$actor_2_facebook_likes
     + train$actor_3_facebook_likes
     + train$cast_total_facebook_likes
     , train$gross)
model = lm(gross~budget
           + movie_facebook_likes
           + actor_1_facebook_likes
           + actor_2_facebook_likes
           + actor_3_facebook_likes
           + cast_total_facebook_likes, data = train)
abline(model)

new <- data.frame(budget = test$budget,
                  movie_facebook_likes = test$movie_facebook_likes,
                  actor_1_facebook_likes = test$actor_1_facebook_likes,
                  actor_2_facebook_likes = test$actor_2_facebook_likes,
                  actor_3_facebook_likes = test$actor_3_facebook_likes,
                  cast_total_facebook_likes = test$cast_total_facebook_likes)

prediction <- predict(model, newdata = new, interval="prediction")

residuals <- prediction[,1] - test$gross
per <- abs(residuals)/test$gross*100


plot(test$gross/1000000, per, ylim=c(0,5000))
summary(per)
