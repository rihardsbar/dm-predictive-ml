d<-read.csv("/Users/syz/Documents/Semester2/DataMining/groupcoursework/moviedata/movie_metadata.csv", head=TRUE, sep=",")
sum(d)
plot(d$budget/100000, d$gross/1000000)
plot(d$movie_facebook_likes, d$gross/1000000)
d<-subset(d, d$country == 'USA')
plot(d$budget, d$gross/1000000)
plot(d$movie_facebook_likes, d$gross/1000000)
d<-subset(d, d$movie_facebook_likes != 0)
d<-subset(d, d$gross != 0)
plot(d$movie_facebook_likes, d$gross/1000000)
library(caret)
names(d)
nrow(d)
sample_size <- floor(0.75*nrow(d))
sample_size
set.seed(123)
train_ind <- sample(seq_len(nrow(d)), size = sample_size)
train <- d[train_ind, ]
test <-  d[-train_ind, ]
names(train)
part <- createDataPartition(d, p = 0.8, list = FALSE)
length(d$gross)
data <- d[part]
names(data)
summary(data)
data[0]
d<-subset(d, d$movie_facebook_likes >=100)
plot(d$movie_facebook_likes, d$gross/1000000)

summary(d)
length(d$movie_facebook_likes)
summary(d$movie_facebook_likes)
x = d$movie_facebook_likes
y = d$imdb_score
model <- lm(exp(y)~log(x))
plot(log(log(x)), exp(y), pch=1)
abline(model)
length(d$gross)

plot(d$budget, d$gross/1000000)

plot(d$movie_facebook_likes, d$gross/1000000)

small <- subset(d, d$gross>1000000)

plot(small$budget, small$gross/1000000)

model1 <- lm(small$imdb_score, small$gross)

plot(small$budget, (small$gross/1000000)^(1/4), log='x')

plot(log(log(small$budget)), small$gross/1000000)

#facebook likes of the movies

plot(small$budget, small$gross/1000000)

small1 <- subset(small, small$movie_facebook_likes>1000)
plot(log(small1$movie_facebook_likes/10000), small1$gross/1000000)
plot(small$num_user_for_reviews, small$gross/1000000)
length(small$gross)
cor(x, y)
hist(d$movie_facebook_likes)
flike <- d$actor_1_facebook_likes + d$actor_2_facebook_likes + d$actor_3_facebook_likes
hist(flike/10)
hist(d$actor_1_facebook_likes/1000)
summary(d$actor_1_facebook_likes)
boxplot(d$actor_1_facebook_likes/1000)
lm(d$imdb_score~flike)
plot(d$actor_1_facebook_likes, d$imdb_score)
hist(d$num_voted_users)
summary(flike)
length(flike)
plot(exp(flike), d$imdb_score)
cor(d$imdb_score, d$gross)
cor(d)
summary(d)
hist(d$gross)
hist(d$imdb_score)
d[1:2, 2:3]
d[d$imdb_score >9, 12]
#summary(lm(d$gross~d$imdb_score, data=d))
plot(log(d$imdb_score), d$gross/1000000*d$gross/1000000)
names(d)
x <- c(1,3,4,6,8,9,12)
y <- c(5,8,6,10,9,13,12)
plot(x,y)

for (df in seq(3,31,2)){
  for(i in 1:30){
    x<-rnorm(df,mean=10, sd=2)
    points(df, var(x))
  }
}
plot(c(2,5), c(16,10), type="n", ylab = "y", xlab = "x", ylim=c(0,20), xlim=c(0,6))
points(c(2,5), c(16, 10), pch=16)
lines(c(2,2), c(16, 10))
text(1,13, "delta y")
lines(c(2,5), c(10,10))
text(3.5,5, "delta x")
lines(c(2,5), c(16,10))
abline(20,-2)
