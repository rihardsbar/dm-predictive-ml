library(tidyverse)
library(magrittr)
library(dplyr)




# Load data created at the last of the previous step
movies <- read_csv('movie_updated.csv')

########## transformation ###############
movies$worlwide_gross <- log(movies$worldwide_worlwide_gross)
movies$budget <- log(movies$budget)

summary(movies)
quantile(movies$worlwide_gross, c(.25, .50,  .75, .90,.95, .99))

cols <- c('language',
          'country',
          'content_rating',
          'title_year',
          'aspect_ratio',
          'Genre.N',
          'short.genre',
          'director_rating',
          'actor1_rating',
          'actor2_rating'
)

movies[,cols] <- lapply(movies[,cols], factor)
movies %>%
  mutate_each_(funs(factor(.)),cols)
str(movies)


keep_var<-c('num_critic_for_reviews',
            'duration',
            'worldwide_worlwide_gross',
            #'movie_title',
            'num_voted_users',
            #'facenumber_in_poster',
            #'movie_imdb_link',
            'num_user_for_reviews',
            #'language',
            #'country',
            'content_rating',
            'budget',
            #'title_year',
            'imdb_score',
            'aspect_ratio',
            #'Genre.N',
            'short.genre',
            'director_rating',
            'actor1_rating',
            'actor2_rating')

movies <- movies[keep_var]
str(movies)

# Sampling to create training and validation data
set.seed(123)
d <- sort(sample(nrow(movies)*0.7))
dev<-movies[d,]
val<-movies[-d,]

# Dependent variable equation
All.predictor.var <- colnames(subset( movies, select = -worlwide_gross ))
yColumn <- "worlwide_gross"
formula <- paste(yColumn,paste(All.predictor.var,collapse=' + '),sep=' ~ ')
str(movies)

### Random Forest 
install.packages("miscTools")
library(randomForest)
library(miscTools)
library(ggplot2)

rf <- randomForest(worlwide_gross ~., dev, ntree=20)
importance(rf)
varImpPlot(rf,type=2)

# importance(rf,type=1)
# make predictions
predictions <- predict(rf, dev)
# summarize accuracy
rmse <- sqrt(mean((dev$worlwide_gross - predictions)^2))
print(rmse)
(r2 <- rSquared(dev$worlwide_gross, dev$worlwide_gross - predict(rf, dev)))

# make predictions - Test data
predictions <- predict(rf, val)
# summarize accuracy
rmse <- sqrt(mean((val$worlwide_gross - predictions)^2))
print(rmse)



(r2 <- rSquared(val$worlwide_worlwide_gross, val$worlwide_gross - predict(rf, val)))
# [1] 0.6192
(mse <- mean((val$worlwide_gross - predict(rf, val))^2))
# [1] 1.745427

p <- ggplot(aes(x=actual, y=pred),
            data=data.frame(actual=val$worlwide_gross, pred=predict(rf, val)))
p + geom_point() +
  geom_abline(color="red") +
  ggtitle(paste("RandomForest Regression in R r^2=", r2, sep=""))

# run the party implementation
install.packages("party")
library(party)
cf1 <- cforest(worlwide_gross~.,data=dev,control=cforest_unbiased(ntree=20))
varimp(cf1)
varimp(cf1,conditional=TRUE)

# Add fit lines
# abline(lm(mpg~wt), col="red") # regression line (y~x) 
# lines(lowess(wt,mpg), col="blue") # lowess line (x,y)