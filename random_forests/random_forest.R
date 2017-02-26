library(tidyverse)
library(stringr)
library(dplyr)
library(caret)
library(Boruta)
library(randomForest)
library(ggplot2)

##########read in movie data##########
movies <- read_csv("movie_metadata.csv")
attach(movies)

##########read in CPI data##########
CPI_1913_2016 <- read_csv("CPI_values.csv")
attach(CPI_1913_2016)

##########rename title_year column##########
movies <- movies %>% rename(year=title_year)

##########filter for USA only movies##########
movies_usa <- movies %>% filter(country=="USA")

##########omit na values##########
s = sum(is.na(movies_usa))
movies_usa <- na.omit(movies_usa)


##########Adjust budget and gross values for inflation##########
#Formula is X_t = Y_t / Z_t*Z_2016
usa_adj_inflation <- movies_usa %>% 
  left_join(CPI_1913_2016, by="year") %>% 
  mutate(adj_budget = 240.007/CPI*budget, adj_gross= 240.007/CPI*gross )

movies_usa <- usa_adj_inflation

########## Transformation ##########
movies_usa$adj_gross <- log(movies_usa$adj_gross)
movies_usa$adj_budget <- log(movies_usa$adj_budget)

##########display movie title, previous and adjusted budgets and gross##########
movies_usa %>% select(movie_title, year, budget, adj_budget, adj_gross)


#select_columns = c('imdb_score', 'num_user_for_reviews', 'num_critic_for_reviews', 'duration', 
#                   'director_facebook_likes', 'movie_facebook_likes', 'cast_total_facebook_likes',
#                   'actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes',
#                   'num_voted_users', 'facenumber_in_poster', 
#                   'year', 'aspect_ratio', 'adj_budget', 'adj_gross')


select_columns = c('imdb_score', 'num_user_for_reviews', 'num_critic_for_reviews', 'cast_total_facebook_likes',
                   'actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes', 'num_voted_users', 'adj_budget', 'adj_gross')

movies_usa <- movies_usa[,select_columns]

#preprocessParams <- preProcess(movies_usa[,1:10], method=c("range"))
#print(preprocessParams)
#transformed <- predict(preprocessParams, movies_usa[,1:10])
#movies_usa <- transformed

##########Divide into training and testing data##########
inTrain = createDataPartition (movies_usa$adj_gross, p=0.8, list= FALSE ) 
trainData = movies_usa[inTrain,] 
testData = movies_usa[-inTrain,]

###Feature Selection(Variable Importance)
set.seed(123)
boruta.train <- Boruta(adj_gross~ . -year, data = trainData, doTrace = 2)
boruta.train

##########Boxplot of variable importance##########
plot(boruta.train, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i)
  boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)

##########classify tentative attribute##########
final.boruta <- TentativeRoughFix(boruta.train)
print(final.boruta)

##########Display attributes of importance##########
getSelectedAttributes(final.boruta, withTentative = F)

##########create data frame of results##########
boruta.df <- attStats(final.boruta)
print(boruta.df)


##########Random Forest########## 
library(miscTools)
rf <- randomForest(adj_gross ~., dev, ntree=20)
importance(rf)
varImpPlot(rf,type=2)

########## make predictions - Test data########## 
predictions <- predict(rf, testData)
########## summarize accuracy########## 
rmse <- sqrt(mean((testData$adj_gross - predictions)^2))
print(rmse)

r2 <- rSquared(testData$adj_gross, testData$adj_gross - predict(rf, testData))

# libary(tree)
# ### A Single Tree based on all variables
# fit2.single <- tree(adj_gross~., trainData)
# 
# png("fit2.png", width = 800, height = 400)
# plot(fit2.single)
# text(fit2.single, pretty=1)
# par(mfrow=c(1,1))
# dev.off()
# 
# fit2.single.result <- summary(fit2.single)
# rss2.single = fit2.single.result$dev
# 
# # A Random forest based on all variables
# # Set mtry = 5 because of the rule of thumb: mtry = n/3
# fit3.rf <- randomForest(adj_gross~., trainData, mtry=5, ntree=100)
# yhat <- predict(fit3.rf, trainData)   # predicted values we use 
# 
# png("fit2.png", width = 800, height = 400)
# plot(trainData$adj_gross, yhat, pch=16,  # add a 45 degree line:looks very good!
#      main="Y vs. Predicted Y", col="blue")
# abline(0, 1, lwd=5, col="red")
# dev.off()
# 
# rss.rf = sum((trainData$adj_gross-yhat)^2)
# plot(fit3.rf, col="red", pch=16, type="p", main="Random Forest Error")
# 
# # Plot says we need 500 trees
# #Run above loop a few time, it is not very unstable. 
# #The recommended mtry for reg trees are mtry=p/3=19/3 about 6 or 7. Are you convinced with p/3?
# 
# fit2.rf <- randomForest(loggross~., train, mtry=5, ntree=500, importance=TRUE)
# yhat2 <- predict(fit2.rf, train)   # predicted values we use 
# 
# plot(train$loggross, yhat2, pch=16,  # add a 45 degree line:looks very good!
#      main="Y vs. Predicted Y", col="blue")
# abline(0, 1, lwd=5, col="red")
# 
# rss2.rf = sum((train$loggross-yhat2)^2)
# 
# # RSS went down, cool
# 
# #### Evaluate vs Test Data
# 
# # four fits fit0.single, fit1.single, fit.rf, fit2.rf
# 
# fit0.single.test <- predict(fit0.single, test)
# plot(test$loggross, fit0.single.test, pch=16,  # add a 45 degree line:looks very good!
#      main="Y vs. Predicted Y", col="blue")
# abline(0, 1, lwd=5, col="red")
# rss2.rf = sum((test$loggross-fit0.single.test)^2)
# 
# fit1.single.test <- predict(fit1.single, test)
# plot(test$loggross, fit1.single.test, pch=16,  # add a 45 degree line:looks very good!
#      main="Y vs. Predicted Y", col="blue")
# abline(0, 1, lwd=5, col="red")
# rss5.rf = sum((test$loggross-fit1.single.test)^2)
# 
# 
# fit.rf.test <- predict(fit.rf, test)
# plot(test$loggross, fit.rf.test, pch=16,  # add a 45 degree line:looks very good!
#      main="Y vs. Predicted Y", col="blue")
# abline(0, 1, lwd=5, col="red")
# rss2.rf = sum((test$loggross-fit.rf.test)^2)


# fit2.rf.test <- predict(fit2.rf, test)
# plot(test$loggross, fit2.rf.test, pch=16,  # add a 45 degree line:looks very good!
#      main="Y vs. Predicted Y", col="blue")
# abline(0, 1, lwd=5, col="red")
# rss7.rf = sum((test$loggross-fit2.rf.test)^2)
#testData$rightPred <- predictions == testData$adj_gross
#accuracy <- sum(testData$rightPred)/nrow(testData)

