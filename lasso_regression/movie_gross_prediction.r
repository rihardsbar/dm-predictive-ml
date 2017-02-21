library(ggplot2)
library(VIM)
library(mice)
library(vcd)
require(car)
library(tabplot)
library(PerformanceAnalytics)
library(MASS)
library(glmnet)

# ===============================================================================================
movies = read.csv("../imdb_data/movie_metadata.csv")

# ===============================================================================================
# EDA
summary(movies)
str(movies)
colnames(movies)

# ===============================================================================================
# Filter dataset
movies = movies[movies$country=='USA',]
ms_all_rows = movies[, c("imdb_score",
                "director_facebook_likes", 
                "cast_total_facebook_likes", 
                #"actor_1_facebook_likes",
                #"actor_2_facebook_likes",
                #"actor_3_facebook_likes",
                #"movie_facebook_likes", 
                "facenumber_in_poster",
                "gross",
                "budget")]

ms = na.omit(ms_all_rows)

write.csv(ms, file = "movies2.csv")

# ===============================================================================================
# EDA on smaller data
cor(ms)
plot(ms, pch='.')
chart.Correlation(ms) # Interesting relationship: gross and score
scatterplotMatrix(ms, pch=".")

# ===============================================================================================
# https://cran.r-project.org/web/packages/tabplot/vignettes/tabplot-vignette.html
tableplot(ms, sortCol="gross") # Provides a sorted image according to class

# ===============================================================================================
# fit multiple linear regression model
model.linear  = lm(gross ~ imdb_score, data = ms)
model.linear2 = lm(gross ~ imdb_score+director_facebook_likes+cast_total_facebook_likes+budget, data=ms)
model.linear3 = lm(imdb_score ~ ., data = ms) #The model with ALL variables.
model.linear4 = lm(imdb_score ~ director_facebook_likes
                   +facenumber_in_poster+gross, data = ms) 

summary(model.linear)
summary(model.linear4)

view_model(model.linear4)

# Simple regression 
plot(ms$imdb_score, ms$gross)
abline(model.linear$coefficients,col="red")

AIC(model.linear,model.linear2,model.linear3,model.linear4)

##########################
#####Ridge Regression#####
##########################
x = as.matrix(ms[, -1])
y = ms[, 1]

# Fitting the ridge regression. Alpha = 0 for ridge regression.
grid = 10^seq(5, -2, length = 100)
model.ridge = glmnet(x, y, alpha = 0, lambda = grid)
dim(coef(model.ridge)) #20 different coefficients, estimated 100 times --once each per lambda value.
coef(model.ridge) #Inspecting the various coefficient estimates.

# Visualizing the ridge regression shrinkage.
plot(model.ridge, xvar = "lambda", label = TRUE, main = "Ridge Regression")

# Creating training and testing sets. Here we decide to use a 70-30 split with approximately 70% of our data in the training 
# set and 30% of our data in the test set.
set.seed(0)
train = sample(1:nrow(x), 7*nrow(x)/10)
test = (-train)
y.test = y[test]

length(train)/nrow(x)
length(y.test)/nrow(x)

#Instead of arbitrarily choosing random lambda values and calculating the MSE
#manually, it's a better idea to perform cross-validation in order to choose
#the best lambda over a slew of values.

#Running 10-fold cross validation.
set.seed(0)
cv.ridge.out = cv.glmnet(x[train, ], y[train], lambda = grid, alpha = 0, nfolds = 10)
plot(cv.ridge.out, main = "Ridge Regression\n")
bestlambda.ridge = cv.ridge.out$lambda.min
bestlambda.ridge
log(bestlambda.ridge)

#What is the test MSE associated with this best value of lambda?
ridge.bestlambdatrain = predict(model.ridge, s = bestlambda.ridge, newx = x[test, ])
sqrt(mean((ridge.bestlambdatrain - y.test)^2))

#Refit the ridge regression on the overall dataset using the best lambda value
#from cross validation; inspect the coefficient estimates.
ridge.out = glmnet(x, y, alpha = 0)
predict(ridge.out, type = "coefficients", s = bestlambda.ridge)

#Let's also inspect the MSE of our final ridge model on all our data.
ridge.bestlambda = predict(ridge.out, s = bestlambda.ridge, newx = x)
sqrt(mean((ridge.bestlambda - y)^2))


##########################
#####Lasso Regression#####
##########################

#Fitting the lasso regression. Alpha = 1 for lasso regression.
lasso.models = glmnet(x, y, alpha = 1, lambda = grid)

dim(coef(lasso.models)) #20 different coefficients, estimated 100 times --
#once each per lambda value.
coef(lasso.models) #Inspecting the various coefficient estimates.

#Instead of arbitrarily choosing random lambda values and calculating the MSE
#manually, it's a better idea to perform cross-validation in order to choose
#the best lambda over a slew of values.

#Running 10-fold cross validation.
set.seed(0)
cv.lasso.out = cv.glmnet(x[train, ], y[train], lambda = grid, alpha = 1, nfolds = 10)
plot(cv.lasso.out, main = "Lasso Regression\n")
bestlambda.lasso = cv.lasso.out$lambda.min
bestlambda.lasso
log(bestlambda.lasso)

#What is the test MSE associated with this best value of lambda?
lasso.bestlambdatrain = predict(lasso.models, s = bestlambda.lasso, newx = x[test, ])
mean((lasso.bestlambdatrain - y.test)^2)

#Here the MSE is much lower at approximately 89,452; a further improvement
#on that which we have seen above.

#Refit the lasso regression on the overall dataset using the best lambda value
#from cross validation; inspect the coefficient estimates.
lasso.out = glmnet(x, y, alpha = 1)
predict(lasso.out, type = "coefficients", s = bestlambda.lasso)

#Let's also inspect the MSE of our final lasso model on all our data.
lasso.bestlambda = predict(lasso.out, s = bestlambda.lasso, newx = x)
mean((lasso.bestlambda - y)^2)


