library(tidyverse)
library(ggplot2)
library(forcats)
library(plotly)
library(formattable)
library(dtplyr)
library(rvest)


######Load data######
movie <- read_csv('dataset/movie_metadata_cleaned_filled_values.csv')
glimpse(movie)
head(movie)
str(movie)

spec(movie)
unique(movie$color)
unique(movie$title_year)


###### What are the total number of movies reviewed by year? ######
attach(movie)
temp <- movie %>% select(movie_title,title_year)
temp <- temp %>% group_by(title_year) %>% summarise(n=n())
temp <- na.omit(temp)

temp %>% ggplot(aes(x=title_year,y=n)) + geom_point() + geom_line(linetype='dotted')


######What is the average IMDB rating by year?######
temp <- movie %>% select(imdb_score,title_year)
temp <- temp %>% group_by(title_year)%>% summarise(score=mean(imdb_score))
temp <- na.omit(temp)
temp %>% ggplot(aes(x=title_year,y=score)) + geom_point() + geom_line(linetype='dotted')


###### How do the average score change for each type of content rating?######
temp <- movie %>% select(content_rating,imdb_score)
temp <- temp %>% group_by(content_rating)%>% summarise(score = mean(imdb_score))

p <- plot_ly(
  x = temp$content_rating,
  y = temp$score,
  name = "Avg score by Rating",
  type = "bar")
p ## the highest average score seems to be bagged by TV-MA category


######How do these scores vary by category?######
temp <- movie %>% select(imdb_score,content_rating)
temp <- na.omit(temp)
plot_ly(temp, x = imdb_score, color = content_rating, type = "box")
## We see that the IQR of each distribution is above 5. The highest imdb_scores tend to be of the TV-MA content rating type. 
## The R rated category has the largest number of outliers that range from a score of 1.9 to 4.2.





####Filter data, target:  movies after 1990####
# movie.color <- movie %>%
#   filter(color == "Color") %>%
#   filter(title_year > 1990) %>% 
#   filter(complete.cases(gross)) %>%  filter(gross > 1000000)

#glimpse(movie.color)

for (i in 1: max(nrow(movie))){
  temp<-strsplit(as.character(movie$genres[i]), "|", fixed=TRUE)
  movie[i, "Genre.N"]<-length(temp[[1]])
  for (j in 1:length(temp[[1]])){
    movie[i,paste("Genre",j,sep=".")]<-temp[[1]][j]
  }
  
}

# unique(movie.color$Genre.1)

####Break Genres apart and transform into binary####
movie$Genre.1 <- if_else(movie$Genre.1 == "Sci-Fi","Sci.Fi",movie$Genre.1)
movie$Genre.2 <- if_else(movie$Genre.2 == "Sci-Fi","Sci.Fi",movie$Genre.2)
movie$Genre.3 <- if_else(movie$Genre.3 == "Sci-Fi","Sci.Fi",movie$Genre.3)
movie$Genre.4 <- if_else(movie$Genre.4 == "Sci-Fi","Sci.Fi",movie$Genre.4)
movie$Genre.5 <- if_else(movie$Genre.5 == "Sci-Fi","Sci.Fi",movie$Genre.5)
movie$Genre.6 <- if_else(movie$Genre.6 == "Sci-Fi","Sci.Fi",movie$Genre.6)
movie$Genre.7 <- if_else(movie$Genre.7 == "Sci-Fi","Sci.Fi",movie$Genre.7)
movie$Genre.8 <- if_else(movie$Genre.8 == "Sci-Fi","Sci.Fi",movie$Genre.8)

movie$Genre.1<-as.factor(gsub(" ","", movie$Genre.1))
movie$Genre.2<-as.factor(gsub(" ","", movie$Genre.2))
movie$Genre.3<-as.factor(gsub(" ","", movie$Genre.3))
movie$Genre.4<-as.factor(gsub(" ","", movie$Genre.4))
movie$Genre.5<-as.factor(gsub(" ","", movie$Genre.5))
movie$Genre.6<-as.factor(gsub(" ","", movie$Genre.6))
movie$Genre.7<-as.factor(gsub(" ","", movie$Genre.7))
movie$Genre.8<-as.factor(gsub(" ","", movie$Genre.8))


movie$Genre.1 <- as.factor(ifelse(is.na(movie$Genre.1)==T, "0",as.character(movie$Genre.1))) 
movie$Genre.2 <- as.factor(ifelse(is.na(movie$Genre.2)==T, "0",as.character(movie$Genre.2))) 
movie$Genre.3 <- as.factor(ifelse(is.na(movie$Genre.3)==T, "0",as.character(movie$Genre.3))) 
movie$Genre.4 <- as.factor(ifelse(is.na(movie$Genre.4)==T, "0",as.character(movie$Genre.4))) 
movie$Genre.5 <- as.factor(ifelse(is.na(movie$Genre.5)==T, "0",as.character(movie$Genre.5)))
movie$Genre.6 <- as.factor(ifelse(is.na(movie$Genre.6)==T, "0",as.character(movie$Genre.6))) 
movie$Genre.7 <- as.factor(ifelse(is.na(movie$Genre.7)==T, "0",as.character(movie$Genre.7))) 
movie$Genre.8 <- as.factor(ifelse(is.na(movie$Genre.8)==T, "0",as.character(movie$Genre.8)))

t<-unique(c(as.character(unique(movie["Genre.1"])$Genre.1), as.character(unique(movie["Genre.2"])$Genre.2),
            as.character(unique(movie["Genre.3"])$Genre.3),as.character(unique(movie["Genre.4"])$Genre.4),
            as.character(unique(movie["Genre.5"])$Genre.5),as.character(unique(movie["Genre.6"])$Genre.6),
            as.character(unique(movie["Genre.7"])$Genre.7),as.character(unique(movie["Genre.8"])$Genre.8)))
t<-t[-c(23)]
t[14] <- "Sci.Fi"
t<- make.names(t, unique=TRUE)

for (i in 1:length(t)){
  for (j in 1: nrow(movie)){
    
    if (movie[j,"Genre.1"]==t[i] | movie[j,"Genre.2"]==t[i] | movie[j,"Genre.3"]==t[i] | movie[j,"Genre.4"]==t[i] | movie[j,"Genre.5"]==t[i] | movie[j,"Genre.6"]==t[i] | movie[j,"Genre.7"]==t[i] | movie[j,"Genre.8"]==t[i]) 
      movie[j,paste(t[i],"","")]<-1
    else
      movie[j,paste(t[i],"","")]<-0
  }
}

#####Delete spaces from colmn names#####
colnames(movie) <- gsub(" ","",colnames(movie))

movie$short.genre<-  ifelse(movie$Animation==1,'Adventure',
                                  ifelse(movie$Sci.Fi==1, 'Sci_fi',
                                         ifelse(movie$Fantasy==1,'Fantasy',
                                                ifelse(movie$Comedy ==1,'Comedy',
                                                       ifelse(movie$Action==1,'Action',
                                                              ifelse(movie$Drama==1,'Drama',
                                                                     ifelse(movie$Horror==1 | movie$Mystery==1 | movie$Thriller==1,'Thriller',
                                                                            ifelse(movie$Documentary==1,'Documentary',
                                                                                   'Drama'))))))))
unique(movie$short.genre)
table(movie$short.genre)
unique(movie$aspect_ratio) 
table(movie$aspect_ratio,movie$aspect_ratio)

movie.color$aspect_ratio <- ifelse(movie.color$aspect_ratio == 1.85 , "a) 1.85:1",
                                   ifelse(movie.color$aspect_ratio == 2.35 , "b) 2.35:1","c) Others"))
table(movie.color$aspect_ratio)
movie.color$aspect_ratio <- as.factor(movie.color$aspect_ratio)

x <<- 0

for(i in 2:length(colnames(movie.color))){
  x <- append(x,length(which(is.na(movie.color[,i]) == TRUE)))  
}

x1 <<- 0
for(i in 2:length(colnames(movie.color))){
  x1 <- append(x1,length(which(movie.color[,i] == 0)))  
}


missingvalues_record <- data.frame(column = colnames(movie.color),missing = x)
missingvalues_record[missingvalues_record[,2] > 0,]

values_equal_tozero_record <- data.frame(column = colnames(movie.color),zero = x1)
missingvalues_record[values_equal_tozero_record[,2] > 0,]

################## Add code here ###################





######Show average movie ratings and number of movies per actor_1######
actor.deciles <- movie %>%
  group_by(actor_1_name) %>%
  summarize(average_rating = mean(imdb_score), number_movie = length(budget)) 


#quantile(actor.deciles$number_movie, c(.25, .50, .75,.90,.93,.95, 1)) 
#quantile(actor.deciles$average_rating, c(.25, .50, .75,.90,.95, 1)) 

######Categorize actor_1s by average movie rating and number of movies######
attach(actor.deciles)
actor.deciles$actor1_rating <- ifelse(average_rating >= 6.5 & number_movie >= 5,"Class I actor",
                                      ifelse(average_rating >= 6.5 & number_movie >= 3,"Rising Stars",
                                             ifelse(average_rating >= 7 ,"One hit wonder",
                                                    ifelse(average_rating >= 6 ,"General actor",
                                                           "Underdog"))))

(table(actor.deciles$actor1_rating)) 


######Show average movie ratings and number of movies per director######
director.deciles <- movie %>%
  group_by(director_name) %>%
  summarize(average_rating = mean(imdb_score), number_movie = length(budget)) 


#quantile(director.deciles$number_movie, c(.25, .50, .75,.90,.95, 1)) 
#quantile(director.deciles$average_rating, c(.25, .50, .75,.90,.95, 1)) 

######Categorize directors by average movie rating and number of movies######
attach(director.deciles)
director.deciles$director_rating <- ifelse(average_rating >= 7 & number_movie >= 7,"Class I director",
                                           ifelse(average_rating >= 7 & number_movie >= 3,"Rising Stars",
                                                  ifelse(average_rating >= 7 ,"One hit wonder",
                                                         ifelse(average_rating >= 6 ,"General director",
                                                                "Underdog"))))
table(director.deciles$director_rating) 

actor2.deciles <- movie %>%
  group_by(actor_2_name) %>%
  summarize(average_rating = mean(imdb_score), number_movie = length(budget)) 


quantile(actor2.deciles$number_movie, c(.25, .50, .75,.90,.95, 1)) 
quantile(actor2.deciles$average_rating, c(.25, .50, .75,.90,.95, 1)) 

attach(actor2.deciles)
actor2.deciles$actor2_rating <- ifelse(average_rating >= 6.5 & number_movie >= 5,"Class I actor",
                                       ifelse(average_rating >= 6.5 & number_movie >= 3,"Rising Stars",
                                              ifelse(average_rating >= 7 ,"One hit wonder",
                                                     ifelse(average_rating >= 6 ,"General actor",
                                                            "Underdog"))))

table(director.deciles$director_rating)
table(actor.deciles$actor1_rating)
table(actor2.deciles$actor2_rating)

#### Add Bar Charts  ------------
#### Top 5 under class 1 and rising stars  ------------

(movie <- movie %>%
   left_join(director.deciles[,c(1,4)],by="director_name"))
(movie <- movie %>%
    left_join(actor.deciles[,c(1,4)],by="actor_1_name"))
movie <- movie %>%
  left_join(actor2.deciles[,c(1,4)],by="actor_2_name")

table(movie$director_rating)
table(movie$actor1_rating)
table(movie$actor2_rating)

str(movie)
missing.records <- movie.color[!complete.cases(movie.color),]

####percentage of each feature missing in the data set####
pMiss <- function(x){sum(is.na(x))/length(x)*100}
apply(movie.color,2,pMiss)


library(mice)
library(VIM)
pattern.missing.data <- md.pattern(movie.color)
aggr_plot <- aggr(movie.color, col=c('navyblue','red'), 
                  numbers=TRUE, sortVars=TRUE, 
                  labels=names(data), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))

##########################################################################
# Key statistics by directors, average_rating = average imdb score of the movies which are directed by the 
# director
director.stats <- movie.color %>%
  group_by(director_name) %>%
  summarize(average_rating = mean(imdb_score), number_movie = length(budget), 
            average_duration = mean(duration), 
            total_budget = sum(budget/1000000, na.rm = TRUE), 
            total_gross = sum(worldwide_gross/1000000, na.rm = TRUE),
            average_gross = total_budget/number_movie) %>%
  filter(total_budget > 100) %>%
  filter(total_gross > 10)%>%
  mutate(profit_margin = (total_gross-total_budget)/total_budget) %>%
  mutate(Average.profit = (total_gross-total_budget)/number_movie) %>%
  filter(number_movie >3) %>%
  arrange(desc(profit_margin))

################ Profit Margin ######################
top10_directors <- head(director.stats,n = 10) 
top10_directors$director_name <- as.character(top10_directors$director_name)
#Then turn it back into an ordered factor
top10_directors$director_name <- factor(top10_directors$director_name, levels=unique(top10_directors$director_name))
top10_directors$profit_margin <- round(top10_directors$profit_margin,digits=2)

top10_directors %>%
  ggplot(aes(x=director_name,y=profit_margin,label=format(profit_margin, digits=2))) + 
  geom_bar(stat='identity',fill="steelblue") + 
  geom_text(aes(label=profit_margin), vjust=1.6, color="white", size=3.5)+
  theme(axis.text.x=element_text(angle = -90, vjust = 0.5, hjust=1)) +
  labs(x="",
       y="",
       title="Top 10 Profit Margins by Directors")

################ Average Gross ######################
top10_directors <- director.stats %>% arrange(desc(average_gross)) %>% head(.,n = 10)
top10_directors$director_name <- as.character(top10_directors$director_name)
#Then turn it back into an ordered factor
top10_directors$director_name <- factor(top10_directors$director_name, levels=unique(top10_directors$director_name))
top10_directors$average_gross <- round(top10_directors$average_gross,digits=2)

top10_directors %>% 
  ggplot(aes(x=director_name,y=average_gross,label=format(average_gross, digits=2))) + 
  geom_bar(stat='identity',fill="steelblue") + 
  geom_text(aes(label=average_gross), vjust=1.6, color="white", size=3.5)+
  theme(axis.text.x=element_text(angle = -90, vjust = 0.5, hjust=1)) +
  labs(x="",
       y="Avg Gross earnings in MM",
       title="Top 10 Average gross earning Directors")

################ Average Profit ######################
top10_directors <- director.stats %>% arrange(desc(Average.profit)) %>% head(.,n = 10)
top10_directors$director_name <- as.character(top10_directors$director_name)
#Then turn it back into an ordered factor
top10_directors$director_name <- factor(top10_directors$director_name, levels=unique(top10_directors$director_name))
top10_directors$Average.profit <- round(top10_directors$Average.profit,digits=2)

top10_directors %>% 
  ggplot(aes(x=director_name,y=Average.profit,label=format(Average.profit, digits=2))) + 
  geom_bar(stat='identity',fill="steelblue") + 
  geom_text(aes(label=Average.profit), vjust=1.6, color="white", size=3.5)+
  theme(axis.text.x=element_text(angle = -90, vjust = 0.5, hjust=1)) +
  labs(x="",
       y="Avg Gross profits in MM",
       title="Top 10 Average Profit earning Directors")


# Key statistics by staring actors, average_rating = average imdb score of the movies which are starring by the 
# the actor
actor.stats <- movie.color %>%
  group_by(actor_1_name) %>%
  summarize(average_rating = mean(imdb_score), number_movie = length(budget), 
            average_duration = mean(duration), 
            total_budget = sum(budget/1000000, na.rm = TRUE), 
            total_gross = sum(worldwide_gross/1000000, na.rm = TRUE),
            average_gross = total_budget/number_movie) %>%
  filter(total_budget > 100) %>%
  filter(total_gross > 1)%>%
  mutate(profit_margin = (total_gross-total_budget)/total_budget) %>%
  mutate(Average.profit = (total_gross-total_budget)/number_movie) %>%
  filter(profit_margin >1) %>%
  arrange(desc(profit_margin))

################ Profit Margin ######################
top10_leadactor <- head(actor.stats,n = 10) 
top10_leadactor$actor_1_name <- as.character(top10_leadactor$actor_1_name)
#Then turn it back into an ordered factor
top10_leadactor$actor_1_name <- factor(top10_leadactor$actor_1_name, levels=unique(top10_leadactor$actor_1_name))
top10_leadactor$profit_margin <- round(top10_leadactor$profit_margin,digits=2)

top10_leadactor %>%
  ggplot(aes(x=actor_1_name,y=profit_margin,label=format(profit_margin, digits=2))) + 
  geom_bar(stat='identity',fill="steelblue") + 
  geom_text(aes(label=profit_margin), vjust=1.6, color="white", size=3.5)+
  theme(axis.text.x=element_text(angle = -90, vjust = 0.5, hjust=1)) +
  labs(x="",
       y="",
       title="Top 10 Profit Margins by leadactor")

################ Average Gross ######################
top10_leadactor <- actor.stats %>% arrange(desc(average_gross)) %>% head(.,n = 10)
top10_leadactor$actor_1_name <- as.character(top10_leadactor$actor_1_name)
#Then turn it back into an ordered factor
top10_leadactor$actor_1_name <- factor(top10_leadactor$actor_1_name, levels=unique(top10_leadactor$actor_1_name))
top10_leadactor$average_gross <- round(top10_leadactor$average_gross,digits=2)

top10_leadactor %>% 
  ggplot(aes(x=actor_1_name,y=average_gross,label=format(average_gross, digits=2))) + 
  geom_bar(stat='identity',fill="steelblue") + 
  geom_text(aes(label=average_gross), vjust=1.6, color="white", size=3.5)+
  theme(axis.text.x=element_text(angle = -90, vjust = 0.5, hjust=1)) +
  labs(x="",
       y="Avg Gross earnings in MM",
       title="Top 10 Average gross earning leadactor")

################ Average Profit ######################
top10_leadactor <- actor.stats %>% arrange(desc(Average.profit)) %>% head(.,n = 10)
top10_leadactor$actor_1_name <- as.character(top10_leadactor$actor_1_name)
#Then turn it back into an ordered factor
top10_leadactor$actor_1_name <- factor(top10_leadactor$actor_1_name, levels=unique(top10_leadactor$actor_1_name))
top10_leadactor$Average.profit <- round(top10_leadactor$Average.profit,digits=2)

top10_leadactor %>% 
  ggplot(aes(x=actor_1_name,y=Average.profit,label=format(Average.profit, digits=2))) + 
  geom_bar(stat='identity',fill="steelblue") + 
  geom_text(aes(label=Average.profit), vjust=1.6, color="white", size=3.5)+
  theme(axis.text.x=element_text(angle = -90, vjust = 0.5, hjust=1)) +
  labs(x="",
       y="Avg Gross profits in MM",
       title="Top 10 Average Profit earning leadactor")

library("ggplot2")
library("RColorBrewer")
library("data.table")
library(corrgram)

color_scheme = brewer.pal(8, "Blues")

keep_var <- c('num_critic_for_reviews',
              'duration',
              'worldwide_gross',
              'movie_title',
              'num_voted_users',
              'movie_imdb_link',
              'num_user_for_reviews',
              'language',
              'country',
              'content_rating',
              'budget',
              'title_year',
              'imdb_score',
              'aspect_ratio',
              'Genre.N',
              'short.genre',
              'director_rating',
              'actor1_rating',
              'actor2_rating')

#movie.color.nonmissing <- movie.color[complete.cases(movie.color[keep_var]),]
movie <- movie[complete.cases(movie[keep_var]),]

movie.color.nonmissing <- movie.color[,keep_var]

s = sum(is.na(movie.color.nonmissing))
movie.color.nonmissing <- na.omit(movie.color.nonmissing)


#################### Subset Directors #################
# Subset the directors from entire dataset
director = movie.color['director_name']
# Count how many times each director is in the dataset
director = data.frame(table(movie.color$director_name))
# Sort the dataset by the frequency each director appears
director = director[order(director$Freq,decreasing=TRUE),]
# Remove the row without a director name
director = director[-c(1),]
# Plot the top 20 directors with the most movies
# reorder(factor(director), Freq)


#################### Subset Facenumber #################
# Subset the posters
poster = movie.color['facenumber_in_poster']

# Count how many times each count of faces is in the dataset
poster = data.frame(table(movie.color$facenumber_in_poster))

# Sort the dataset by the frequency each face count appears
poster = poster[order(poster$Freq,decreasing=TRUE),]
poster$Var1 <- factor(poster$Var1, levels=unique(poster$Var1))

# Plot the face count occurences in posters
ggplot(poster, aes(x=Var1, y=Freq, alpha=Freq)) + 
  geom_bar(stat = "identity", fill=color_scheme[8]) + 
  xlab("Number of Faces on Movie Poster") + 
  ylab("Frequency") + 
  ggtitle("Distribution of the Number Faces on Movie Posters") + 
  coord_flip()

# There are some years that there was no data
year = movie.color['title_year']
year = data.frame(table(movie.color$title_year))
year = year[order(year$Freq,decreasing=TRUE),]
# year$Var1 <- factor(year$Var1, levels=unique(year$Var1))
# # Bar Graph
# ggplot(data=year, aes(x=Var1, y=Freq)) + 
#   geom_bar(colour = "black", fill = "blue", width = 0.8, stat="identity") + 
#   xlab("Year") +
#   ylab("Count") +
#   ggtitle("Number of Movies by Year") +
#   scale_x_discrete(breaks = seq(1916, 2016, 5))

# Line Graph
ggplot(data=year, aes(x=Var1, y=Freq, group = 1)) +
  geom_line(colour="red", linetype=1, size=1.5) +
  geom_point(colour="blue", size=4, shape=21, fill="blue") +
  xlab("Year") +
  ylab("Count") +
  ggtitle("Number of Movies by Year") +
  scale_x_discrete(breaks = seq(1916, 2016, 5)) 


# IMDB Score Averages by Director
imdb_scores = as.data.table(subset(movie.color, movie.color$director_name != ''))
imdb_scores = imdb_scores[, mean(imdb_score), by=director_name]
names(imdb_scores) = c("director", "Average_Score")
imdb_scores = imdb_scores[order(imdb_scores$Average_Score,decreasing=TRUE),]
imdb_scores

ggplot(imdb_scores[1:20,], aes(x=reorder(factor(director), Average_Score), y=Average_Score, alpha=Average_Score)) + 
  geom_bar(stat = "identity", fill=color_scheme[8]) + 
  xlab("Director") + 
  ylab("Average IMDB Score") + 
  ggtitle("Top 20 Director Average IMDB Scores") + 
  coord_flip()

# Distribution of IMDB Scores
imdb_score = as.data.table(subset(movie.color, movie.color$imdb_score >= 0 & !is.na(movie.color$imdb_score)))
ggplot(imdb_score, aes(x=imdb_score)) + 
  geom_histogram(aes(fill=..count..), binwidth = 0.1) + 
  xlab("IMDB Score") + 
  ylab("Frequency") + 
  ggtitle("Distribution of IMDB Scores") + 
  geom_vline(aes(xintercept=mean(imdb_score, na.rm=T)), color="black", linetype="dashed", size=1) +
  scale_fill_gradient("Frequency", low = "blue", high = "red") +
  scale_x_continuous(breaks = seq(0, 10, 0.5)) 


### Select only Numeric columns for correlation check ####
##########################################################################

movie.color.numeric <-movie.color[sapply(movie.color, is.numeric)]
movie.color.numeric <- movie.color.numeric[,1:16]
movie.color.filtered <- movie.color.numeric[,c(1,2,6:7,9:11,14,16)]

#s = sum(is.na(movie.color.numeric))
#movie.color.numeric <- na.omit(movie.color.numeric)

library(corrplot)
M <- cor(movie.color.numeric)

corrplot(M)
corrplot(cor(movie.color.filtered), method="circle")

write.csv(movie,"movie_updated.csv")