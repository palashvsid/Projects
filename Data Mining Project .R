#rm(list=ls())
#Libraries
library(ggplot2)
library(grid)
library(gridExtra)
library(RCurl)
library(XML)
library(stringr)
#install.packages('pretty_doc')
library("pretty_doc")
library(boot)
library("verification")
library(nnet)
library(e1071)
#install.packages("randomForest")
library(randomForest)
#Load Data
news = read.csv('OnlineNewsPopularity.csv')
summary(news)
news$popularity <- cut(news$shares, breaks=quantile(news$shares, probs=c(0,0.25,0.75,0.95,1)),labels=c('Obscure','Mediocre','Popular','Viral'), include.lowest = TRUE)
summary(news$popularity)
#news$shares <- NULL
#Data Cleaning
news$Date <- strsplit(as.character(news$url),"/")
news$Year <- sapply((news$Date), "[[", 4)
news$Month <- sapply(news$Date, "[[", 5)
news$Day <- sapply(news$Date, "[[", 6)
news$Date <- paste(news$Year,news$Month,news$Day,sep="-")
news$weekday <- NULL
news$Type <- ifelse(news$data_channel_is_bus==1, "Business",
                    ifelse(news$data_channel_is_lifestyle==1, "Lifestyle", 
                           ifelse(news$data_channel_is_entertainment==1, "Entertainment", 
                                  ifelse(news$data_channel_is_socmed==1, "SocialMedia", 
                                         ifelse(news$data_channel_is_tech==1, "Technology", 
                                                ifelse(news$data_channel_is_world==1, "World", "Others")
                                         )
                                  )
                           )
                    )
)

news$Type <- as.factor(news$Type)
summary(news$Type)
news <- subset(news, select = c(- data_channel_is_lifestyle, - data_channel_is_entertainment, -  data_channel_is_bus, - data_channel_is_socmed, - data_channel_is_tech, - data_channel_is_world))
ggplot(news, aes(Type))+geom_bar()
news$url <- NULL
news$timedelta<- NULL
news$Day <- NULL
news$Month <- NULL
news$Year <- NULL
news$Date <- NULL
news$LDA_04 <- NULL
news$is_weekend <- NULL
news$weekday_is_sunday <- NULL
news <- news[news$n_unique_tokens<699,]
news$shares <- NULL

news$mediocre <- 0
news$popular <- 0
news$viral <- 0
news[news$popularity=="Mediocre",]$mediocre <- 1
news[news$popularity=="Popular",]$popular <- 1
news[news$popularity=="Viral",]$viral <- 1
#news$popularity <- NULL
news$mediocre <- NULL
news$popular <- NULL
news$viral <- NULL

news$business <- 0
news$entertainment <- 0
news$lifestyle <- 0
news$socialmedia <- 0
news$technology <- 0
news$world <- 0
news[news$Type=="Business",]$business <- 1
news[news$Type=="Entertainment",]$entertainment <- 1
news[news$Type=="Lifestyle",]$lifestyle <- 1
news[news$Type=="World",]$world <- 1
news[news$Type=="SocialMedia",]$socialmedia <- 1
news[news$Type=="Technology",]$technology <- 1
news$Type <- NULL
#news$business <- NULL
#news$entertainment <- NULL
#news$lifestyle <- NULL
#news$socialmedia <- NULL
#news$technology <- NULL
#news$world <- NULL
maxs = apply(news[c(1, 2, 6, 7, 8, 12:23)], 2, max)
mins = apply(news[c(1, 2, 6, 7, 8, 12:23)], 2, min)
news[c(1, 2, 6, 7, 8, 12:23)] <-  as.data.frame(scale(news[c(1, 2, 6, 7, 8, 12:23)], center=mins, scale=maxs-mins))


#Test and Training Data
set.seed(1)
sample_index <- sample(1:nrow(news), 0.8*nrow(news))
train_data <- news[sample_index,]
test_data <- news[-sample_index,]
validation_sample_index <-sample(1:nrow(train_data),0.25*nrow(train_data))
validation_data <- train_data[validation_sample_index,]
train_data <- train_data[-validation_sample_index,]

#Logistic
model.logistic = polr(popularity ~., data=train_data, Hess=T)

##Insample
predicted <- predict(model.logistic, train_data)
table(train_data$popularity, predicted)
mean(ifelse(train_data$popularity != predicted, 1, 0))

##Outsample
predicted <- predict(model.logistic, validation_data)
table(validation_data$popularity, predicted)
mean(ifelse(validation_data$popularity != predicted, 1, 0))


#SVM
model.svm = svm(popularity~., data = train_data, cost = 1, gamma = 1/length(train_data))

##Insample
predicted <- predict(model.svm, train_data)
table(train_data$popularity, predicted)
mean(ifelse(train_data$popularity != predicted, 1, 0))

##Outsample
predicted <- predict(model.svm, validation_data)
table(validation_data$popularity, predicted)
mean(ifelse(validation_data$popularity != predicted, 1, 0))



#RF
model.rf = randomForest(popularity~., data = train_data, importance=T)

##Insample
predicted <- predict(model.rf, train_data)
table(train_data$popularity, predicted)
mean(ifelse(train_data$popularity != predicted, 1, 0))

##Outsample
predicted <- predict(model.rf, validation_data)
table(validation_data$popularity, predicted)
mean(ifelse(validation_data$popularity != predicted, 1, 0))



#NN
model.nn <- nnet(popularity ~ ., data = train_data, size = 16)

##Insample
predicted <- predict(model.nn, train_data, type="class")
table(train_data$popularity, predicted)
mean(ifelse(train_data$popularity != predicted, 1, 0))

##Outsample
predicted <- predict(model.nn, validation_data, type="class")
table(validation_data$popularity, predicted)
mean(ifelse(validation_data$popularity != predicted, 1, 0))

#NN2
n <- names(train_data)
f <- as.formula(paste("viral+popular+mediocre~", paste(n[!n %in% c("viral", "popular", "mediocre")], collapse="+")))

model.nn1 <- neuralnet(f, data = train_data, hidden = c(16,10))

##Insample
predicted <- predict(model.nn, train_data, type="class")
table(train_data$popularity, predicted)
mean(ifelse(train_data$popularity != predicted, 1, 0))

##Outsample
predicted <- predict(model.nn, validation_data, type="class")
table(validation_data$popularity, predicted)
mean(ifelse(validation_data$popularity != predicted, 1, 0))







logistic.i <- predict(model.logistic, train_data)
logistic.o <- predict(model.logistic, validation_data)

svm.i <- predict(model.svm, train_data)
svm.o <- predict(model.svm, validation_data)

rf.i <- predict(model.rf, train_data)
rf.o <- predict(model.rf, validation_data)

nn.i <- predict(model.nn, train_data, type="class")
nn.o <- predict(model.nn, validation_data, type="class")

confusionMatrix(logistic.i,train_data$popularity)
confusionMatrix(logistic.o,validation_data$popularity)

confusionMatrix(svm.i,train_data$popularity)
confusionMatrix(svm.o,validation_data$popularity)

confusionMatrix(rf.i,train_data$popularity)
confusionMatrix(rf.o,validation_data$popularity)

confusionMatrix(nn.i,train_data$popularity)
confusionMatrix(nn.o,validation_data$popularity)


obj <- tune.svm(popularity~., data = train_data, 
                cost = 2^(2:8), 
                kernel = "linear")

