rm(list=ls())
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
news$popularity_2classes <- 0
news[news$shares > median(news$shares), ]$popularity_2classes <- 1
news$Day <- NULL
news$Month <- NULL
news$Year <- NULL
news$Date <- NULL
news$LDA_04 <- NULL
news$is_weekend <- NULL
news$weekday_is_sunday <- NULL
news <- news[news$n_unique_tokens<699,]
news$shares <- NULL
news$popularity_2classes <- as.factor(news$popularity_2classes)

#Test and Training Data
set.seed(1)
sample_index <- sample(1:nrow(news), 0.8*nrow(news))
train_data <- news[sample_index,]
test_data <- news[-sample_index,]
validation_sample_index <-sample(1:nrow(train_data),0.25*nrow(train_data))
validation_data <- train_data[validation_sample_index,]
train_data <- train_data[-validation_sample_index,]


#Logistic
model.logistic <- glm(popularity_2classes ~ ., family = binomial, train_data)

cost1 <- function(r, pi) {
  mean(((r == 0) & (pi > pcut)) | ((r == 1) & (pi < pcut)))}
searchgrid = seq(0.01, 1, 0.02)
result = cbind(searchgrid, NA)
for (i in 1:length(searchgrid)) {
  pcut <- result[i, 1]
  result[i, 2] <- cv.glm(data = train_data, glmfit = model.svm, cost = cost1, 
                         K = 3)$delta[2]
}
plot(result, ylab = "CV Cost")
result[which(result[,2]==min(result[,2]))]

#ROC
prob.glm.outsample <- predict(model.logistic, validation_data, type="response")
roc.plot(validation_data$popularity_2classes == "1", prob.glm.outsample)
roc.plot(validation_data$popularity_2classes == "1", prob.glm.outsample)$roc.vol

#Insample
prob.glm.insample <- predict(model.logistic, train_data, type="response")
predicted.glm.insample <- prob.glm.insample > 0.51
predicted.glm.insample <- as.numeric(predicted.glm.insample)
table(train_data$popularity_2classes, predicted.glm.insample, dnn = c("Truth", "Predicted"))
mean(ifelse(train_data$popularity_2classes != predicted.glm.insample, 1, 0))

#Outsample
predicted.glm.outsample <- prob.glm.outsample > 0.49
predicted.glm.outsample <- as.numeric(predicted.glm.outsample)
table(validation_data$popularity_2classes, predicted.glm.outsample, dnn = c("Truth", "Predicted"))
mean(ifelse(validation_data$popularity_2classes != predicted.glm.outsample, 1, 0))


#SVM
model.svmx = svm(popularity_2classes~., data = train_datax, cost = 1, gamma = 1/length(train_data))

#Insample
prob.svm = predict(model.svm, train_data)
pred.svm = as.numeric((prob.svm >= 0.51))
table(train_data$popularity_2classes, pred.svm, dnn = c("Obs", "Pred"))
mean(ifelse(train_data$popularity != pred.svm, 1, 0))

#Outsample
prob.svm = predict(model.svm, validation_data)
pred.svm = as.numeric((prob.svm >= 0.51))
table(validation_data$popularity_2classes, pred.svm, dnn = c("Obs", "Pred"))
mean(ifelse(validation_data$popularity != pred.svm, 1, 0))

prob.svm1 = predict(model.svm, validation_data, probability = TRUE)
prob.svm1 = attr(prob.svm1, "probabilities")[, 2]  #This is needed because prob.svm gives a 
pred.svm1 = as.numeric((prob.svm1 >= 0.51))
table(validation_data$popularity_2classes, pred.svm1, dnn = c("Obs", "Pred"))
mean(ifelse(validation_data$popularity != pred.svm1, 1, 0))

#ROC Curve
roc.plot(validation_data$popularity_2classes == "1", prob.svm)
roc.plot(validation_data$popularity_2classes == "1", prob.svm)$roc.vol



#RF
model.rfx = randomForest(popularity_2classes~., data = train_datax, importance=T)

#ROC
prob.rfx.outsample <- predict(model.rfx, validation_datax, type="prob")
roc.plot(validation_datax$popularity_2classes == "1", prob.rfx.outsample)
roc.plot(validation_datax$popularity_2classes == "1", prob.rfx.outsample)$roc.vol

#Insample
prob.rf.insample <- predict(model.rf, train_data)
predicted.rf.insample <- prob.rf.insample
predicted.rf.insample <- as.numeric(predicted.rf.insample)
table(train_data$popularity_2classes, predicted.rf.insample, dnn = c("Truth", "Predicted"))
mean(ifelse(train_data$popularity_2classes != predicted.rf.insample, 1, 0))

#Outsample
predicted.rf.outsample <- predict(model.rf, validation_data)
predicted.rf.outsample <- as.numeric(predicted.rf.outsample)
table(validation_data$popularity_2classes, predicted.rf.outsample, dnn = c("Truth", "Predicted"))
mean(ifelse(validation_data$popularity_2classes != predicted.rf.outsample, 1, 0))



#NN
news1 <- news
maxs = apply(news1[c(1, 2, 6, 7, 8, 12:23)], 2, max)
mins = apply(news1[c(1, 2, 6, 7, 8, 12:23)], 2, min)
news1[c(1, 2, 6, 7, 8, 12:23)] <-  as.data.frame(scale(news1[c(1, 2, 6, 7, 8, 12:23)], center=mins, scale=maxs-mins))
news1$business <- 0
news1$entertainment <- 0
news1$lifestyle <- 0
news1$socialmedia <- 0
news1$technology <- 0
news1$world <- 0
news1[news1$Type=="Business",]$business <- 1
news1[news1$Type=="Entertainment",]$entertainment <- 1
news1[news1$Type=="Lifestyle",]$lifestyle <- 1
news1[news1$Type=="World",]$world <- 1
news1[news1$Type=="SocialMedia",]$socialmedia <- 1
news1[news1$Type=="Technology",]$technology <- 1
news1$Type <- NULL
#news$business <- NULL
#news$entertainment <- NULL
#news$lifestyle <- NULL
#news$socialmedia <- NULL
#news$technology <- NULL
#news$world <- NULL

#Test and Training Data
set.seed(1)
sample_index <- sample(1:nrow(news1), 0.8*nrow(news1))
train_data <- news1[sample_index,]
test_data <- news1[-sample_index,]
validation_sample_index <-sample(1:nrow(train_data),0.25*nrow(train_data))
validation_data <- train_data[validation_sample_index,]
train_data <- train_data[-validation_sample_index,]


model.nn <- nnet(popularity_2classes ~ ., data = train_data, size = 16)

#ROC
prob.nn.outsample <- predict(model.nn, validation_data, type="raw")
roc.plot(validation_data$popularity_2classes == "1", prob.nn.outsample)
roc.plot(validation_data$popularity_2classes == "1", prob.nn.outsample)$roc.vol

##Insample
predicted <- predict(model.nn, train_data, type="class")

table(train_data$popularity, predicted)
mean(ifelse(train_data$popularity != predicted, 1, 0))

##Outsample
predicted <- predict(model.nn, validation_data, type="class")
table(validation_data$popularity, predicted)
mean(ifelse(validation_data$popularity != predicted, 1, 0))


#NN1
n <- names(train_data)
f <- as.formula(paste("popularity_2classes~", paste(n[!n %in% c("popularity_2classes")], collapse="+")))

model.nn1 <- neuralnet(f, data = train_data, hidden = c(16,11))

##Insample
predicted <- predict(model.nn, train_data, type="class")
table(train_data$popularity, predicted)
mean(ifelse(train_data$popularity != predicted, 1, 0))

##Outsample
predicted <- predict(model.nn, validation_data, type="class")
table(validation_data$popularity, predicted)
mean(ifelse(validation_data$popularity != predicted, 1, 0))


logistic.i <- predict(model.logistic, train_data)
logistic.i = as.numeric((logistic.i >= 0.51))
logistic.o <- predict(model.logistic, validation_data)
logistic.o = as.numeric((logistic.o >= 0.51))

svm.i <- predict(model.svm, train_data)
svm.i = as.numeric((svm.i >= 0.51))
svm.o <- predict(model.svm, validation_data)
svm.o = as.numeric((svm.o >= 0.51))

rf.i <- predict(model.rf, train_data)
rf.o <- predict(model.rf, validation_data)

nn.i <- predict(model.nn, train_data, type="class")
nn.o <- predict(model.nn, validation_data, type="class")

confusionMatrix(logistic.i,train_data$popularity_2classes)
confusionMatrix(logistic.o,validation_data$popularity_2classes)

confusionMatrix(svm.i,train_data$popularity_2classes)
confusionMatrix(svm.o,validation_data$popularity_2classes)

confusionMatrix(rf.i,train_data$popularity_2classes)
confusionMatrix(rf.o,validation_data$popularity_2classes)

confusionMatrix(nn.i,train_data$popularity_2classes)
confusionMatrix(nn.o,validation_data$popularity_2classes)