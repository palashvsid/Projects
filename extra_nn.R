#Load Data
newsx = read.csv('OnlineNewsPopularity.csv')
#news$shares <- NULL
#Data Cleaning
newsx$Date <- strsplit(as.character(newsx$url),"/")
newsx$Year <- sapply((newsx$Date), "[[", 4)
newsx$Month <- sapply(newsx$Date, "[[", 5)
newsx$Day <- sapply(newsx$Date, "[[", 6)
newsx$Date <- paste(newsx$Year,newsx$Month,newsx$Day,sep="-")
newsx$weekday <- NULL
newsx$Type <- ifelse(newsx$data_channel_is_bus==1, "Business",
                    ifelse(newsx$data_channel_is_lifestyle==1, "Lifestyle", 
                           ifelse(newsx$data_channel_is_entertainment==1, "Entertainment", 
                                  ifelse(newsx$data_channel_is_socmed==1, "SocialMedia", 
                                         ifelse(newsx$data_channel_is_tech==1, "Technology", 
                                                ifelse(newsx$data_channel_is_world==1, "World", "Others")
                                         )
                                  )
                           )
                    )
)

newsx$Type <- as.factor(newsx$Type)
summary(newsx$Type)
newsx <- subset(newsx, select = c(- data_channel_is_lifestyle, - data_channel_is_entertainment, -  data_channel_is_bus, - data_channel_is_socmed, - data_channel_is_tech, - data_channel_is_world))
newsx$url <- NULL
newsx$timedelta<- NULL
newsx$popularity_2classes <- 0
newsx[newsx$shares > median(newsx$shares), ]$popularity_2classes <- 1
newsx$Day <- NULL
newsx$Month <- NULL
newsx$Year <- NULL
newsx$Date <- NULL
newsx$LDA_04 <- NULL
newsx$is_weekend <- NULL
newsx$weekday_is_sunday <- NULL
newsx <- newsx[newsx$n_unique_tokens<699,]
newsx$shares <- NULL

#Test and Training Data
set.seed(1)
sample_index <- sample(1:nrow(newsx), 0.8*nrow(newsx))
train_datax <- newsx[sample_index,]
test_datax <- newsx[-sample_index,]
validation_sample_index <-sample(1:nrow(train_datax),0.25*nrow(train_datax))
validation_datax <- train_datax[validation_sample_index,]
train_datax <- train_datax[-validation_sample_index,]


#NN
maxs = apply(newsx[c(1, 2, 6, 7, 8, 12:23)], 2, max)
mins = apply(newsx[c(1, 2, 6, 7, 8, 12:23)], 2, min)
newsx[c(1, 2, 6, 7, 8, 12:23)] <-  as.data.frame(scale(newsx[c(1, 2, 6, 7, 8, 12:23)], center=mins, scale=maxs-mins))
newsx$business <- 0
newsx$entertainment <- 0
newsx$lifestyle <- 0
newsx$socialmedia <- 0
newsx$technology <- 0
newsx$world <- 0
newsx[newsx$Type=="Business",]$business <- 1
newsx[newsx$Type=="Entertainment",]$entertainment <- 1
newsx[newsx$Type=="Lifestyle",]$lifestyle <- 1
newsx[newsx$Type=="World",]$world <- 1
newsx[newsx$Type=="SocialMedia",]$socialmedia <- 1
newsx[newsx$Type=="Technology",]$technology <- 1
newsx$Type <- NULL
#news$business <- NULL
#news$entertainment <- NULL
#news$lifestyle <- NULL
#news$socialmedia <- NULL
#news$technology <- NULL
#news$world <- NULL

#Test and Training Data
set.seed(1)
sample_index <- sample(1:nrow(newsx), 0.8*nrow(newsx))
train_datax <- newsx[sample_index,]
test_datax <- newsx[-sample_index,]
validation_sample_index <-sample(1:nrow(train_datax),0.25*nrow(train_datax))
validation_datax <- train_datax[validation_sample_index,]
train_datax <- train_datax[-validation_sample_index,]


#NN2
n <- names(train_datax)
f <- as.formula(paste("popularity_2classes~", paste(n[!n %in% c("popularity_2classes")], collapse="+")))

model.nnx <- neuralnet(f, data = train_datax, hidden = c(16,10))

#Tuning NN
model.nnt <- train(popularity~., train_data, method='nnet', trace = FALSE,
               #Grid of tuning parameters to try:
               tuneGrid=expand.grid(.size=c(2:16), .decay=c(0,0.001,0.1))) 

model.nnxxt <- train(popularity_2classes ~ ., train_datax, method='nnet', trace = FALSE,
                   #Grid of tuning parameters to try:
                   tuneGrid=expand.grid(.size=c(2:16), .decay=c(0,0.001,0.1)))

model.nn <- nnet(popularity ~ ., data = train_data, size = 3, decay=0, maxit=500)

model.nnxx <- nnet(popularity_2classes ~ ., data = train_datax, size = 3, decay=0.1, maxit=500)


nn.i <- predict(model.nn, train_data, type="class")
nn.o <- predict(model.nn, validation_data, type="class")

confusionMatrix(nn.i,train_data$popularity)
confusionMatrix(nn.o,validation_data$popularity)

nnxx.i <- predict(model.nnxx, train_datax)
nnxx.i = as.numeric((nnxx.i >= 0.51))
nnxx.o <- predict(model.nnxx, validation_datax)
nnxx.o = as.numeric((nnxx.o >= 0.51))

confusionMatrix(nnxx.i,train_datax$popularity_2classes)
confusionMatrix(nnxx.o,validation_datax$popularity_2classes)


#model.nn #original 4 classes
#model.nn1 #neuralnet 4 classes
#model.nnx #neuralnet 2 classes
#model.nnxx #original 2 classes

prob.nnxx.outsample <- predict(model.nnxx, validation_datax, type="raw")
roc.plot(validation_datax$popularity_2classes == "1", prob.nnxx.outsample)
roc.plot(validation_datax$popularity_2classes == "1", prob.nnxx.outsample)$roc.vol

#Varaible Importance
barplot(model.rfx$importance[,1], horiz=T, las=2)

var_imp_rf1 <- as.data.frame((model.rf$importance[,1]*100)[rev(order(model.rf$importance[,1]))][1:10])
var_imp_rf1$Variable <- rownames(var_imp_rf1)
names(var_imp_rf1) <- c("Importance", "Variable")
ggplot(var_imp_rf1, aes(x=reorder(Variable, Importance), y=Importance))+geom_bar(stat="identity")+coord_flip()

var_imp_nn <- as.data.frame(varImp(model.nn)[,1][rev(order(varImp(model.nn)[,1]))][1:10])
var_imp_nn$Variable <- rownames(var_imp_nn)
var_imp_nnxx <- varImp(model.nnxx)
var_imp_nnxx$Variable <- rownames(var_imp_nnxx)
names(var_imp_nn) <- c("Importance", "Variable")
names(var_imp_nnxx) <- c("Importance", "Variable")
var_imp_nnxx <- var_imp_nnxx[order(var_imp_nnxx$Importance),][1:10,]
ggplot(var_imp_nn, aes(x=reorder(Variable, Importance), y=Importance))+geom_bar(stat="identity")+coord_flip()
ggplot(var_imp_nnxx, aes(x=reorder(Variable, Importance), y=Importance))+geom_bar(stat="identity")+coord_flip()

varImp(model.svm)
