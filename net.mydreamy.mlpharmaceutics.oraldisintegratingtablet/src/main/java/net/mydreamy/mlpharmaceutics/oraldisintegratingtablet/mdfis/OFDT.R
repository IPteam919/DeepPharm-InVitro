library(prodlim)
library(dplyr)

rm(list = ls())

source("~/Desktop/PharmaceuticsData/trunk/HPMC/maxDissim9.R")

allX <- read.csv("~/Desktop/PharmaceuticsData/trunk/OFDT/Remaining data-20161222.csv")
extra<- read.csv("~/Desktop/PharmaceuticsData/trunk/OFDT/Final testing data 20161222.csv")
allX <- rbind(allX, extra)

##scale range from 0 to 1
alldata <- data.matrix(allX)
extradata <- data.matrix(extra)

y <- alldata[,27]/100
X <- alldata[, 1:26]

extraY <- extradata[,27]/100
extraX <- extradata[, 1:26]

maxs <- apply(X, 2, max)
mins <- apply(X, 2, min)
ranges <- maxs - mins
means <- apply(X, 2, mean)
scaledallx <- scale(X, center = mins, scale = ranges)
scaleddata <- cbind(scaledallx, y)

scaledextrax <- scale(extraX, center = mins, scale = ranges)
scaledextradata <- cbind(scaledextrax, extraY)

## get rid of groups less 3 

df <- data.frame(scaledallx[, 1:9])
bb <- aggregate(list(numdup=rep(1,nrow(df))), df, length)
dd <- bb[order(bb$numdup, decreasing = FALSE),]
forbidgroup <- filter(dd, numdup <= 3)


conformIndex <- which(is.na(row.match(data.frame(scaleddata[, 1:9]), forbidgroup[, 1:9])))
#conformIndex <- c(conformIndex, index80)

conformdata <- scaleddata[conformIndex, ]
less3scaleddata <- scaleddata[-conformIndex, ]

conformalldata <- alldata[conformIndex, ]
less3conformalldata <- alldata[-conformIndex, ]





## Get best inital dataset
numbers = dim(conformdata)[1];

allIndexes <- NULL
allsumdiss <- NULL

times <- choose(numbers, 5)


## Generate 10000 intial data set and get best one
for (i in 1:10000) {
  ## A random sample of 5 data points
  set.seed(i)
  initalIndexes <- sample(numbers, 5)
  
  TrainningSet <- conformdata[-initalIndexes, ]
  initalTestSet <- conformdata[initalIndexes, ]
  
  allIndexes <- rbind(allIndexes, initalIndexes)
  
  diss <- proxy::dist(initalTestSet, TrainningSet)
  sumdiss <- sum(diss)
  allsumdiss <- c(allsumdiss, sumdiss)
  #initalIndexes <- c(14, 30, 46, 54, 91)
  #initalIndexes <- c(5,50,78,99,117)
  #initalIndexes <- c(18,64,65,66,84)
  #initalIndexes <- c(18,64,65,66,74,83,84)
  
}

bestInitalIndex <- allIndexes[which.min(allsumdiss), ]
bestDistance <- min(allsumdiss)

#Begin compute remaining testset
RemainingSet <- conformdata[-bestInitalIndex, ]
initalSet <- conformdata[bestInitalIndex, ]

SelectedIndex <- maxDissim(initalSet, RemainingSet, n = 15, obj = minDiss, alpha = 3)
SelectedSet <- RemainingSet[SelectedIndex, ]

FinalTestingSet <- rbind(initalSet, SelectedSet)
#FinalTestingSet <- SelectedSet
#FinalTrainingSet <- rbind(RemainingSet[-SelectedIndex, ], less3scaleddata, big80scaleddata, initalSet)
FinalTrainingSet <- rbind(RemainingSet[-SelectedIndex, ], less3scaleddata)


# compute un-scaled data
UnScaledRemainingSet <- conformalldata[-bestInitalIndex, ]
UnScaledinitalSet <- conformalldata[bestInitalIndex, ]
UnScaledSelectedSet <- UnScaledRemainingSet[SelectedIndex, ]
UnScaledFinalTestingSet <- rbind(UnScaledinitalSet, UnScaledSelectedSet)
#UnScaledFinalTestingSet <- UnScaledSelectedSet
#UnScaledFinalTrainingSet <- rbind(UnScaledRemainingSet[-SelectedIndex, ], less3conformalldata, big80alldata, UnScaledinitalSet)
UnScaledFinalTrainingSet <- rbind(UnScaledRemainingSet[-SelectedIndex, ], less3conformalldata)

#cat("Selected Indexes are: ", SelectedIndex, "\n", sep=",")
write.csv(FinalTestingSet, "~/Desktop/PharmaceuticsData/trunk/OFDT/testingset.csv", row.names = FALSE)
write.csv(FinalTrainingSet, "~/Desktop/PharmaceuticsData/trunk/OFDT/trainingset.csv", row.names = FALSE)
write.csv(UnScaledFinalTestingSet, "~/Desktop/PharmaceuticsData/trunk/OFDT/testingset(readyforcheck).csv", row.names = FALSE)
write.csv(UnScaledFinalTrainingSet, "~/Desktop/PharmaceuticsData/trunk/OFDT/trainingset(readyforcheck).csv", row.names = FALSE)
write.csv(scaledextradata, "~/Desktop/PharmaceuticsData/trunk/OFDT/extrascaledtestset.csv", row.names = FALSE)


