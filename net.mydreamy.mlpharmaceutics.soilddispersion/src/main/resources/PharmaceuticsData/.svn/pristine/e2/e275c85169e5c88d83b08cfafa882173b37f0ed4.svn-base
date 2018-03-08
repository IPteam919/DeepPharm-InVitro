rm(list = ls())

source("~/Desktop/PharmaceuticsData/HPMC/maxDissim.R")


allX<- read.csv("~/Desktop/PharmaceuticsData/HPMC/remaining_data.csv")
extra <- read.csv("~/Desktop/PharmaceuticsData/HPMC/final_testing_data.csv")


##scale range from 0 to 1
alldata <- data.matrix(allX)
extradata <- data.matrix(extra)

y <- alldata[,19:22]/100
X <- alldata[, 1:18]

extraY <- extradata[,19:22]/100
extraX <- extradata[, 1:18]

maxs <- apply(X, 2, max)
mins <- apply(X, 2, min)
ranges <- maxs - mins
means <- apply(X, 2, mean)
scaledallx <- scale(X, center = mins, scale = ranges)
scaleddata <- cbind(scaledallx, y)

scaledextrax <- scale(extraX, center = mins, scale = ranges)
scaledextradata <- cbind(scaledextrax, extraY)

## Get best inital dataset
numbers = dim(scaledallx)[1];

allIndexes <- NULL
allsumdiss <- NULL

times <- choose(numbers, 5)

for (i in 1:10000) {
## A random sample of 5 data points
set.seed(i+10000)
initalIndexes <- sample(numbers, 5)
allIndexes <- rbind(allIndexes, initalIndexes)

TrainningSet <- scaleddata[-initalIndexes, ]
initalTestSet <- scaleddata[initalIndexes, ]

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
RemainingSet <- scaleddata[-bestInitalIndex, ]
initalSet <- scaleddata[bestInitalIndex, ]

SelectedIndex <- maxDissim(initalSet, RemainingSet, n = 15, obj = minDiss)
SelectedSet <- RemainingSet[SelectedIndex, ]

FinalTestingSet <- rbind(initalSet, SelectedSet)
FinalTrainingSet <- RemainingSet[-SelectedIndex, ]

# compute un-scaled data
UnScaledRemainingSet <- alldata[-bestInitalIndex, ]
UnScaledinitalSet <- alldata[bestInitalIndex, ]
UnScaledSelectedSet <- UnScaledRemainingSet[SelectedIndex, ]
UnScaledFinalTestingSet <- rbind(UnScaledinitalSet, UnScaledSelectedSet)
UnScaledFinalTrainingSet <- UnScaledRemainingSet[-SelectedIndex, ]

#cat("Selected Indexes are: ", SelectedIndex, "\n", sep=",")
write.csv(FinalTestingSet, "~/Desktop/PharmaceuticsData/HPMC/testingset.csv", row.names = FALSE)
write.csv(FinalTrainingSet, "~/Desktop/PharmaceuticsData/HPMC/trainingset.csv", row.names = FALSE)
write.csv(UnScaledFinalTestingSet, "~/Desktop/PharmaceuticsData/HPMC/testingset(readyforcheck).csv", row.names = FALSE)
write.csv(UnScaledFinalTrainingSet, "~/Desktop/PharmaceuticsData/HPMC/trainingset(readyforcheck).csv", row.names = FALSE)
write.csv(scaledextradata, "~/Desktop/PharmaceuticsData/HPMC/extrascaledtestset.csv", row.names = FALSE)



