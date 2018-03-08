library(mlbench)
library(caret)
rm(list = ls())

allX <- read.csv("~/Desktop/PharmaceuticsData/OFDT/Remaining data-20161222.csv")

alldata <- data.matrix(allX)

y <- alldata[,27]/100
X <- alldata[, 1:26]


maxs <- apply(X, 2, max)
mins <- apply(X, 2, min)
ranges <- maxs - mins
means <- apply(X, 2, mean)

cat("maxs vertor: ", maxs, "\n", sep=",")
cat("mins vertor: ", mins, "\n", sep=",")

scaledallx <- scale(X, center = mins, scale = ranges)
scaleddata <- cbind(scaledallx, y)

set.seed(5)

numbers = dim(scaledallx)[1];

## A random sample of 5 data points
initalIndexes <- sample(numbers, 5)
#initalIndexes <- c(4,53,85,88,55)
#initalIndexes <- c(140,145,146,150,152)
#initalIndexes <- c(4,5,54,55,74,75,76,77,85,135,136,137)

TrainningSet <- scaledallx[-initalIndexes, ]
initalTestSet <- scaledallx[initalIndexes, ]

SelectedIndex <- maxDissim(initalTestSet, TrainningSet, n = 20)
FinalSelectedSet <- TrainningSet[SelectedIndex, ]

cat("Selected Indexes are: ", SelectedIndex+1, "\n", sep=",")
write.csv(alldata[SelectedIndex,], "selecteddata.csv", row.names = FALSE)

