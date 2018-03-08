library(mlbench)
library(caret)

rm(list = ls())

allX<- read.csv("~/Desktop/PharmaceuticsData/OFDT/alldata.csv")

##range 0 to 1
alldata <- data.matrix(allX)
y <- alldata[,28]/100
X <- alldata[, 1:27]

maxs <- apply(X, 2, max)
mins <- apply(X, 2, min)
ranges <- maxs - mins
means <- apply(X, 2, mean)
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

SelectedIndex <- maxDissim(initalTestSet, TrainningSet, n = 15)
FinalSelectedSet <- TrainningSet[SelectedIndex, ]

cat("Selected Indexes are: ", SelectedIndex, "\n", sep=",")
write.csv(scaleddata[SelectedIndex,], "~/Desktop/PharmaceuticsData/OFDT/testingset.csv", row.names = FALSE)
write.csv(scaleddata[-SelectedIndex,], "~/Desktop/PharmaceuticsData/OFDT/trainingset.csv", row.names = FALSE)
write.csv(alldata[SelectedIndex,], "~/Desktop/PharmaceuticsData/OFDT/original-testingset.csv", row.names = FALSE)
write.csv(alldata[-SelectedIndex,], "~/Desktop/PharmaceuticsData/OFDT/original-trainingset.csv", row.names = FALSE)

