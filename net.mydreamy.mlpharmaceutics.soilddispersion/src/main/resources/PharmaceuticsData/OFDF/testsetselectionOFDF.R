library(mlbench)
library(caret)

rm(list = ls())

set.seed(5)


allX<- read.csv("/Users/yylonly/Desktop/PharmaceuticsData/OFDF/Remaining data 2016.12.22.csv")
extra<- read.csv("/Users/yylonly/Desktop/PharmaceuticsData/OFDF/extratestset.csv")

##range 0 to 1
alldata <- data.matrix(allX)
extradata <- data.matrix(extra)

y <- alldata[,17]/100
X <- alldata[, 1:16]

extraY <- extradata[,17]/100
extraX <- extradata[, 1:16]

maxs <- apply(X, 2, max)
mins <- apply(X, 2, min)
ranges <- maxs - mins
means <- apply(X, 2, mean)

cat("maxs vertor: ", maxs, "\n", sep=",")
cat("mins vertor: ", mins, "\n", sep=",")

scaledallx <- scale(X, center = mins, scale = ranges)
scaleddata <- cbind(scaledallx, y)

scaledextrax <- scale(extraX, center = mins, scale = ranges)
scaledextradata <- cbind(scaledextrax, extraY)

numbers = dim(scaledallx)[1];

## A random sample of 5 data points
initalIndexes <- sample(numbers, 5)
#initalIndexes <- c(9,36,51,80,97)
#initalIndexes <- c(44,69,70,84,107)


TrainningSet <- scaledallx[-initalIndexes, ]
initalTestSet <- scaledallx[initalIndexes, ]

SelectedIndex <- maxDissim(initalTestSet, TrainningSet, n = 20)
FinalSelectedSet <- TrainningSet[SelectedIndex, ]

cat("Selected Indexes are: ", SelectedIndex+1, "\n", sep=",")
write.csv(scaleddata[SelectedIndex,], "/Users/yylonly/Desktop/PharmaceuticsData/OFDF/testingset.csv", row.names = FALSE)
write.csv(scaleddata[-SelectedIndex,], "/Users/yylonly/Desktop/PharmaceuticsData/OFDF/trainingset.csv", row.names = FALSE)
write.csv(scaledextradata, "/Users/yylonly/Desktop/PharmaceuticsData/OFDF/extrascaledtestset.csv", row.names = FALSE)
