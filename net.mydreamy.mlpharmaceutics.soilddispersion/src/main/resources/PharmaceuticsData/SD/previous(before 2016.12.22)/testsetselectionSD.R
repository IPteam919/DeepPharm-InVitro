library(mlbench)
library(caret)

allX <- read.csv("~/Desktop/PharmaceuticsData/SD/alldata.csv")

numbers = dim(allX)[1];
cols = dim(allX)[2];

scaledallx <- scale(data.matrix(allX[,1:15]))
set.seed(5)

## A random sample of 5 data points
#initalIndexes <- sample(numbers, 5)
#initalIndexes <- c(12,20,45,77,90)
initalIndexes <- c(1,2,5,136,109)

TrainningSet <- scaledallx[-initalIndexes, ]
initalTestSet <- scaledallx[initalIndexes, ]

SelectedIndex <- maxDissim(initalTestSet, TrainningSet, n = 15)
FinalSelectedSet <- TrainningSet[SelectedIndex, ]

cat("Selected Indexes are: ", SelectedIndex, "\n", sep=",")

final3m <- allX[SelectedIndex,-17]
final6m <- allX[SelectedIndex,-16]
write.csv(final3m, "~/Desktop/PharmaceuticsData/SD/selected3mdata.csv", row.names = FALSE)
write.csv(final6m, "~/Desktop/PharmaceuticsData/SD/selected6mdata.csv", row.names = FALSE)
