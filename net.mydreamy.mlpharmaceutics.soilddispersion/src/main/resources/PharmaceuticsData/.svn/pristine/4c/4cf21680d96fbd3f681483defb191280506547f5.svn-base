library(mlbench)
library(caret)

rm(list = ls())

allX <- read.csv("~/Desktop/PharmaceuticsData/SD/2016122remainingdata.csv")
XXX <- read.csv("~/Desktop/PharmaceuticsData/SD/20161222finaltestdata.csv")
extra<- read.csv("~/Desktop/PharmaceuticsData/SD/extratestset.csv")


numbers = dim(allX)[1];
cols = dim(allX)[2];
y <- data.matrix(allX[,16:17])
X <- data.matrix(allX[,1:15])

extraY <- data.matrix(extra[,16:17])
extraX <- data.matrix(extra[,1:15])

#scaledallx <- scale(data.matrix(allX[,1:15]))
set.seed(5)

allXwithExtra <- rbind(X, extraX)

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


## A random sample of 5 data points

initalIndexes <- sample(numbers, 5)
#initalIndexes <- c(5,13,21,45,57)
#initalIndexes <- c(2,92,97,132,134)


#initalIndexes <- c(12,20,45,77,90)
#initalIndexes <- c(1,2,5,136,109)

TrainningSet <- scaledallx[-initalIndexes, ]
initalTestSet <- scaledallx[initalIndexes, ]

SelectedIndex <- maxDissim(initalTestSet, TrainningSet, n = 20)
FinalSelectedSet <- TrainningSet[SelectedIndex, ]

cat("Selected Indexes are: ", SelectedIndex+1, "\n", sep=",")
write.csv(scaleddata[SelectedIndex,], "~/Desktop/PharmaceuticsData/SD/testingsetabnormal.csv", row.names = FALSE)
write.csv(scaleddata[-SelectedIndex,], "~/Desktop/PharmaceuticsData/SD/trainingsetabnormal.csv", row.names = FALSE)
write.csv(scaleddata[-SelectedIndex,], "~/Desktop/PharmaceuticsData/SD/trainingsetabnormal.csv", row.names = FALSE)
write.csv(scaledextradata, "~/Desktop/PharmaceuticsData/SD/extrascaledtestset.csv", row.names = FALSE)




#final3m <- allX[SelectedIndex,-17]
#final6m <- allX[SelectedIndex,-16]
#write.csv(final3m, "~/Desktop/PharmaceuticsData/SD/selected3mdata.csv", row.names = FALSE)
#write.csv(final6m, "~/Desktop/PharmaceuticsData/SD/selected6mdata.csv", row.names = FALSE)
