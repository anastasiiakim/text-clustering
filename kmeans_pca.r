library(FactoMineR)
library(cluster)
library(ggfortify)

data<-read.table('/sample_data/authors19th.txt', header=TRUE)
attach(data)

# plot the first 2 principal components 
data2 <- data[,-1]
rownames(data2) <- data[,1]
res.pca = PCA(data2, scale.unit=TRUE, ncp=7, graph=T)

# k-means
cluster <- kmeans(data2, centers=4)
cluster2 <- data.frame(rownames(data2),cluster$cluster)
colnames(cluster2) <- c("group","cluster")
cluster2[order(cluster2$cluster),]


# plot the first 2 principal components 
autoplot(kmeans(data2, 4), data = data2, label = TRUE, label.size = 2)


# Ward's clustering
d <- dist(data[-1], method = "euclidean")
sl<-hclust(d, method="ward.D")
plot(sl)
plot(sl,labels=author, main="Ward's method")
rect.hclust(sl,k=4)
