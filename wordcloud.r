library(wordcloud)
library(tm)

filePath<-"/.../book.txt"
text<-readLines(filePath)
text<-iconv(enc2utf8(text),sub="byte")

docs<-Corpus(VectorSource(text))

docs<-tm_map(docs,stripWhitespace)
  
docs<-tm_map(docs,tolower)
docs <- tm_map(docs, PlainTextDocument)
  
docs<-tm_map(docs,removeNumbers)
docs<-tm_map(docs,removePunctuation)
 
dtm<-TermDocumentMatrix(docs)
m <- as.matrix(dtm)

v <- sort(rowSums(m),decreasing=TRUE)
sum(v)

docs<-tm_map(docs,removeWords, stopwords('english'))
 
dtm<-TermDocumentMatrix(docs)
m <- as.matrix(dtm)
 
v1 <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v1),freq=v1, prop=v1/sum(v))
head(d, 10)


wordcloud(words = d$word, freq = d$freq, min.freq = 1, max.words=50, random.order=FALSE, rot.per=0.35, colors=brewer.pal(8, "Dark2"))
plot(w, main="Hearts of three (1920)")

