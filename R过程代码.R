setwd('C:\\Users\\Administrator\\Desktop\\kaggle')
short_train=read.csv('short_train.csv',encoding = 'UTF-8')
head(short_train)
tail(short_train)
summary(short_train)

####缺失值处理

short_train=na.omit(short_train)

library(tidyverse)
library(dplyr)
library(caret)
library(lubridate)
train_cl=short_train%>%group_by(小区编号)%>%summarise(Mean_up=mean(上行业务量GB),sd_up=sd(上行业务量GB),
                                         Max_up=max(上行业务量GB),Min_up=min(上行业务量GB),
                                         Mean_low=mean(下行业务量GB),sd_low=sd(下行业务量GB),
                                         Max_low=max(下行业务量GB),Min_low=min(下行业务量GB))
#### 小区聚类 #####
library(flexclust)
set.seed(2020)
clk<-cclust(train_cl[,-1],k=3,dist = "euclidean", method = "kmeans")
summary(clk)
table(clk@cluster)/nrow(train_cl)
clk@cluster
train_cl$type=factor(clk@cluster)
barchart(clk,legend=TRUE)

train_cl%>%group_by(type)
library(GGally)
library(ggsci)
res<-kmeans(train_cl[,-1],centers=3)

ggpairs(train_cl,columns = c(2,3,6,7,10),
        mapping=aes(colour=as.character(clk@cluster)))+theme_minimal()



str(short_train)
short_train$日期=as.Date(short_train$日期)
short_train$weekday=weekdays(short_train$日期)
short_train$week=week(short_train$日期)
short_train$month=month(short_train$日期)
short_train$day=day(short_train$日期)


data.frame(as.numeric(sub(':',"",substr(short_train$时间,1,2))),short_train$时间)
short_train$hour=as.numeric(sub(':',"",substr(short_train$时间,1,2)))

head(paste(short_train$日期,short_train$时间))
head(short_train)

short_train=short_train%>%arrange(小区编号,month,day,hour)


short_train$小区类别=NA
for(i in 1:nrow(train_cl)){
  short_train$小区类别[which(short_train$小区编号==train_cl$小区编号[i])]=train_cl$type[i]
}
summary(short_train)

#### EDA ####
arrange(short_train%>%group_by(weekday)%>%summarise(mean(上行业务量GB),sd(上行业务量GB),
                                            mean(下行业务量GB),sd(下行业务量GB)),by=`mean(上行业务量GB)`)
hour_train=short_train%>%group_by(小区类别,hour)%>%summarise(上行业务量均值=mean(上行业务量GB),上行业务量标准差=sd(上行业务量GB),
                                                           下行业务量均值=mean(下行业务量GB),下行业务量标准差=sd(下行业务量GB))
hour_train$小区类别=factor(hour_train$小区类别)

###周六上下行高峰特征
short_train$是否周六=ifelse(short_train$weekday=='星期六',1,0)

theme_set(theme_bw())

tail(arrange(short_train%>%group_by(hour)%>%summarise(上行业务量均值=mean(上行业务量GB),
                                                     下行业务量均值=mean(下行业务量GB)),by=上行业务量均值))
tail(arrange(short_train%>%group_by(hour)%>%summarise(上行业务量均值=mean(上行业务量GB),
                                                             下行业务量均值=mean(下行业务量GB)),by=下行业务量均值))
###小时高峰特征
short_train$高峰时段=ifelse(short_train$hour==12 | 
                          short_train$hour==18 | 
                          short_train$hour==19 | 
                          short_train$hour==22 | 
                          short_train$hour==21 | 
                          short_train$hour==20,1,0)

# plot
ggplot(hour_train, aes(x=hour)) + 
  geom_area(aes(y=下行业务量均值, fill=小区类别))+
  labs(title="各类别小区下行业务量均值时序图", 
       subtitle="2018年3月1日至4月19日期间各小时业务量均值", 
       y="业务量(GB)",
       x='小时')
ggplot(hour_train, aes(x=hour)) + 
  geom_area(aes(y=上行业务量均值, fill=小区类别))+
  labs(title="各类别小区上行业务量均值时序图", 
       subtitle="2018年3月1日至4月19日期间各小时业务量均值", 
       y="业务量(GB)",
       x='小时')

#### XGBoost ####
control<- trainControl(method = "repeatedcv", number = 3, repeats = 2, 
                       search = 'grid', verboseIter = T)

set.seed(2020)

tunegrid <- expand.grid(
  nrounds = c(150,200,250,300,350,400,450,500,600,700,800,900,1000) ,max_depth = c(4,5,6), eta = 0.4, gamma = 0, 
  colsample_bytree = 0.8, min_child_weight = 1 ,subsample = 0.75
)


### up model
names(short_train)
str(short_train)
xgbTree.fit <- train(上行业务量GB~weekday+week+month+day+hour+小区类别+是否周六+高峰时段+小区编号
                      , data=short_train
                      ,trControl=control,method="xgbTree",metric='RMSE')
xgbTree.fit.low <- train(下行业务量GB~weekday+week+month+day+hour+小区类别+是否周六+高峰时段+小区编号
                          , data=short_train
                          ,trControl=control,method="xgbTree",metric='RMSE')

plot(xgbTree.fit)
xgbTree.fit

#fine tune
xgbTree.fit.tune <- train(上行业务量GB~weekday+week+month+day+hour+小区类别+是否周六+高峰时段+小区编号
                          , data=short_train
                          ,trControl=control,tuneGrid=tunegrid,method="xgbTree",metric='RMSE')

plot(xgbTree.fit.tune)
xgbTree.fit.tune
xgbTree.fit.tune.low <- train(下行业务量GB~weekday+week+month+day+hour+小区类别+是否周六+高峰时段+小区编号
                               , data=short_train
                               ,trControl=control,tuneGrid=tunegrid,method="xgbTree",metric='RMSE')

###186小区验证
test_oof=predict(xgbTree.fit.tune,short_train%>%filter(小区编号==186))
off_186=short_train%>%filter(小区编号==186)%>%select(上行业务量GB)
summary(off_186)
plot(density(off_186$上行业务量GB))
plot(density(log(off_186$上行业务量GB)))
plot(density(short_train$上行业务量GB))
plot(density(log(short_train$上行业务量GB)))
plot(density(test_oof))

plot(1:nrow(off_186),off_186$上行业务量GB,type='l')
lines(test_oof,col=2)
test=read.csv('附件2：短期验证选择的小区数据集.csv')
head(test)
####对数变换####
theme_set(theme_bw())
# Plot
p1=ggplot(short_train, aes(上行业务量GB,fill=factor(小区类别)))+geom_density(alpha=0.8)+
  labs(title="密度分布图", 
       subtitle="各类别小区业务量(GB)密度分布对比",
       x="上行业务量（GB）",
       fill="# 小区类别")  
p2=ggplot(short_train, aes(log2(上行业务量GB),fill=factor(小区类别)))+geom_density(alpha=0.8)+
  labs(title="密度分布图", 
       subtitle="各类别小区业务量(GB)密度分布对比",
       x="log2(上行业务量GB)）",
       fill="# 小区类别") 

p3=ggplot(short_train, aes(下行业务量GB,fill=factor(小区类别)))+geom_density(alpha=0.8)+
  labs(title="密度分布图", 
       subtitle="各类别小区业务量(GB)密度分布对比",
       x="下行业务量（GB）",
       fill="# 小区类别")  
p4=ggplot(short_train, aes(log2(下行业务量GB),fill=factor(小区类别)))+geom_density(alpha=0.8)+
  labs(title="密度分布图", 
       subtitle="各类别小区业务量(GB)密度分布对比",
       x="log2(下行业务量GB)）",
       fill="# 小区类别") 
cowplot::plot_grid(p1, p2, p3, p4, nrow = 2, labels = LETTERS[1:4])   

g1=ggplot(short_train%>%filter(小区编号==186), aes(上行业务量GB))+geom_density(alpha=0.8,fill='red',col='red')+
  labs(title="", 
       subtitle="编号186小区业务量(GB)密度分布",
       x="上行业务量（GB）")  
g2=ggplot(short_train%>%filter(小区编号==186), aes(log2(上行业务量GB)))+geom_density(alpha=0.8,fill='blue',col='blue')+
  labs(title="", 
       subtitle="编号186小区业务量(GB)取对数后密度分布",
       x="log2(上行业务量（GB）)") 
cowplot::plot_grid(g1,g2, nrow = 1, labels = c('变换前','变换后'))  


plot(density(log2(short_train$上行业务量GB)))
plot(density(short_train$上行业务量GB))
plot(density(log2(short_train$下行业务量GB)))
plot(density(short_train$下行业务量GB))

#####测试集特征#####
test$日期=as.Date(test$日期)
test$weekday=weekdays(test$日期)
test$week=week(test$日期)
test$month=month(test$日期)
test$day=day(test$日期)
data.frame(as.numeric(sub(':',"",substr(test$时间,1,2))),test$时间)
test$hour=as.numeric(sub(':',"",substr(test$时间,1,2)))

test$小区类别=NA
for(i in 1:nrow(train_cl)){
  test$小区类别[which(test$小区编号==train_cl$小区编号[i])]=train_cl$type[i]
}
summary(test)
test$是否周六=ifelse(test$weekday=='星期六',1,0)
test$高峰时段=ifelse(test$hour==12 | 
                          test$hour==18 | 
                          test$hour==19 | 
                          test$hour==22 | 
                          test$hour==21 | 
                          test$hour==20,1,0)




test_pre=predict(xgbTree.fit,test%>%filter(小区编号==186))
test_pre=ifelse(test_pre<0,0,test_pre)
test_pre.tune=predict(xgbTree.fit.tune,test%>%filter(小区编号==186))
test_pre.tune=ifelse(test_pre.tune<0,0,test_pre.tune)
length(test_pre)
par(mfrow=c(3,1))
plot(1:(nrow(off_186)-1000+169),c(off_186$上行业务量GB[1:(nrow(off_186)-1000)],rep(NA,169)),type='l',
     ylim=c(-0.001,0.11),xlab='Time',ylab='上行业务量GB',main='编号186小区未来一周小时级流量变化时序预测')
lines((nrow(off_186)-1000+2):(nrow(off_186)-1000+169),test_pre,col=2,lty=1,lwd=2)
legend('topright',col = c(1,2),lty=c(1,1),lwd=c(1,2),legend = c('原序列最后一周小时级流量','XGBoost(原始特征)'),cex=1)

plot(1:(nrow(off_186)-1000+169),c(off_186$上行业务量GB[1:(nrow(off_186)-1000)],rep(NA,169)),type='l',
     ylim=c(-0.001,0.15),xlab='Time',ylab='上行业务量GB',main='编号186小区未来一周小时级流量变化时序预测')
lines((nrow(off_186)-1000+2):(nrow(off_186)-1000+169),test_pre.tune,col=4,lwd=2)
legend('topleft',col = c(1,4),lty=c(1,1),lwd=c(1,2),legend = c('原序列最后一周小时级流量','XGBoost(特征工程+调参后)'),cex=1)


plot(1:(nrow(off_186)-1000+169),c(off_186$上行业务量GB[1:(nrow(off_186)-1000)],rep(NA,169)),type='l',
     ylim=c(-0.001,0.15),xlab='Time',ylab='上行业务量GB',main='编号186小区未来一周小时级流量变化时序预测')
lines((nrow(off_186)-1000+2):(nrow(off_186)-1000+169),(test_pre.tune+test_pre)/2,col='purple',lwd=2)
summary(test_pre.tune)

####最终XGBoost####
tunegrid <- expand.grid(
  nrounds = 500 ,max_depth = 5, eta = 0.4, gamma = 0, 
  colsample_bytree = 0.8, min_child_weight = 1 ,subsample = 0.75
)
#目标变换
short_train$log上行业务量GB=ifelse(log2(short_train$上行业务量GB)==-Inf,1,log2(short_train$上行业务量GB))
short_train$log下行业务量GB=ifelse(log2(short_train$下行业务量GB)==-Inf,1,log2(short_train$下行业务量GB))


xgbTree.fit.final <- train(log上行业务量GB~weekday+week+month+day+hour+小区类别+是否周六+高峰时段+小区编号
                           , data=short_train
                           ,trControl=control,method="xgbTree",metric='RMSE')

xgbTree.fit.final
xgbTree.fit.final.low <- train(log下行业务量GB~weekday+week+month+day+hour+小区类别+是否周六+高峰时段+小区编号
                           , data=short_train
                           ,trControl=control,method="xgbTree",metric='RMSE')

xgbTree.fit.final.low
xgb_pre186=2^predict(xgbTree.fit.final,train_186)
summary(xgb_pre186)
sqrt(mean((xgb_pre186-train_186$上行业务量GB)^2))# 0.02016879

xgb_pre=2^predict(xgbTree.fit.final,short_train)
summary(xgb_pre)
sqrt(mean((xgb_pre-short_train$上行业务量GB)^2))

xgb_pre_test=predict(xgbTree.fit.final,test%>%filter(小区编号==186))
xgb_pre_test=2^xgb_pre_test
summary(xgb_pre_test)
plot(1:(nrow(off_186)-1000+169),c(off_186$上行业务量GB[1:(nrow(off_186)-1000)],rep(NA,169)),type='l',
     ylim=c(-0.001,0.11),xlab='Time',ylab='上行业务量GB',main='编号186小区未来一周小时级流量变化时序预测')
lines((nrow(off_186)-1000+2):(nrow(off_186)-1000+169),xgb_pre_test,col='#f8766d',lwd=2)
legend('topright',col = c(1,'#f8766d'),lty=c(1,1),lwd=c(1,2),legend = c('原序列最后一周小时级流量','XGBoost(特征工程+调参+目标变换)'),cex=1)




#### Prophet ####
par(mfrow=c(3,1))
library(prophet)
train_186 = short_train%>%filter(小区编号==186)
plot(density(log2(train_186$上行业务量GB)))
plot(density(train_186$上行业务量GB))

ds<-as.POSIXct(paste(train_186$日期,train_186$时间), origin="1970-01-01 00:00:00")
###原序列值
nextweek<-data.frame(ds,y=train_186$上行业务量GB)
m<-prophet(nextweek)
m
future <- make_future_dataframe(m, periods = length(test_pre),freq = 'hour')
tail(future)
forecast_result <- predict(m, future)
tail(forecast_result[c('ds', 'yhat', 'yhat_lower', 'yhat_upper')])
head(forecast_result[c('ds', 'yhat', 'yhat_lower', 'yhat_upper')])
plot(m, forecast_result)


sqrt(mean((forecast_result$yhat[1:nrow(train_186)]-train_186$上行业务量GB)^2))#0.02128192

prophet186_pre=forecast_result$yhat[(nrow(train_186)+1):(nrow(train_186)+length(test_pre))]
summary(prophet186_pre)
prophet186_pre=ifelse(prophet186_pre<0,0,prophet186_pre)
plot(1:(nrow(off_186)-1000+169),c(off_186$上行业务量GB[1:(nrow(off_186)-1000)],rep(NA,169)),type='l',
     ylim=c(-0.001,0.11),xlab='Time',ylab='上行业务量GB',main='编号186小区未来一周小时级流量变化时序预测')
lines((nrow(off_186)-1000+2):(nrow(off_186)-1000+169),prophet186_pre,col='purple',lwd=2)
legend('topright',col = c(1,'purple'),lty=c(1,1),lwd=c(1,2),legend = c('原序列最后一周小时级流量','Prophet(原序列值)'),cex=1)

####取对数
nextweek<-data.frame(ds,y=log2(train_186$上行业务量GB))
m2<-prophet(nextweek)
m2
future <- make_future_dataframe(m2, periods = length(test_pre),freq = 'hour')
tail(future)
forecast_result2 <- predict(m2, future)
tail(forecast_result2[c('ds', 'yhat', 'yhat_lower', 'yhat_upper')])
head(forecast_result2[c('ds', 'yhat', 'yhat_lower', 'yhat_upper')])
#plot(m, forecast_result)

##186下行
nextweek<-data.frame(ds,y=train_186$下行业务量GB)
m3<-prophet(nextweek)
m3
future <- make_future_dataframe(m3, periods = length(test_pre),freq = 'hour')
forecast_result.low <- predict(m3, future)

sqrt(mean((2^(forecast_result.low$yhat[1:nrow(train_186)])-train_186$下行业务量GB)^2))

cbind(MAE(forecast_result.low$yhat[1:(nrow(forecast_result.low)-168)],2^log2(train_186$上行业务量GB)),
      RMSE(2^forecast_result.low$yhat[1:(nrow(forecast_result.low)-168)],2^log2(train_186$上行业务量GB)),
      R2(2^forecast_result.low$yhat[1:(nrow(forecast_result.low)-168)],train_186$上行业务量GB))

nextweek<-data.frame(ds,y=log2(train_186$下行业务量GB))
m4<-prophet(nextweek)
m4
future <- make_future_dataframe(m4, periods = length(test_pre),freq = 'hour')
forecast_result.low2 <- predict(m4, future)

cbind(MAE(2^forecast_result.low2$yhat[1:(nrow(forecast_result.low)-168)],2^log2(train_186$下行业务量GB)),
      RMSE(2^forecast_result.low2$yhat[1:(nrow(forecast_result.low)-168)],2^log2(train_186$下行业务量GB)),
      R2(2^forecast_result.low2$yhat[1:(nrow(forecast_result.low)-168)],train_186$下行业务量GB))

prophet186_pre2=2^(forecast_result2$yhat[(nrow(train_186)+1):(nrow(train_186)+length(test_pre))])
summary(prophet186_pre2)

plot(1:(nrow(off_186)-1000+169),c(off_186$上行业务量GB[1:(nrow(off_186)-1000)],rep(NA,169)),type='l',
     ylim=c(-0.001,0.11),xlab='Time',ylab='上行业务量GB',main='编号186小区未来一周小时级流量变化时序预测')
lines((nrow(off_186)-1000+2):(nrow(off_186)-1000+169),prophet186_pre2,col='skyblue',lwd=2)
legend('topright',col = c(1,'skyblue'),lty=c(1,1),lwd=c(1,2),legend = c('原序列最后一周小时级流量','Prophet(原序列值取对数后)'),cex=1)


#### prophet 上行短期 ##### 

train_up_list=list()
for(i in 1:nrow(train_cl)){
  train_up_list[i]=short_train%>%filter(小区编号==train_cl$小区编号[i])%>%select(上行业务量GB)
}
train_up_list

pre_p_up=matrix(NA,length(test_pre),nrow(train_cl))
for(i in 1:nrow(train_cl)){
  sb=prophet(data.frame(ds=ds[1:length(unlist(train_up_list[i]))],y=as.numeric(unlist(train_up_list[i]))))
  print(paste0("*********",'model',i,"*********"))
  pre_p_up[,i]=as.numeric(unlist(tail(predict(sb, make_future_dataframe(sb, periods = 168,freq = 'hour'))['yhat'],168)))
}
pre_p_up
test_dat=c()
for(i in 1:nrow(train_cl)){
  test_dat=c(test_dat,nrow(test%>%filter(小区编号==train_cl$小区编号[i])))
}
train_cl$小区编号[which(test_dat<168)]##日期异常小区编号
test_time=test%>%filter(小区编号==train_cl$小区编号[1])%>%select(日期,时间)
test_abnormal=data.frame(test_time,pre_p_up[,which(test_dat<168)])##日期异常小区编号及对应时间

n=length(train_cl$小区编号[which(test_dat<168)]);n

####异常小区检测提取
final_abnormal_test=list()
for(j in 1:n){
  test1=test%>%filter(小区编号==train_cl$小区编号[which(test_dat<168)][j])%>%select(日期,时间)
  test1_num=c()
  for(i in 1:nrow(test1)){
    test1_num=c(test1_num,which(test_abnormal$日期==test1$日期[i] & test_abnormal$时间==test1$时间[i]))
    final_abnormal_test[j]=data.frame(up=test_abnormal[test1_num,j+2])
  }
}
final_abnormal_test
for(i in 1:n){
  test$上行业务量GB[test$小区编号==train_cl$小区编号[which(test_dat<168)][i]]=as.numeric(unlist(final_abnormal_test[i]))
}
dim(pre_p_up)



colnames(pre_p_up)=train_cl$小区编号
summary(test)
####正常小区
N=length(which(test_dat==168));N
for(i in 1:N){
  test$上行业务量GB[test$小区编号==train_cl$小区编号[which(test_dat==168)][i]]=pre_p_up[,i]
}
summary(test)

#### prophet 下行短期 ##### 

train_low_list=list()
for(i in 1:nrow(train_cl)){
  train_low_list[i]=short_train%>%filter(小区编号==train_cl$小区编号[i])%>%select(下行业务量GB)
}
train_low_list

pre_p_low=matrix(NA,length(test_pre),nrow(train_cl))
for(i in 1:nrow(train_cl)){
  sb=prophet(data.frame(ds=ds[1:length(unlist(train_low_list[i]))],y=as.numeric(unlist(train_low_list[i]))))
  print(paste0("*********",'model',i,"*********"))
  pre_p_low[,i]=as.numeric(unlist(tail(predict(sb, make_future_dataframe(sb, periods = 168,freq = 'hour'))['yhat'],168)))
}
pre_p_low
test_dat=c()
for(i in 1:nrow(train_cl)){
  test_dat=c(test_dat,nrow(test%>%filter(小区编号==train_cl$小区编号[i])))
}
train_cl$小区编号[which(test_dat<168)]##日期异常小区编号
test_time=test%>%filter(小区编号==train_cl$小区编号[1])%>%select(日期,时间)
test_abnormal=data.frame(test_time,pre_p_low[,which(test_dat<168)])##日期异常小区编号及对应时间

n=length(train_cl$小区编号[which(test_dat<168)]);n

####异常小区检测提取
final_abnormal_test=list()
for(j in 1:n){
  test1=test%>%filter(小区编号==train_cl$小区编号[which(test_dat<168)][j])%>%select(日期,时间)
  test1_num=c()
  for(i in 1:nrow(test1)){
    test1_num=c(test1_num,which(test_abnormal$日期==test1$日期[i] & test_abnormal$时间==test1$时间[i]))
    final_abnormal_test[j]=data.frame(low=test_abnormal[test1_num,j+2])
  }
}
final_abnormal_test
for(i in 1:n){
  test$下行业务量GB[test$小区编号==train_cl$小区编号[which(test_dat<168)][i]]=as.numeric(unlist(final_abnormal_test[i]))
}
dim(pre_p_low)



colnames(pre_p_low)=train_cl$小区编号
summary(test)
####正常小区
N=length(which(test_dat==168));N
for(i in 1:N){
  test$下行业务量GB[test$小区编号==train_cl$小区编号[which(test_dat==168)][i]]=pre_p_low[,i]
}
summary(test)

prophet_test = test ###prophet_test结果
write.csv(prophet_test,'短期验证prophet.csv',row.names = F)
names(prophet_test)
long_test_dat=prophet_test%>%group_by(小区类别,month,day,weekday)%>%summarise(上行业务量GB均值=mean(上行业务量GB),
                                                              下行业务量GB均值=mean(下行业务量GB))
write.csv(long_test_dat,'long_test_dat_week.csv',row.names = F)
#####最终模型####
xgbTree.fit#0.1 test_pre
xgbTree.fit.tune#0.1 test_pre.tune
xgbTree.fit.final#0.3 xgb_pre_test
prophet186_pre#0.3
prophet186_pre2#0.2
final_pre_186=0.1*test_pre+0.2*test_pre.tune+0.3*xgb_pre_test+0.2*prophet186_pre+0.2*prophet186_pre2

plot(1:(nrow(off_186)-1000+169),c(off_186$上行业务量GB[1:(nrow(off_186)-1000)],rep(NA,169)),type='l',
     ylim=c(-0.001,0.11),xlab='Time',ylab='上行业务量GB',main='编号186小区未来一周小时级流量变化时序预测')
lines((nrow(off_186)-1000+2):(nrow(off_186)-1000+169),final_pre_186,col=1,lwd=1,lty=5)
legend('topright',col = c(1,1),lty=c(1,5),lwd=c(1,1),legend = c('原序列最后一周小时级流量','XGBoost+Prophet'),cex=1)

####### 问题1 预测结果 #######
test_pre_up_totoal1=predict(xgbTree.fit,test)
test_pre_up_totoal2=predict(xgbTree.fit.tune,test)
test_pre_up_totoal3=2^(predict(xgbTree.fit.final,test))
test_pre_up_totoal=0.1*test_pre_up_totoal1+0.2*test_pre_up_totoal2+0.3*test_pre_up_totoal3
summary(test_pre_up_totoal)
test_pre_up_totoal=ifelse(test_pre_up_totoal<0,0,test_pre_up_totoal)

test_pre_low_totoal1=predict(xgbTree.fit.low,test)
test_pre_low_totoal2=predict(xgbTree.fit.tune.low,test)
test_pre_low_totoal3=2^(predict(xgbTree.fit.final.low,test))
test_pre_low_totoal=0.1*test_pre_low_totoal1+0.2*test_pre_low_totoal2+0.3*test_pre_low_totoal3
summary(test_pre_low_totoal)
test_pre_low_totoal=ifelse(test_pre_low_totoal<0,0,test_pre_low_totoal)

test$上行业务量GB=0.4*ifelse(prophet_test$上行业务量GB<0,0,prophet_test$上行业务量GB)+test_pre_up_totoal
test$下行业务量GB=0.4*ifelse(prophet_test$下行业务量GB<0,0,prophet_test$下行业务量GB)+test_pre_low_totoal

plot(density(test$上行业务量GB))
plot(density(test$下行业务量GB))

summary(test)
summary(short_train)
#write.csv(test,'短期验证结果.csv',row.names = F)

final_long_day=rbind(short_train%>%select(日期,时间,小区编号,上行业务量GB,下行业务量GB,month,day,hour,小区类别),
      test%>%select(日期,时间,小区编号,上行业务量GB,下行业务量GB,month,day,hour,小区类别))
final_long_day=final_long_day%>%group_by(month,day,小区类别)%>%summarise(上行业务量均值=mean(上行业务量GB),下行业务量均值=mean(下行业务量GB))
final_long_day
write.csv(final_long_day,'final_long_day.csv',row.names = F)

#####短期验证提交#####
sub_test=read.csv('附件2：短期验证选择的小区数据集.csv')
head(sub_test)
final_sub_test=test%>%select(日期,时间,小区编号,上行业务量GB,下行业务量GB)
data.frame(sub_test[,1:3],final_sub_test)
sum(ifelse(sub_test$日期==final_sub_test$日期 & sub_test$时间==final_sub_test$时间 &
           sub_test$小区编号==final_sub_test$小区编号,0,1))
final_sub_test$日期=sub_test$日期
final_sub_test$时间=sub_test$时间
write.csv(final_sub_test,'短期验证结果.csv',row.names = F)


#####模型对比####
###Mean
apply(xgbTree.fit$results[,c('RMSE','Rsquared','MAE')],2,mean)
apply(xgbTree.fit.tune$results[,c('RMSE','Rsquared','MAE')],2,mean)
apply(xgbTree.fit.final$results[,c('RMSE','Rsquared','MAE')],2,mean)

apply(xgbTree.fit.low$results[,c('RMSE','Rsquared','MAE')],2,mean)
apply(xgbTree.fit.tune.low$results[,c('RMSE','Rsquared','MAE')],2,mean)
apply(xgbTree.fit.final.low$results[,c('RMSE','Rsquared','MAE')],2,mean)
###best RMSE
#up
rbind(xgbTree.fit$results[,c('MAE','RMSE','Rsquared')][which(xgbTree.fit$results[,'RMSE']==min(xgbTree.fit$results[,'RMSE'])),]
,xgbTree.fit.tune$results[,c('MAE','RMSE','Rsquared')][which(xgbTree.fit.tune$results[,'RMSE']==min(xgbTree.fit.tune$results[,'RMSE'])),]
,xgbTree.fit.final$results[,c('MAE','RMSE','Rsquared')][which(xgbTree.fit.final$results[,'RMSE']==min(xgbTree.fit.final$results[,'RMSE'])),])



#low
rbind(xgbTree.fit.low$results[,c('MAE','RMSE','Rsquared')][which(xgbTree.fit.low$results[,'RMSE']==min(xgbTree.fit.low$results[,'RMSE'])),]
,xgbTree.fit.tune.low$results[,c('MAE','RMSE','Rsquared')][which(xgbTree.fit.tune.low$results[,'RMSE']==min(xgbTree.fit.tune.low$results[,'RMSE'])),]
,xgbTree.fit.final.low$results[,c('MAE','RMSE','Rsquared')][which(xgbTree.fit.final.low$results[,'RMSE']==min(xgbTree.fit.final.low$results[,'RMSE'])),])
val=short_train[sample(1:nrow(short_train),251*168),]

c(mean(abs(val$下行业务量GB-2^predict(xgbTree.fit.final.low,val))),
sqrt(mean((val$下行业务量GB-2^predict(xgbTree.fit.final.low,val))^2)))

c(mean(abs(val$下行业务量GB-2^predict(xgbTree.fit.final,val))),
  sqrt(mean((val$下行业务量GB-2^predict(xgbTree.fit.final,val))^2)))

library(greybox)
###训练集
##原始特征
#up
XGB1=data.frame(MAE=MAE(short_train$上行业务量GB,predict(xgbTree.fit,short_train)),
           RMSE=RMSE(short_train$上行业务量GB,predict(xgbTree.fit,short_train)),
           R2=R2(short_train$上行业务量GB,predict(xgbTree.fit,short_train)))
XGB1
#low
XGB.low1=data.frame(MAE=MAE(short_train$上行业务量GB,predict(xgbTree.fit.low,short_train)),
                   RMSE=RMSE(short_train$上行业务量GB,predict(xgbTree.fit.low,short_train)),
                   R2=R2(short_train$上行业务量GB,predict(xgbTree.fit.low,short_train)))
XGB.low1
##特征工程+调参
#up
XGB2=data.frame(MAE=MAE(short_train$上行业务量GB,predict(xgbTree.fit.tune,short_train)),
                RMSE=RMSE(short_train$上行业务量GB,predict(xgbTree.fit.tune,short_train)),
                R2=R2(short_train$上行业务量GB,predict(xgbTree.fit.tune,short_train)))
XGB2

#low
XGB.low2=data.frame(MAE=MAE(short_train$上行业务量GB,predict(xgbTree.fit.tune.low,short_train)),
                    RMSE=RMSE(short_train$上行业务量GB,predict(xgbTree.fit.tune.low,short_train)),
                    R2=R2(short_train$上行业务量GB,predict(xgbTree.fit.tune.low,short_train)))
XGB.low2

##特征工程+调参+目标变换
#up
XGB3=data.frame(MAE=MAE(short_train$上行业务量GB,2^predict(xgbTree.fit.final,short_train)),
                RMSE=RMSE(short_train$上行业务量GB,2^predict(xgbTree.fit.final,short_train)),
                R2=R2(short_train$上行业务量GB,2^predict(xgbTree.fit.final,short_train)))
XGB3

#low
XGB.low3=data.frame(MAE=MAE(short_train$上行业务量GB,2^predict(xgbTree.fit.final.low,short_train)),
                    RMSE=RMSE(short_train$上行业务量GB,2^predict(xgbTree.fit.final.low,short_train)),
                    R2=R2(short_train$上行业务量GB,2^predict(xgbTree.fit.final.low,short_train)))
XGB.low3

rbind(XGB1,XGB2,XGB3)
rbind(XGB.low1,XGB.low2,XGB.low3)


###prophet验证####
pre_val_up=list()
for(i in 1:nrow(train_cl)){
  sb=prophet(data.frame(ds=ds[1:length(unlist(train_up_list[i]))],y=as.numeric(unlist(train_up_list[i]))))
  print(paste0("*********",'model',i,"*********"))
  pre_val_up[i]=data.frame(pred=as.numeric(unlist(head(predict(sb, make_future_dataframe(sb, periods = 168,freq = 'hour'))['yhat'],length(unlist(train_up_list[i]))))))
}
RMSE_up=c()
for(i in 1:251){
  RMSE_up=c(RMSE_up,sqrt(mean((as.numeric(unlist(pre_val_up[i]))-as.numeric(unlist(train_up_list[i])))^2)))
}
mean(RMSE_up)

MAE_up=c()
for(i in 1:251){
  MAE_up=c(MAE_up,(mean(abs(as.numeric(unlist(pre_val_up[i]))-as.numeric(unlist(train_up_list[i]))))))
}
mean(MAE_up)

R2_up=c()
for(i in 1:251){
  R2_up=c(R2_up,sum((as.numeric(unlist(pre_val_up[i]))-as.numeric(unlist(train_up_list[i])))^2)/sum((as.numeric(unlist(train_up_list[i]))-mean(as.numeric(unlist(pre_val_up[i]))))^2))
}
mean(R2_up)

up_metric=c(mean(MAE_up),mean(RMSE_up),mean(R2_up));up_metric

pre_val_low=list()
for(i in 1:nrow(train_cl)){
  sb=prophet(data.frame(ds=ds[1:length(unlist(train_low_list[i]))],y=as.numeric(unlist(train_low_list[i]))))
  print(paste0("*********",'model',i,"*********"))
  pre_val_low[i]=data.frame(pred=as.numeric(unlist(head(predict(sb, make_future_dataframe(sb, periods = 168,freq = 'hour'))['yhat'],length(unlist(train_low_list[i]))))))
}

RMSE_low=c()
for(i in 1:251){
  RMSE_low=c(RMSE_low,sqrt(mean((as.numeric(unlist(pre_val_low[i]))-as.numeric(unlist(train_low_list[i])))^2)))
}
mean(RMSE_low)

MAE_low=c()
for(i in 1:251){
  MAE_low=c(MAE_low,(mean(abs(as.numeric(unlist(pre_val_low[i]))-as.numeric(unlist(train_low_list[i]))))))
}
mean(MAE_low)

R2_low=c()
for(i in 1:251){
  R2_low=c(R2_low,sum((as.numeric(unlist(pre_val_low[i]))-as.numeric(unlist(train_low_list[i])))^2)/sum((as.numeric(unlist(train_low_list[i]))-mean(as.numeric(unlist(pre_val_low[i]))))^2))
}
mean(R2_low)

low_metric=c(mean(MAE_low),mean(RMSE_low),mean(R2_low));low_metric


###log目标
pre_val_logup=list()
for(i in 1:nrow(train_cl)){
  sb=prophet(data.frame(ds=ds[1:length(unlist(train_up_list[i]))],y=ifelse(log2(as.numeric(unlist(train_up_list[i])))==-Inf,1,log2(as.numeric(unlist(train_up_list[i]))))))
  print(paste0("*********",'model',i,"*********"))
  pre_val_logup[i]=data.frame(pred=2^as.numeric(unlist(head(predict(sb, make_future_dataframe(sb, periods = 168,freq = 'hour'))['yhat'],length(unlist(train_up_list[i]))))))
}
RMSE_uplog=c()
for(i in 1:251){
  RMSE_uplog=c(RMSE_uplog,sqrt(mean((as.numeric(unlist(pre_val_logup[i]))-as.numeric(unlist(train_up_list[i])))^2)))
}
mean(RMSE_uplog)

MAE_uplog=c()
for(i in 1:251){
  MAE_uplog=c(MAE_uplog,(mean(abs(as.numeric(unlist(pre_val_logup[i]))-as.numeric(unlist(train_up_list[i]))))))
}
mean(MAE_uplog)

R2_uplog=c()
for(i in 1:251){
  R2_uplog=c(R2_uplog,sum((as.numeric(unlist(pre_val_logup[i]))-as.numeric(unlist(train_up_list[i])))^2)/sum((as.numeric(unlist(train_up_list[i]))-mean(as.numeric(unlist(pre_val_logup[i]))))^2))
}
mean(R2_uplog)

uplog_metric=c(mean(MAE_uplog),mean(RMSE_uplog),mean(R2_uplog));uplog_metric


pre_val_loglow=list()
for(i in 1:nrow(train_cl)){
  sb=prophet(data.frame(ds=ds[1:length(unlist(train_low_list[i]))],y=ifelse(log2(as.numeric(unlist(train_low_list[i])))==-Inf,1,log2(as.numeric(unlist(train_low_list[i]))))))
  print(paste0("*********",'model',i,"*********"))
  pre_val_loglow[i]=data.frame(pred=2^as.numeric(unlist(head(predict(sb, make_future_dataframe(sb, periods = 168,freq = 'hour'))['yhat'],length(unlist(train_low_list[i]))))))
}
RMSE_lowlog=c()
for(i in 1:251){
  RMSE_lowlog=c(RMSE_lowlog,sqrt(mean((as.numeric(unlist(pre_val_loglow[i]))-as.numeric(unlist(train_low_list[i])))^2)))
}
mean(RMSE_lowlog)

MAE_lowlog=c()
for(i in 1:251){
  MAE_lowlog=c(MAE_lowlog,(mean(abs(as.numeric(unlist(pre_val_loglow[i]))-as.numeric(unlist(train_low_list[i]))))))
}
mean(MAE_lowlog)

R2_lowlog=c()
for(i in 1:251){
  R2_lowlog=c(R2_lowlog,sum((as.numeric(unlist(pre_val_loglow[i]))-as.numeric(unlist(train_low_list[i])))^2)/sum((as.numeric(unlist(train_low_list[i]))-mean(as.numeric(unlist(pre_val_loglow[i]))))^2))
}
mean(R2_lowlog)

lowlog_metric=c(mean(MAE_lowlog),mean(RMSE_lowlog),mean(R2_lowlog));lowlog_metric

###最终模型验证####
final_val=(0.1*predict(xgbTree.fit,short_train)+
  0.2*predict(xgbTree.fit.tune,short_train)+
  0.3*(2^predict(xgbTree.fit.final,short_train)))+0.2*as.numeric(unlist(pre_val_up))+0.2*as.numeric(unlist(pre_val_logup))
c(MAE(final_val,short_train$上行业务量GB),
RMSE(final_val,short_train$上行业务量GB),
R2(final_val,short_train$上行业务量GB))


final_val.low=(0.1*predict(xgbTree.fit.low,short_train)+
             0.2*predict(xgbTree.fit.tune.low,short_train)+
             0.3*(2^predict(xgbTree.fit.final.low,short_train)))+0.2*as.numeric(unlist(pre_val_low))+0.2*as.numeric(unlist(pre_val_loglow))
c(MAE(final_val.low,short_train$下行业务量GB),
RMSE(final_val.low,short_train$下行业务量GB),
R2(final_val.low,short_train$下行业务量GB))



final_val186=(0.1*predict(xgbTree.fit,train_186)+
             0.2*predict(xgbTree.fit.tune,train_186)+
             0.3*(2^predict(xgbTree.fit.final,train_186)))+
  0.4*(2^(forecast_result$yhat[1:(nrow(train_186))]))
c(MAE(final_val186,train_186$上行业务量GB),
  RMSE(final_val186,train_186$上行业务量GB),
  R2(final_val186,train_186$上行业务量GB))


final_val186.low=(0.1*predict(xgbTree.fit.low,train_186)+
                0.2*predict(xgbTree.fit.tune.low,train_186)+
                0.3*(2^predict(xgbTree.fit.final.low,train_186)))+
  0.4*(2^(forecast_result$yhat[1:(nrow(train_186))]))
c(MAE(final_val186.low,train_186$下行业务量GB),
  RMSE(final_val186.low,train_186$下行业务量GB),
  R2(final_val186.low,train_186$下行业务量GB))
#####短期验证
short_test=read.csv('附件2：短期验证选择的小区数据集.csv')
short_test_pre=read.csv('短期验证结果.csv')
head(short_test)
head(short_test_pre)
summary(short_test_pre)
plot(1:(169+169),c(as.numeric(unlist(short_train%>%filter(小区编号==186)%>%select(上行业务量GB)))[1000:1169],rep(NA,168)),
     type='l')
lines(169:(169+167),as.numeric(unlist(short_test_pre%>%filter(小区编号==186)%>%select(上行业务量GB))),col='2')

plot(1:(169+169),c(as.numeric(unlist(short_train%>%filter(小区编号==186)%>%select(下行业务量GB)))[1000:1169],rep(NA,168)),
     type='l')
lines(169:(169+167),as.numeric(unlist(short_test_pre%>%filter(小区编号==186)%>%select(下行业务量GB))),col='2')

plot(1:(169+169),c(as.numeric(unlist(short_train%>%filter(小区编号==420)%>%select(上行业务量GB)))[1000:1169],rep(NA,168)),
     type='l')
lines(169:(169+167),as.numeric(unlist(short_test_pre%>%filter(小区编号==420)%>%select(上行业务量GB))),col='2')

plot(1:(169+169),c(as.numeric(unlist(short_train%>%filter(小区编号==420)%>%select(下行业务量GB)))[1000:1169],rep(NA,168)),
     type='l')
lines(169:(169+167),as.numeric(unlist(short_test_pre%>%filter(小区编号==420)%>%select(下行业务量GB))),col='2')
write.csv(short_test_pre,'A485Test1.csv',row.names=F)

sum(ifelse(sub_test$日期==short_test_pre$日期 & sub_test$时间==short_test_pre$时间 &
             sub_test$小区编号==short_test_pre$小区编号,0,1))

#####长期验证####
long_test=read.csv('附件3：长期验证选择的小区数据集.csv')
long_test_pre=read.csv('final_long_test.csv',encoding = 'UTF-8')
head(long_test)
head(long_test_pre)
colnames(long_test_pre)=colnames(long_test)
dim(long_test)
dim(long_test_pre)
summary(long_test)
summary(long_test_pre)

sum(ifelse(long_test$日期==long_test_pre$日期 & long_test$时间==long_test_pre$时间 &
             long_test$小区编号==long_test_pre$小区编号,0,1))
long_test_pre$下行业务量GB=ifelse(long_test_pre$下行业务量GB<0,0,long_test_pre$下行业务量GB)
write.csv(long_test_pre,'A485Test2.csv',row.names=F)
table(long_test$小区编号)[which(table(long_test$小区编号)<25)]
nrow(long_test%>%group_by(小区编号)%>%summarise(n()))
nrow(long_test_pre%>%group_by(小区编号)%>%summarise(n()))

plot(density(long_test_pre$上行业务量GB))
plot(density(long_test_pre$下行业务量GB))