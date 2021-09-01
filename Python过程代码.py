import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from pylab import mpl
from sklearn.preprocessing import LabelEncoder
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
# matplotlib.rcParams['font.family'] = 'SimHei'
import warnings
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold,train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
warnings.filterwarnings('ignore')

train=pd.read_csv("data/train.csv",encoding="GBK")
long_test=pd.read_csv("data/long/long_test.csv",encoding="GBK")
short_test=pd.read_csv("data/short/short_test.csv",encoding="GBK")

# 提取所有短期小区编号
short_no=short_test['小区编号'].unique()
short_train=train[train['小区编号'].isin(short_no)]
short_train=short_train.drop_duplicates(keep='first').reset_index(drop=True)
short_train.to_pickle("data/short_train.pkl")
short_train.to_csv("data/short_train.csv",index=False)
del short_train


# 提取所有长期小区编号
long_no=long_test['小区编号'].unique()
long_train=train[train['小区编号'].isin(long_no)]
long_train=long_train.drop_duplicates(keep='first').reset_index(drop=True)

long_train.to_pickle("data/long_train.pkl")
long_train.to_csv("data/long_train.csv",index=False)
del long_train


short_train=pd.read_pickle('train/short_train.pkl')

long_train=pd.read_pickle('train/long_train.pkl')
short_test=pd.read_csv("short/short_test.csv",encoding="GBK")
long_test=pd.read_csv("long/long_test.csv",encoding="GBK")
long_test_dat_day=pd.read_csv("long/final_long_day.csv",encoding="GBK")
short_final=pd.read_csv("short/Short.csv",encoding="GBK")
long_test_dat_week=pd.read_csv("long/long_test_dat_week.csv",encoding="GBK")


long_test_dat_day.columns = ['month', 'day', '小区类别', 'day_up_mean', 'day_down_mean']
long_test_dat_week.columns = ['小区类别', 'month', 'day', 'weekday', 'week_up_mean', 'week_down_mean']
long_test_dat_day


def Process(df):
    df['日期']=df['日期'].apply(lambda x: x.replace('018-','2018/').replace('-','/'))
    df['year']=pd.to_datetime(df['日期']).dt.year
    df['month']=pd.to_datetime(df['日期']).dt.month
    df['day']=pd.to_datetime(df['日期']).dt.day
    df['hour']=pd.to_datetime(df['时间']).dt.hour
    df=df.sort_values(by=['小区编号','year','month','day','hour']).reset_index(drop=True)
    first_month=df.drop_duplicates(subset=['小区编号'],keep='first')[['小区编号','month','day','hour']]
    first_month.columns=['小区编号','first_month','first_day','first_hour']
    df=df.merge(first_month,on='小区编号')
    df['all_hours']=(df['month']-df['first_month'])*30*24+                (df['day']-df['first_day'])*24+df['hour']-df['first_hour']
    return df
short_train=Process(short_train)
long_train=Process(long_train)


group_up=long_train.groupby(['小区编号'])['上行业务量GB'].agg(
        Mean_up= 'mean',
        sd_up='std',
        Max_up='max',
        Min_up='min',
        ).reset_index()
group_down=long_train.groupby(['小区编号'])['下行业务量GB'].agg(
        Mean_down= 'mean',
        sd_down='std',
        Max_down='max',
        Min_down='min',
        ).reset_index().drop('小区编号',axis=1)  
group=pd.concat([group_up,group_down],axis=1)

group[1]=(group_up['Mean_up']-0.09617856)**2+(group_up['sd_up']-0.08429029)**2+        (group_up['Max_up']-0.8312091)**2+(group_up['Min_up']-0.0022922078)**2+        (group_down['Mean_down']-0.6707744)**2+(group_down['sd_down']-0.6153886)**2+        (group_down['Max_down']-6.378675)**2+(group_down['Min_down']-0.0020579534)**2


group[3]=(group_up['Mean_up']-0.06336309)**2+(group_up['sd_up']-0.05697710)**2+        (group_up['Max_up']-0.5427470)**2+(group_up['Min_up']-0.0014971490)**2+        (group_down['Mean_down']-0.4476131)**2+(group_down['sd_down']-0.3883523)**2+        (group_down['Max_down']-3.329850)**2+(group_down['Min_down']-0.0017484253)**2


group[2]=(group_up['Mean_up']-0.02002979)**2+(group_up['sd_up']-0.02352023)**2+       (group_up['Max_up']-0.2696601)**2+(group_up['Min_up']-0.0002678037)**2+       (group_down['Mean_down']-0.1338717)**2+(group_down['sd_down']-0.1385608)**2+       (group_down['Max_down']-1.205907)**2+(group_down['Min_down']-0.0005185034)**2

group['小区类别']=group[[1,2,3]].idxmin(axis=1)


long_train[['up_label','down_label']]=long_train.groupby(['小区编号','year','month','day'])[['上行业务量GB','下行业务量GB']].transform('sum')


Train_data=long_train.drop_duplicates(subset=['小区编号','month','day'],keep='first').reset_index(drop=True)

week={0:'星期一',1:'星期二',2:'星期三',3:'星期四',4:'星期五',5:'星期六',6:'星期日'}
Train_data['weekday']=pd.to_datetime(Train_data['日期']).dt.weekday.replace(week)

Train_data=Train_data.merge(group[['小区编号','小区类别']],on='小区编号')
#day均值聚合
day_mean=long_test_dat_day.groupby(['小区类别','day'])[['day_up_mean','day_down_mean']].mean()
Train_data=Train_data.merge(day_mean,on=['小区类别','day'],how='left')
#按照week均值
week_mean=long_test_dat_week.groupby(['小区类别','weekday'])[['week_up_mean','week_down_mean']].mean()
Train_data=Train_data.merge(week_mean,on=['小区类别','weekday'],how='left')


long_test=pd.read_csv("long/long_test.csv",encoding="GBK")
long_test['year']=pd.to_datetime(long_test['日期']).dt.year
long_test['month']=pd.to_datetime(long_test['日期']).dt.month
long_test['day']=pd.to_datetime(long_test['日期']).dt.day
week={0:'星期一',1:'星期二',2:'星期三',3:'星期四',4:'星期五',5:'星期六',6:'星期日'}
long_test['weekday']=pd.to_datetime(long_test['日期']).dt.weekday.replace(week)
long_test=long_test.merge(group[['小区编号','小区类别']],on='小区编号',how='outer')
long_test['小区类别'].fillna(3,inplace=True)

#day均值聚合
day_mean=long_test_dat_day.groupby(['小区类别','day'])[['day_up_mean','day_down_mean']].mean()
long_test=long_test.merge(day_mean,on=['小区类别','day'],how='left')

#按照week均值
week_mean=long_test_dat_week.groupby(['小区类别','weekday'])[['week_up_mean','week_down_mean']].mean()

long_test=long_test.merge(week_mean,on=['小区类别','weekday'],how='left')


le =LabelEncoder()
le.fit(Train_data['weekday'].values)
Train_data['weekday']=le.transform(Train_data['weekday'].values)
long_test['weekday']=le.transform(long_test['weekday'].values)

feats=['小区编号','month','weekday','day','day_up_mean','day_down_mean','week_up_mean','week_down_mean']


def model_train(train_data,test_data,target):

    X_train, X_val, Y_train, Y_val=train_test_split(train_data[feats].values,train_data[target].values,test_size=0.1, shuffle=True)
    model = lgb.LGBMRegressor(
                              num_leaves=31,
                              learning_rate=0.01,
                              n_estimators=10000,
                              metric ='mse',
                              random_state=1080)
    lgb_model = model.fit(X_train, 
                          Y_train,
                          eval_set=[(X_train, Y_train), (X_val, Y_val)],
                          verbose=100,
                          eval_metric='mse',
                          early_stopping_rounds=100)
    
    val_pred=lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration_)
#     Y_val=np.power(2,Y_val)
#     val_pred=np.power(2,val_pred)
    print("MSE:",mean_squared_error(Y_val, val_pred))
    pred_test = lgb_model.predict(test_data[feats].values, num_iteration=lgb_model.best_iteration_)
    print(pred_test)
    return pred_test,Y_val, val_pred

pred_all=[]
Y_val_all=[]
val_pred_all=[]
for i in range(1,4):
    print("Training ",i," start >>>>>>>>>>>>>>>>>>>>>>")
    tmp_test=long_test.loc[long_test['小区类别']==i,]
    pred,Y_val, val_pred=model_train(Train_data,tmp_test,'up_label')
    long_test.loc[long_test['小区类别']==i,'上行业务量GB']=pred
    pred_all.append(pred)
    Y_val_all.append(Y_val)
    val_pred_all.append(val_pred)

down_pred_all=[]
down_Y_val_all=[]
down_val_pred_all=[]
for i in range(1,4):
    print("Training ",i," start >>>>>>>>>>>>>>>>>>>>>>")
    tmp_test=long_test.loc[long_test['小区类别']==i,]
    pred,Y_val,val_pred=model_train(Train_data,tmp_test,'down_label')
    long_test.loc[long_test['小区类别']==i,'下行业务量GB']=pred
    down_pred_all.append(pred)
    down_Y_val_all.append(Y_val)
    down_val_pred_all.append(val_pred)

long_test.to_csv('long/tmp_res.csv',index=False,encoding='utf-8-sig')
res=pd.read_csv('long/tmp_res.csv')
long_test_dat_day[['up_draw','down_draw']]=long_test_dat_day.groupby(['小区类别','day'])[['day_up_mean','day_down_mean']].transform('mean')
day_df=long_test_dat_day[long_test_dat_day['month']<4]
res=pd.read_csv('long/长期验证结果.csv')
# res.loc[res['下行业务量GB']>10,'下行业务量GB']=10
res.loc[res['下行业务量GB']<0,'下行业务量GB']=0
res['day']=pd.to_datetime(res['日期']).dt.day
res=res.drop(['day_up_mean','day_down_mean'],axis=1)
res=res.merge(day_df[['day','小区类别','day_up_mean','day_down_mean']],on=['小区类别','day'],how='left')
res['上行业务量GB']=res['上行业务量GB']*0.6+res['day_up_mean']*0.4
res['下行业务量GB']=res['下行业务量GB']*0.6+res['day_down_mean']*0.4
res[['日期','小区编号','上行业务量GB','下行业务量GB']].to_csv('long/final_long_test.csv',index=False,encoding='utf-8-sig')

