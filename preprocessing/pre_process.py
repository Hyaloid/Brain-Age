import pandas as pd
import os
import chardet
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
def add_prefix_to_features(dataframe, prefix):
    # 获取所有特征列的名称
    columns = dataframe.columns.tolist()

    # 使用前缀来修改特征名称
    new_columns = [prefix + col if col not in ["index", "Unnamed: 0"] and col != 'subject_ID' else col for col in columns]

    # 为数据框设置新的列名称
    dataframe.columns = new_columns
    return dataframe

def merge_lr(datalist):
    df = pd.DataFrame()
    selected_lh = [file for file in datalist if os.path.basename(file).startswith('lh')]
    selected_rh = [file for file in datalist if os.path.basename(file).startswith('rh')]
    df_l = pd.read_csv(f'./Test_2/{selected_lh[0]}',index_col='subject_ID')
    df_l = df_l.dropna()
    df_r = pd.read_csv(f'./Test_2/{selected_rh[0]}',index_col='subject_ID')
    for file in selected_lh[1:]:
        df = pd.read_csv(f'./Test_2/{file}')
        df_l = pd.merge(df_l, df, on='subject_ID', how='left')
    for file in selected_rh[1:]:
        df = pd.read_csv(f'./Test_2/{file}')
        df_r = pd.merge(df_r, df, on='subject_ID', how='left')

    df = pd.merge(df_l, df_r, on='subject_ID', how='left')

    aseg = pd.read_csv('./Test/aseg - 389.csv')
    df = pd.merge(df, aseg, on='subject_ID', how='left')
    wmparc = pd.read_csv('./Test/wmparc - 389.csv')
    df = pd.merge(df, wmparc, on='subject_ID', how='left')
    # 读取文件内容并检测编码
    with open('./Test/subject_info - 389.csv', 'rb') as f:
        result = chardet.detect(f.read())

    # 获取检测到的编码
    encoding = result['encoding']

    # 使用检测到的编码读取文件
    label = pd.read_csv('./Test/subject_info - 389.csv',encoding=encoding)
    label.columns = ['subject_ID','MRI扫描仪类型','性别',"注：性别（1=Female, 2=Male)"]
    df = pd.merge(df, label, on='subject_ID', how='left')

    # 删除最后一列
    df = df.drop(df.columns[-1], axis=1)
    # 使用factorize()函数对列进行自动编码
    df['MRI扫描仪类型'], unique_values = pd.factorize(df['MRI扫描仪类型'])
    return df


import pandas as pd
import os


def re_create(file_path,datalist):
    lh_files = [file for file in datalist if os.path.basename(file).startswith('lh')]
    rh_files = [file for file in datalist if os.path.basename(file).startswith('rh')]

    if len(lh_files) != len(rh_files):
        raise ValueError("Number of lh files does not match number of rh files.")

    # 创建一个空的 DataFrame 用于存储结果
    result_df = pd.DataFrame()

    # 逐对处理 lh 和 rh 文件
    for lh_file, rh_file in zip(lh_files, rh_files):
        lh_data = pd.read_csv(file_path+'/'+lh_file,index_col=0)
        lh_data.dropna(inplace=True)# 假设是 CSV 文件
        rh_data = pd.read_csv(file_path+'/'+rh_file,index_col=0)

        if not lh_data.shape == rh_data.shape:
            raise ValueError("Shape of lh and rh dataframes must match.")

        # 计算差值并添加到结果 DataFrame
        diff_data = lh_data - rh_data

        # 获取文件名后面的文字（不包括扩展名部分）
        col_name_suffix = os.path.splitext(os.path.basename(lh_file))[0].split('lh.')[-1]

        # 重命名差值的列
        diff_data.columns = [f'lh_rh_{col_name_suffix[:-6]}_{col}' for col in diff_data.columns]

        result_df = pd.concat([result_df, diff_data], axis=1)  # 横向拼接

    return result_df




if __name__ == '__main__':
    # datalist = os.listdir('./Test')
    # selected_lh = [file for file in datalist if os.path.basename(file).startswith('lh')]
    # selected_rh = [file for file in datalist if os.path.basename(file).startswith('rh')]
    # for file in selected_lh:
    #      df = pd.read_csv(f'./Test/{file}')
    #      # 获取第一列的列名
    #      first_column_name = df.columns[0]
    #      # 为第一列分配新的名称
    #      new_column_name = 'subject_ID'
    #      df.rename(columns={first_column_name: new_column_name}, inplace=True)
    #      df = add_prefix_to_features(df,'lh_'+file[3:-10]+'_')
    #      df.to_csv(f'./Test_2/{file}',index=False)
    # for file in selected_rh:
    #      df = pd.read_csv(f'./Test/{file}')
    #      # 获取第一列的列名
    #      first_column_name = df.columns[0]
    #      # 为第一列分配新的名称
    #      new_column_name = 'subject_ID'
    #      df.rename(columns={first_column_name: new_column_name}, inplace=True)
    #      df = add_prefix_to_features(df,'rh_'+file[3:-10]+'_')
    #      df.to_csv(f'./Test_2/{file}',index=False)
    #
    # data = merge_lr(datalist)
    # # data.to_csv('./Train/data.csv',index=False,encoding='utf-8-sig')
    # data.to_csv('./Test_2/data_test.csv',index=False,encoding='utf-8-sig')
    # data = pd.read_csv('./Test/data_test.csv',encoding='utf-8-sig')

    # 检查空值
    # has_null_values = data.isnull().any().any()

    # 输出结果
    # print("DataFrame中是否有空值：", has_null_values)
    # data_numeric = data.apply(pd.to_numeric, errors='coerce')
    # 检查无限大和无限小的数
    # has_inf_values = np.isinf(data_numeric.values).any()

    # 输出结果
    # print("DataFrame中是否有无限大或无限小的数：", has_inf_values)
    # 去除空值、无限大和无限小的数
    # cleaned_data = data.dropna()
    # cleaned_data.to_csv('./Test/cleaned_test_data.csv',index=False,encoding='utf-8-sig')
    # 找出包含缺失值的行
    # rows_with_missing_values = data[data.isna().any(axis=1)]

    # 输出结果
    # print("包含缺失值的行：")
    # print(rows_with_missing_values)

    # df1 = pd.read_csv('./Train/lh.GausCurv - 1600.csv')
    # df2 = pd.read_csv('./Train/subject_info - 1600.csv')
    # list1 = df1['subject_ID'].tolist()
    # list2 = df2['subject_ID'].tolist()
    # set1 = set(list1)
    # set2 = set(list2)
    # # 找出list1和list2各自列表独有的部分（差集）
    # unique_in_list1 = set1.difference(set2)
    # unique_in_list2 = set2.difference(set1)
    #
    # # 输出结果
    # print("list1独有的部分:", unique_in_list1)
    # print("list2独有的部分:", unique_in_list2)
    # list1独有的部分: {'CNBD_00876', 'CNBD_01289', 'CNBD_00223', 'CNBD_00607'}
    # list2独有的部分: {'CNBD_00035', 'CNBD_00216', 'CNBD_02142', 'CNBD_00284'}
    # df1 = pd.read_csv('./Test/cleaned_test_data.csv')
    # data = df1['lh_GausCurv_caudalmiddlefrontal'].tolist()
    # # 绘制直方图和概率密度图
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.hist(data, bins=30, density=True, alpha=0.7)
    # plt.title('Histogram')
    #
    # plt.subplot(1, 2, 2)
    # plt.hist(data, bins=30, density=True, alpha=0.7)
    # plt.plot(np.linspace(-4, 4, 100), stats.norm.pdf(np.linspace(-4, 4, 100), loc=0, scale=1), 'r', linewidth=2)
    # plt.title('Histogram with PDF')
    # plt.show()
    #
    # 绘制Q-Q图
    # plt.figure()
    # stats.probplot(data, plot=plt)
    # plt.title('Q-Q Plot')
    # plt.show()

    # 进行Shapiro-Wilk检验
    # statistic, p_value = stats.shapiro(data)
    # print('Shapiro-Wilk Test - Statistic:', statistic, 'P-value:', p_value)
    # data = pd.read_csv('./Test/cleaned_test_data.csv',index_col='subject_ID')
    # # 选择需要标准化的列
    # columns_to_transform = data.columns[:-2]
    #
    # # 创建MinMaxScaler对象
    # scaler = MinMaxScaler()
    #
    # # 对DataFrame的选定列进行归一化
    # data[columns_to_transform] = scaler.fit_transform(data[columns_to_transform])
    #
    # data.to_csv('./Test/test_data.csv',encoding='utf-8-sig')
    # 调用 re_create 函数并传入 datalist
    # file_path = './Test'
    # datalist = os.listdir('./Test')
    # result = re_create(file_path,datalist)
    # result.to_csv('./Test_2/diff_test.csv',encoding='utf-8')
    # 结果 DataFrame
    diff_train = pd.read_csv('./Train_2/diff_train.csv')
    diff_test = pd.read_csv('./Test_2/diff_test.csv')
    data_train = pd.read_csv('./Train_2/data_train.csv')
    data_test = pd.read_csv('./Test_2/data_test.csv')
    data_train_z = pd.read_csv('./Train_2/data_train_z.csv')

    # 修改第一列的列名为 new_column_name
    new_column_name = 'subject_ID'
    diff_train.columns.values[0] = new_column_name
    diff_test.columns.values[0] = new_column_name

    data_train_zc = pd.concat([data_train,diff_train],axis = 1)
    data_train_zz = pd.merge(data_train_z, diff_train, on='subject_ID', how='left')

    data_test_zc = pd.concat([data_test,diff_test],axis = 1)

    data_train_zz.to_csv('./Train_2/data_train_zz.csv',encoding='utf-8')
    data_train_zc.to_csv('./Train_2/data_train_zc.csv', encoding='utf-8')
    data_test_zc.to_csv('./Test_2/data_test_zc.csv', encoding='utf-8')
