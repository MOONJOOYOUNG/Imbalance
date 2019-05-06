import cov_function

#main
#x_train_high = cov_function.x_data_To_float(r'D:\exper_data\test5\tsne_feature_x1.csv')
#y_train_high = cov_function.y_data_To_list(r'D:\exper_data\test5\tsne_feature_y1.csv')
#x_test_high = cov_function.x_data_To_float(r'D:\exper_data\test5\tsne_feature_x2.csv')
#y_test_high= cov_function.y_data_To_list(r'D:\exper_data\test5\tsne_feature_y2.csv')

#x_train_high = cov_function.x_data_To_float(r'D:\exper_data\test2_1\cifar_feature_x1.csv')
#y_train_high = cov_function.y_data_To_list(r'D:\exper_data\test2_1\cifar_feature_y1.csv')
#x_test_high = cov_function.x_data_To_float(r'D:\exper_data\test2_1\cifar_feature_x2.csv')
#y_test_high= cov_function.y_data_To_list(r'D:\exper_data\test2_1\cifar_feature_y2.csv')
#df_acc = pd.read_csv('./{0}_cali_corr.csv'.format(name))
x_train_high = cov_function.x_data_To_float('./train_feature_x.csv')
y_train_high = cov_function.y_data_To_list('./train_feature_y.csv')

x_test_high = cov_function.x_data_To_float('./test_feature_x.csv')
y_test_high = cov_function.y_data_To_list('./test_feature_y.csv')

print(len(x_train_high), len(y_train_high),len(x_test_high), len(y_test_high))
# train & test origin_feature_class
train_x_high = cov_function.class_feature(x_train_high, y_train_high)
test_x_high = cov_function.class_feature(x_test_high, y_test_high)

print("train_high")
print(len(train_x_high))
for i in train_x_high:
    determinant = cov_function.high_dim_cov(i)

print('-------------------------')
print("test_high")
for i in test_x_high:
    cov_function.high_dim_cov(i)

# train & test TSNE_feature_extract
x_train_low, y_train_low = cov_function.tsne_feature(x_train_high, y_train_high)
x_test_low, y_test_low = cov_function.tsne_feature(x_test_high, y_test_high)

# train & test TSNE_feature_class
train_x_low = cov_function.class_feature_low(x_train_low, y_train_low)
test_x_low = cov_function.class_feature_low(x_test_low, y_test_low)

# =----------------  TSNE --------------
#
# print('-------------------------')
# print("train_low")
# for i in train_x_low:
#     cov_function.low_dim_cov(i)
#
# print('-------------------------')
# print("test_low")
# for i in test_x_low:
#     cov_function.low_dim_cov(i)
