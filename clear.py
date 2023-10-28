import pandas as pd

# Đọc dữ liệu từ tập tin CSV vào một DataFrame
data = pd.read_csv('DataSetObesity.csv')

# Xem thông tin về dữ liệu
print(data.info())

# Xử lý dữ liệu thiếu
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Height'].fillna(data['Height'].mean(), inplace=True)
data['Weight'].fillna(data['Weight'].mean(), inplace=True)

# Làm tròn giá trị sau khi thêm vào các ô rỗng
data['Height'] = data['Height'].round(2)
data['Weight'] = data['Weight'].round()

# Chuyển đổi kiểu dữ liệu
data['Age'] = data['Age'].astype(int)
data['Height'] = data['Height'].astype(float)
data['Weight'] = data['Weight'].astype(float)

# Xử lý dữ liệu không hợp lệ
data = data[(data['Height'] > 0) & (data['Weight'] > 0)]

# Chuyển đổi kiểu dữ liệu của các cột Gender, family_history_with_0 và SMOKE
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data['family_history_with_0'] = data['family_history_with_0'].map({'no': 0, 'yes': 1})
data['SMOKE'] = data['SMOKE'].map({'no': 0, 'yes': 1})
data['FAVC'] = data['FAVC'].map({'no': 0, 'yes': 1})

# Lưu dữ liệu đã tiền xử lý vào một tập tin mới
data.to_csv('DataSetObesity_processed.csv', index=False)

dataNew = pd.read_csv("DataSetObesity_processed.csv")

# Lấy thông tin và nhãn của bộ dữ liệu
x = dataNew.iloc[:,:-1]
print(x)
y = dataNew.iloc[:,-1]
print(y)
