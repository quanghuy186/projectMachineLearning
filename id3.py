import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tkinter import *

# Huấn luyện mô hình Cây quyết định
data = pd.read_csv('Data.csv') 

# Xác định các đặc trưng và nhãn
X = data.drop('Result', axis=1)
y = data['Result']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Tạo mô hình Cây quyết định
decision_tree_model = DecisionTreeClassifier(max_depth=4, min_samples_split=4, min_samples_leaf=3)

# Huấn luyện mô hình trên tập huấn luyện
decision_tree_model.fit(X_train, y_train)

# Tạo cửa sổ
root = Tk()
root.title("Dự đoán béo phì")
root.geometry("900x800")  # Cài đặt kích thước cửa sổ
root.configure(bg="pink")  # Đặt màu nền cho cửa sổ

# Tạo biến đầu vào
gender_var = IntVar()
age_var = DoubleVar()
height_var = DoubleVar()
weight_var = DoubleVar()
family_history_var = IntVar()
favc_var = IntVar()
ncp_var = IntVar()
smoke_var = IntVar()
ch2o_var = DoubleVar()
faf_var = DoubleVar()
tue_var = IntVar()

# Hàm dự đoán
def predict_obesity():
    # Lấy giá trị từ các biến đầu vào
    gender = gender_var.get()
    age = age_var.get()
    height = height_var.get()
    weight = weight_var.get()
    family_history = family_history_var.get()
    favc = favc_var.get()
    ncp = ncp_var.get()
    smoke = smoke_var.get()
    ch2o = ch2o_var.get()
    faf = faf_var.get()
    tue = tue_var.get()

    # Chuẩn hóa dữ liệu đầu vào
    input_data = [[gender, age, height, weight, family_history, favc, ncp, smoke, ch2o, faf, tue]]

    # Dự đoán kết quả
    result = decision_tree_model.predict(input_data)

    if result[0] == 1:
        result_label.config(text="Bạn bị béo phì rồi lewlew", fg="red")
    else:
        result_label.config(text="Không béo phì đâu ăn nhiều vào", fg="green")
    # Đánh giá mô hình trên tập huấn luyện
y_train_pred = decision_tree_model.predict(X_train)
accuracy_train = accuracy_score(y_train, y_train_pred)
confusion_matrix_train = confusion_matrix(y_train, y_train_pred)
classification_report_train = classification_report(y_train, y_train_pred)

# Đánh giá mô hình trên tập kiểm tra
y_test_pred = decision_tree_model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_test_pred)
confusion_matrix_test = confusion_matrix(y_test, y_test_pred)
classification_report_test = classification_report(y_test, y_test_pred)

# In kết quả
print("Độ chính xác trên tập huấn luyện:", accuracy_train)
print("Ma trận nhầm lẫn trên tập huấn luyện:")
print(confusion_matrix_train)
print("Báo cáo phân loại trên tập huấn luyện:")
print(classification_report_train)

print("Độ chính xác trên tập kiểm tra:", accuracy_test)
print("Ma trận nhầm lẫn trên tập kiểm tra:")
print(confusion_matrix_test)
print("Báo cáo phân loại trên tập kiểm tra:")
print(classification_report_test)
# Tạo giao diện đầu vào
gender_label = Label(root, text="Giới tính (0: Nữ, 1: Nam): ",width=19, height=1, bg="lightblue")
gender_label.pack(pady=10)
gender_entry = Entry(root, textvariable=gender_var, width=22)
gender_entry.pack()

gender_label = Label(root, text="Tuổi (20-50):",width=19, height=1, bg="lightblue")
gender_label.pack(pady=10)
gender_entry = Entry(root, textvariable=age_var, width=22)
gender_entry.pack()

gender_label = Label(root, text="Chiều cao (1m50, 1m80):",width=19, height=1, bg="lightblue")
gender_label.pack(pady=10)
gender_entry = Entry(root, textvariable=height_var, width=22)
gender_entry.pack()

gender_label = Label(root, text="Cân nặng :",width=19, height=1, bg="lightblue")
gender_label.pack(pady=10)
gender_entry = Entry(root, textvariable=weight_var, width=22)
gender_entry.pack()

gender_label = Label(root, text="family_his (tiền xử):",width=19, height=1, bg="lightblue")
gender_label.pack(pady=10)
gender_entry = Entry(root, textvariable=family_history_var, width=22)
gender_entry.pack()

gender_label = Label(root, text="FAVC (Sử dụng thức ăn nhanh):",width=19, height=1, bg="lightblue")
gender_label.pack(pady=10)
gender_entry = Entry(root, textvariable=favc_var, width=22)
gender_entry.pack()

gender_label = Label(root, text="NCP (Số bữa ăn .):",width=19, height=1, bg="lightblue")
gender_label.pack(pady=10)
gender_entry = Entry(root, textvariable=ncp_var, width=22)
gender_entry.pack()

gender_label = Label(root, text="SMOKE (Hút thuốc):",width=19, height=1, bg="lightblue")
gender_label.pack(pady=10)
gender_entry = Entry(root, textvariable=smoke_var, width=22)
gender_entry.pack()

gender_label = Label(root, text="CH20 (Uống nước):",width=19, height=1, bg="lightblue")
gender_label.pack(pady=10)
gender_entry = Entry(root, textvariable=ch2o_var, width=22)
gender_entry.pack()

gender_label = Label(root, text="FAF (Ngày tiêu thụ thức ăn nhanh):",width=19, height=1, bg="lightblue")
gender_label.pack(pady=10)
gender_entry = Entry(root, textvariable=faf_var, width=22)
gender_entry.pack()

gender_label = Label(root, text="TUE(Số giờ xem TV):",width=19, height=1, bg="lightblue")
gender_label.pack(pady=10)
gender_entry = Entry(root, textvariable=tue_var, width=22)
gender_entry.pack()
# (Tiếp theo là tạo các label và entry tương tự như trong mã của bạn)
# ...

# Tạo nút dự đoán và hiển thị kết quả
predict_button = Button(root, text="Dự đoán", command=predict_obesity, bg="green", fg="white", padx=20, pady=10)
predict_button.pack(pady=10)

result_label = Label(root, text="", width=40, height=4)
result_label.pack(pady=10)

root.mainloop()
