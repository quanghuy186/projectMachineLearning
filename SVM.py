import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tkinter import *
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#huấn luyện mô hình SVM
data = pd.read_csv('Data.csv') 

# Xác định các đặc trưng và nhãn
X = data.drop('Result', axis=1)
y = data['Result']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Tiêu chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#gt tb gan bang 0 phuong sai gan bang 1

# Tạo mô hình SVM
svm_model = SVC(kernel='linear')

# Huấn luyện mô hình trên tập huấn luyện
svm_model.fit(X_train, y_train)

# Tạo cửa sổ
root = Tk()
root.title("Dự đoán béo phì")
root.geometry("900x800")  # Cài đặt kích thước cửa sổ

# Đặt màu nền cho cửa sổ
root.configure(bg="pink")

# Tạo các biến đầu vào
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

# Tạo hàm dự đoán
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
    input_data = scaler.transform([[gender, age, height, weight, family_history, favc, ncp, smoke, ch2o, faf, tue]])

    # Dự đoán kết quả
    result = svm_model.predict(input_data)

    # Hiển thị kết quả
    if result[0] == 1:
        result_label.config(text="Bạn bị béo phì rồi!!!", fg="red")
    else:
        result_label.config(text="Không béo phì đâu ăn nhiều vào", fg="green")

    # Đánh giá mô hình trên tập huấn luyện
y_train_pred = svm_model.predict(X_train)
accuracy_train = accuracy_score(y_train, y_train_pred)
confusion_matrix_train = confusion_matrix(y_train, y_train_pred)
classification_report_train = classification_report(y_train, y_train_pred)

# Đánh giá mô hình trên tập kiểm tra
y_test_pred = svm_model.predict(X_test)
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
gender_label = Label(root, text="FAVC: Sử dụng thức ăn nhanh (1.có hoặc 0.không). NCP:Số lượng bữa ăn trong ngày. SMOKE: Hút thuốc(1.có hoặc 0.không).CH2O: Lượng nước uống hàng ngày(1,2,3).FAF:Số ngày tiêu thụ thức ăn nhanh trong tuần(1-4).TUE:Số giờ xem TV trong ngày(0-2).)", height=1, bg="lightblue")
gender_label.pack()

gender_label = Label(root, text="Result: Kết quả kiểm tra béo phì (0: không béo phì, 1: béo phì).", height=1, bg="lightblue")
gender_label.pack()


gender_label = Label(root, text="Giới tính (0: Nữ, 1: Nam): ",width=19, height=1, bg="lightblue")
gender_label.pack(pady=10)
gender_entry = Entry(root, textvariable=gender_var, width=22)
gender_entry.pack()

gender_label = Label(root, text="Tuổi ():",width=19, height=1, bg="lightblue")
gender_label.pack(pady=10)
gender_entry = Entry(root, textvariable=age_var, width=22)
gender_entry.pack()

gender_label = Label(root, text="Chiều cao ():",width=19, height=1, bg="lightblue")
gender_label.pack(pady=10)
gender_entry = Entry(root, textvariable=height_var, width=22)
gender_entry.pack()

gender_label = Label(root, text="Cân nặng :",width=19, height=1, bg="lightblue")
gender_label.pack(pady=10)
gender_entry = Entry(root, textvariable=weight_var, width=22)
gender_entry.pack()

gender_label = Label(root, text="family_his:",width=19, height=1, bg="lightblue")
gender_label.pack(pady=10)
gender_entry = Entry(root, textvariable=family_history_var, width=22)
gender_entry.pack()

gender_label = Label(root, text="FAVC:",width=19, height=1, bg="lightblue")
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

gender_label = Label(root, text="FAF:",width=19, height=1, bg="lightblue")
gender_label.pack(pady=10)
gender_entry = Entry(root, textvariable=faf_var, width=22)
gender_entry.pack()

gender_label = Label(root, text="TUE(Số giờ xem TV):",width=19, height=1, bg="lightblue")
gender_label.pack(pady=10)
gender_entry = Entry(root, textvariable=tue_var, width=22)
gender_entry.pack()

# Tạo nút dự đoán
predict_button = Button(root, text="Dự đoán",command=predict_obesity, bg="green", fg="white", padx=20, pady=10)
predict_button.pack(pady=10)

# Kết quả dự đoán
result_label = Label(root, text="", width=40, height=4)
result_label.pack(pady=10)

root.mainloop()
