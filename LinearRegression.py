import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('coffee_shop_revenue.csv')
df.head()

# Hàm tính toán dự đoán
def predict(X, w):
    return np.dot(X, w)

# Hàm tính toán loss (Mean Squared Error)
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Hàm tính gradient descent
def gradient_descent(X, y, w, learning_rate, epochs):
    m = len(y)
    loss_history = []
    
    for epoch in range(epochs):
        y_pred = predict(X, w)
        loss = compute_loss(y, y_pred)
        loss_history.append(loss)
        
        gradient = (2 / m) * np.dot(X.T, (y_pred - y))
        w -= learning_rate * gradient
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return w, loss_history

# Lấy dữ liệu đầu vào (X) và biến mục tiêu (y)
X = df.drop(columns=["Daily_Revenue"]).values
y = df["Daily_Revenue"].values.reshape(-1, 1)

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Thêm cột bias vào X_train và X_test
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Khởi tạo trọng số
np.random.seed(42)
w = np.random.randn(X_train.shape[1], 1)

# Siêu tham số
learning_rate = 0.01
epochs = 1000

# Huấn luyện mô hình
w, loss_history = gradient_descent(X_train, y_train, w, learning_rate, epochs)

# Dự đoán trên tập test
y_test_pred = predict(X_test, w)
test_loss = compute_loss(y_test, y_test_pred)

# In kết quả
print("Final Training Loss:", loss_history[-1])
print("Test Loss:", test_loss)
print("Trọng số cuối cùng:", w)

# Vẽ biểu đồ loss
plt.plot(range(epochs), loss_history)
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Loss giảm dần theo số epoch")
plt.show()