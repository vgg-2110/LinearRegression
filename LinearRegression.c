#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
//haha
#define MAX_ROW 2000
#define MAX_COL 10
#define MAX_LINE 200
#define MAX_TITLE_LEN 16
#define TRAIN_RATIO 0.8 // Tỷ lệ dữ liệu huấn luyện

typedef struct {
    double mean;
    double std;
} FeatureStats;

typedef struct {
    double y_mean;
    double y_std;
} TargetStats;

// Hàm chuẩn hóa dữ liệu (normalize)
void normalize_data(double **x, int num_samples, int num_features, FeatureStats *stats) {
    for (int j = 0; j < num_features; j++) {
        double sum = 0.0, sum_sq = 0.0;
        for (int i = 0; i < num_samples; i++) {
            sum += x[i][j];
            sum_sq += x[i][j] * x[i][j];
        }
        stats[j].mean = sum / num_samples;
        double variance = (sum_sq / num_samples) - (stats[j].mean * stats[j].mean);
        stats[j].std = sqrt(variance > 0 ? variance : 1.0);
        for (int i = 0; i < num_samples; i++) {
            x[i][j] = (x[i][j] - stats[j].mean) / stats[j].std;
        }
    }
}

// Hàm chuẩn hóa biến mục tiêu (y)
void normalize_target(double *y, int num_samples, TargetStats *stats) {
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < num_samples; i++) {
        sum += y[i];
        sum_sq += y[i] * y[i];
    }
    stats->y_mean = sum / num_samples;
    double variance = (sum_sq / num_samples) - (stats->y_mean * stats->y_mean);
    stats->y_std = sqrt(variance > 0 ? variance : 1.0);
    
    for (int i = 0; i < num_samples; i++) {
        y[i] = (y[i] - stats->y_mean) / stats->y_std;
    }
}

// Hàm kiểm tra phạm vi dữ liệu
void check_data_ranges(double **x, double *y, int num_samples, int num_features) {
    double min_y = DBL_MAX, max_y = -DBL_MAX;
    for (int i = 0; i < num_samples; i++) {
        if (y[i] < min_y) min_y = y[i];
        if (y[i] > max_y) max_y = y[i];
    }
    printf("Range of y: [%f, %f]\n", min_y, max_y);
    
    for (int j = 0; j < num_features; j++) {
        double min_x = DBL_MAX, max_x = -DBL_MAX;
        for (int i = 0; i < num_samples; i++) {
            if (x[i][j] < min_x) min_x = x[i][j];
            if (x[i][j] > max_x) max_x = x[i][j];
        }
        printf("Range of feature %d: [%f, %f]\n", j, min_x, max_x);
    }
}

// Hàm giải phóng bộ nhớ
void free_memory(double **x, double *y, int num_samples) {
    if (x) {
        for (int i = 0; i < num_samples; i++) {
            if (x[i]) // Kiểm tra xem x[i] có NULL không
                free(x[i]);
        }
        free(x);
    }
    if (y) {
        free(y);
    }
}

// Chuyển đổi trọng số về thang đo ban đầu
void denormalize_weights(double *w, double b, FeatureStats *stats, 
                         TargetStats *y_stats, int num_features, 
                         double *orig_w, double *orig_b) {
    *orig_b = b * y_stats->y_std + y_stats->y_mean;
    for (int j = 0; j < num_features; j++) {
        orig_w[j] = w[j] * y_stats->y_std / stats[j].std;
        *orig_b -= w[j] * y_stats->y_std * stats[j].mean / stats[j].std;
    }
}

int read_csv(const char *filename, int num_features, double ***x, double **y) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Khong the mo file\n");
        return 0;
    }

    *x = (double **)malloc(MAX_ROW * sizeof(double *));
    *y = (double *)malloc(MAX_ROW * sizeof(double));

    if (!*x || !*y) {
        printf("Loi cap phat bo nho\n");
        fclose(file);
        return 0;
    }

    char line[MAX_LINE];
    int count = 0;

    if (!fgets(line, MAX_LINE, file)) {
        printf("File rong hoac loi doc file\n");
        fclose(file);
        free_memory(*x, *y, count);
        return 0;
    }

    while (fgets(line, MAX_LINE, file) && count < MAX_ROW) {
        (*x)[count] = (double *)malloc(num_features * sizeof(double));
        if (!(*x)[count]) {
            printf("Loi cap phat bo nho\n");
            fclose(file);
            free_memory(*x, *y, count);
            return 0;
        }

        char *token = strtok(line, ",\n");
        int col = 0;

        while (token && col <= num_features) {
            if (col < num_features) {
                (*x)[count][col] = atof(token);
            } else if (col == num_features) {
                (*y)[count] = atof(token);
            }
            token = strtok(NULL, ",\n");
            col++;
        }

        if (col <= num_features) {
            printf("Dong %d khong du so cot\n", count + 2);
            free((*x)[count]);
            continue; // Bỏ qua dòng lỗi thay vì giảm count
        }        

        count++;
    }

    fclose(file);
    return count;
}

int count_features(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Khong the mo file %s\n", filename);
        return -1;
    }
    char line[MAX_LINE];
    if (!fgets(line, MAX_LINE, file)) {
        fclose(file);
        return -1;
    }
    fclose(file);

    int count = 1;
    for (int i = 0; line[i] != '\0'; i++) {
        if (line[i] == ',')
            count++;
    }
    if (count > 1)
        return count - 1;
    else
        return -1;
}

double predict(double *x, double *w, double b, int num_features) {
    double y_pred = b;
    for (int i = 0; i < num_features; i++)
        y_pred += w[i] * x[i];
    return y_pred;
}

double compute_cost(double **x, double *y, double *w, double b, int num_features, int num_samples) {
    if (num_samples <= 0) {
        printf("Loi: Khong co mau du lieu de tinh loss\n");
        return -1.0;
    }

    double loss = 0.0;
    // L2 regularization
    double lambda = 0.01;
    double reg_term = 0.0;
    
    for (int i = 0; i < num_samples; i++) {
        double y_pred = predict(x[i], w, b, num_features);
        double error = y[i] - y_pred;
        loss += error * error;
    }
    
    for (int j = 0; j < num_features; j++) {
        reg_term += w[j] * w[j];
    }

    if (isnan(loss) || isinf(loss)) {
        printf("Canh bao: Loss khong xac dinh hoac vo cung\n");
        return -1.0;
    }

    return loss / (2.0 * num_samples) + (lambda * reg_term) / (2.0 * num_samples);
}

int gradient_descent(double **x, double *y, double *w, double *b,
                    int num_features, int num_samples,
                    double learning_rate, int max_epochs) {
    if (num_samples <= 0) {
        printf("Loi: Khong co mau du lieu de huan luyen\n");
        return 0;
    }

    double alpha = learning_rate;
    int converged = 0;
    double prev_loss = DBL_MAX;
    double lambda = 0.01; // L2 regularization

    for (int epoch = 0; epoch < max_epochs && !converged; epoch++) {
        double *dw = (double *)calloc(num_features, sizeof(double));
        if (!dw) {
            printf("Loi cap phat bo nho\n");
            return 0;
        }
        double db = 0.0;

        for (int i = 0; i < num_samples; i++) {
            double y_pred = predict(x[i], w, *b, num_features);
            double error = y[i] - y_pred;

            for (int j = 0; j < num_features; j++) {
                dw[j] += error * x[i][j];
            }
            db += error;
        }

        for (int j = 0; j < num_features; j++) {
            double gradient = (dw[j] / num_samples) - (lambda * w[j] / num_samples);
            if (gradient > 1.0) gradient = 1.0;
            if (gradient < -1.0) gradient = -1.0;

            w[j] += alpha * gradient;
        }

        double bias_gradient = db / num_samples;
        if (bias_gradient > 1.0) bias_gradient = 1.0;
        if (bias_gradient < -1.0) bias_gradient = -1.0;

        *b += alpha * bias_gradient;

        free(dw);

        if (epoch % 100 == 0) {
            double current_loss = compute_cost(x, y, w, *b, num_features, num_samples);

            if (current_loss < 0) {
                printf("Epoch %d, Loss: Khong xac dinh (NaN/Inf)\n", epoch);
                return 0;
            }

            printf("Epoch %d, Loss: %f\n", epoch, current_loss);

            if (fabs(prev_loss - current_loss) < 1e-6) {
                printf("Da hoi tu sau %d epochs\n", epoch);
                converged = 1;
            }

            prev_loss = current_loss;

            if (epoch > 0 && epoch % 300 == 0) {
                alpha *= 0.5;
                printf("Giam learning rate xuong %f\n", alpha);
            }
        }
    }

    return 1;
}

// Hàm dự đoán cho dữ liệu mới
double predict_new_sample(double *features, double *w, double b, int num_features, 
    FeatureStats *stats, TargetStats *y_stats) {
    // Chuẩn hóa các đặc trưng đầu vào
    double *normalized_features = (double *)malloc(num_features * sizeof(double));
    if (!normalized_features) {
        printf("Loi cap phat bo nho cho du doan\n");
        return -1;
    }

    // Chuẩn hóa dữ liệu đầu vào
    for (int j = 0; j < num_features; j++) {
        normalized_features[j] = (features[j] - stats[j].mean) / stats[j].std;
    }

    // Dự đoán giá trị chuẩn hóa
    double normalized_prediction = b;
    for (int j = 0; j < num_features; j++) {
        normalized_prediction += w[j] * normalized_features[j];
    }

    // Chuyển đổi dự đoán về thang đo gốc
    double prediction = normalized_prediction * y_stats->y_std + y_stats->y_mean;

    free(normalized_features);
    return prediction;
}

int main() {
    const char *filename = "coffee_shop_revenue.csv";

    FILE *test_file = fopen(filename, "r");
    if (!test_file) {
        printf("Loi: Khong tim thay file %s\n", filename);
        return 1;
    }
    fclose(test_file);

    int num_features = count_features(filename);
    if (num_features <= 0) {
        printf("Loi: Khong the xac dinh so luong features\n");
        return 1;
    }

    double **x = NULL, *y = NULL;
    int num_samples = read_csv(filename, num_features, &x, &y);

    if (num_samples <= 0) {
        printf("Loi: Khong co mau du lieu de huan luyen\n");
        return 1;
    }

    printf("Doc thanh cong %d mau voi %d features\n", num_samples, num_features);
    
    // Kiểm tra phạm vi dữ liệu trước khi chuẩn hóa
    check_data_ranges(x, y, num_samples, num_features);

    // Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    int train_size = (int)(num_samples * TRAIN_RATIO);
    int test_size = num_samples - train_size;

    double **x_train = (double **)malloc(train_size * sizeof(double *));
    double *y_train = (double *)malloc(train_size * sizeof(double));
    double **x_test = (double **)malloc(test_size * sizeof(double *));
    double *y_test = (double *)malloc(test_size * sizeof(double));

    for (int i = 0; i < train_size; i++) {
        x_train[i] = x[i];
        y_train[i] = y[i];
    }
    for (int i = 0; i < test_size; i++) {
        x_test[i] = x[train_size + i];
        y_test[i] = y[train_size + i];
    }

    // Chuẩn hóa dữ liệu huấn luyện
    FeatureStats *stats = (FeatureStats *)malloc(num_features * sizeof(FeatureStats));
    if (!stats) {
        printf("Loi cap phat bo nho cho feature stats\n");
        free_memory(x, y, num_samples);
        return 1;
    }

    // Chuẩn hóa biến mục tiêu y
    TargetStats y_stats;
    normalize_target(y_train, train_size, &y_stats);
    printf("Da chuan hoa bien muc tieu y\n");

    normalize_data(x_train, train_size, num_features, stats);
    printf("Da chuan hoa du lieu huan luyen\n");

    // Chuẩn hóa dữ liệu kiểm tra sử dụng thống kê từ dữ liệu huấn luyện
    for (int i = 0; i < test_size; i++) {
        for (int j = 0; j < num_features; j++) {
            x_test[i][j] = (x_test[i][j] - stats[j].mean) / stats[j].std;
        }
        y_test[i] = (y_test[i] - y_stats.y_mean) / y_stats.y_std;
    }

    double *w = (double *)calloc(num_features, sizeof(double));
    if (!w) {
        printf("Loi cap phat bo nho\n");
        free_memory(x, y, num_samples);
        free(stats);
        return 1;
    }

    double b = 0.0;

    double learning_rate = 0.1;
    int max_epochs = 100000;

    double initial_loss = compute_cost(x_train, y_train, w, b, num_features, train_size);
    printf("Initial Loss: %f\n", initial_loss);

    int success = gradient_descent(x_train, y_train, w, &b, num_features, train_size, learning_rate, max_epochs);

    if (success) {
        double *orig_w = (double *)malloc(num_features * sizeof(double));
        double orig_b = 0.0;

        if (orig_w) {
            denormalize_weights(w, b, stats, &y_stats, num_features, orig_w, &orig_b);

            printf("\nFinal parameters (original scale):\n");
            for (int j = 0; j < num_features; j++) {
                printf("w[%d] = %f\n", j, orig_w[j]);
            }
            printf("b = %f\n", orig_b);

            free(orig_w);
        } else {
            printf("\nFinal parameters (normalized scale):\n");
            for (int j = 0; j < num_features; j++) {
                printf("w[%d] = %f\n", j, w[j]);
            }
            printf("b = %f\n", b);
        }

        // Đánh giá mô hình trên tập kiểm tra
        double test_loss = compute_cost(x_test, y_test, w, b, num_features, test_size);
        printf("Test Loss: %f\n", test_loss);
    } else {
        printf("Huan luyen that bai do van de khong on dinh so\n");
    }

    // Giải phóng bộ nhớ
    free(w);
    free_memory(x, y, num_samples);
    free(stats);

    if (success) {
        char answer = 'y';
        double *new_features = (double *)malloc(num_features * sizeof(double));
        if (!new_features) {
            printf("Loi cap phat bo nho cho du doan\n");
            return 1;
        }
        
        while (answer == 'y' || answer == 'Y') {
            printf("\n=== DU DOAN GIA TRI MOI ===\n");
            printf("Nhap %d dac trung (features):\n", num_features);
            
            for (int j = 0; j < num_features; j++) {
                printf("Feature %d: ", j);
                if (scanf("%lf", &new_features[j]) != 1) {
                    printf("Loi: Nhap khong hop le\n");
                    // Xóa bộ đệm nhập
                    int c;
                    while ((c = getchar()) != '\n' && c != EOF);
                    continue;
                }
            }
            
            double prediction = predict_new_sample(new_features, w, b, num_features, stats, &y_stats);
            printf("Du doan: %.2f\n", prediction);
            
            printf("\nBan co muon du doan gia tri khac? (y/n): ");
            // Xóa bộ đệm nhập
            int c;
            while ((c = getchar()) != '\n' && c != EOF);
            answer = getchar();
        }
        
        free(new_features);
    }

    return 0;
}