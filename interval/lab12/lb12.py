import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from intvalpy import IntLinIncR2, Interval, Tol, precision
from int_f import IntLinIncR2

precision.extendedPrecisionQ = True

BASE_DIR = 'C:/Users/valer/intervals'
DATA_SUBDIR = os.path.join(BASE_DIR, 'lb12', 'bin')
DATA_FILE = '04_10_2024_070_068'
FULL_PATH = os.path.join(DATA_SUBDIR, DATA_FILE)

X_VALUES = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
POINTS_PER_X = 100

def import_sensor_data(directory, side):
    sensors = [[] for _ in range(8)]
    for s_idx in range(8):
        for t_idx in range(1024):
            trial_points = [(X_VALUES[t // POINTS_PER_X], 0) for t in range(POINTS_PER_X * len(X_VALUES))]
            sensors[s_idx].append(trial_points)
    
    for x_offset, x_val in enumerate(X_VALUES):
        file_name = f"{x_val}lvl_side_{side}_fast_data.json"
        file_path = os.path.join(directory, file_name)
        
        if not os.path.isfile(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            continue
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for s_idx in range(8):
            for t_idx in range(1024):
                try:
                    y_vals = data["sensors"][s_idx][t_idx]
                except (IndexError, KeyError, TypeError):
                    continue
                
                for p_idx, y_val in enumerate(y_vals):
                    pos = x_offset * POINTS_PER_X + p_idx
                    if pos < len(sensors[s_idx][t_idx]):
                        sensors[s_idx][t_idx][pos] = (x_val, y_val)
                    else:
                        print(f"Warning: Position {pos} out of range for sensor {s_idx}, trial {t_idx}.")
    
    return sensors

def regression_method_1(points):
    x, y = zip(*points)
    weights = [1 / 16384] * len(y)
    
    X_mat = Interval([[[x_el, x_el], [1, 1]] for x_el in x])
    Y_vec = Interval([[y_el, weights[i]] for i, y_el in enumerate(y)], midRadQ=True)
    
    b_vec, tol_val, _, _, _ = Tol.maximize(X_mat, Y_vec)
    updated = 0
    if tol_val < 0:
        for i in range(len(Y_vec)):
            X_small = Interval([[[x[i], x[i]], [1, 1]]])
            Y_small = Interval([[y[i], weights[i]]], midRadQ=True)
            val = Tol.value(X_small, Y_small, b_vec)
            if val < 0:
                weights[i] = abs(y[i] - (x[i] * b_vec[0] + b_vec[1])) + 1e-8
                updated += 1
        
        Y_vec = Interval([[y_el, weights[i]] for i, y_el in enumerate(y)], midRadQ=True)
        b_vec, tol_val, _, _, _ = Tol.maximize(X_mat, Y_vec)
    
    return b_vec, weights, updated

def regression_method_2(points):
    x, y = zip(*points)
    eps = 1 / 16384
    x_new = X_VALUES.copy()
    
    y_ex_up = [-float('inf')] * 11
    y_ex_down = [float('inf')] * 11
    y_in_up = [-float('inf')] * 11
    y_in_down = [float('inf')] * 11
    
    for i in range(len(x_new)):
        y_list = list(y[i * POINTS_PER_X : (i + 1) * POINTS_PER_X])
        y_list.sort()
        if len(y_list) < 75:
            continue
        y_in_down[i] = y_list[25] - eps
        y_in_up[i] = y_list[75] + eps
        y_ex_up[i] = min(y_list[75] + 1.5 * (y_list[75] - y_list[25]), y_list[-1])
        y_ex_down[i] = max(y_list[25] - 1.5 * (y_list[75] - y_list[25]), y_list[0])
    
    X_mat = []
    Y_vec = []
    for i in range(len(x_new)):
        x_el = x_new[i]
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_ex_down[i], y_ex_up[i]])
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_ex_down[i], y_in_up[i]])
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_in_down[i], y_ex_up[i]])
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_in_down[i], y_in_up[i]])
    
    X_interval = Interval(X_mat)
    Y_interval = Interval(Y_vec)
    b_vec, tol_val, _, _, _ = Tol.maximize(X_interval, Y_interval)
    to_remove = []
    if tol_val < 0:
        for i in range(len(Y_vec)):
            X_small = Interval([X_mat[i]])
            Y_small = Interval([Y_vec[i]])
            val = Tol.value(X_small, Y_small, b_vec)
            if val < 0:
                to_remove.append(i)
    
        for i in sorted(to_remove, reverse=True):
            del X_mat[i]
            del Y_vec[i]
    
        X_interval = Interval(X_mat)
        Y_interval = Interval(Y_vec)
        b_vec, tol_val, _, _, _ = Tol.maximize(X_interval, Y_interval)
    
    vertices1 = IntLinIncR2(X_interval, Y_interval)
    vertices2 = IntLinIncR2(X_interval, Y_interval, consistency='tol')
    
    plt.xlabel("b0")
    plt.ylabel("b1")
    
    b_uni_vertices = []
    for v in vertices1:
        if len(v) > 0:
            xx, yy = v[:, 0], v[:, 1]
            b_uni_vertices += [(xx[i], yy[i]) for i in range(len(xx))]
            plt.fill(xx, yy, linestyle='-', linewidth=1, color='yellow', alpha=0.5, label="Uni")
            plt.scatter(xx, yy, s=0, color='black', alpha=1)
    
    b_tol_vertices = []
    for v in vertices2:
        if len(v) > 0:
            xx, yy = v[:, 0], v[:, 1]
            b_tol_vertices += [(xx[i], yy[i]) for i in range(len(xx))]
            plt.fill(xx, yy, linestyle='-', linewidth=1, color='blue', alpha=0.3, label="Tol")
            plt.scatter(xx, yy, s=10, color='black', alpha=1)
    
    plt.scatter([b_vec[0]], [b_vec[1]], s=10, color='red', alpha=1, label="argmax Tol")
    plt.legend(loc='upper right')
    plt.grid(True)
    return b_vec, (y_in_down, y_in_up), (y_ex_down, y_ex_up), to_remove, b_uni_vertices, b_tol_vertices

def create_visualizations(data, coord_x, coord_y):
    b_vec, rads, to_remove = regression_method_1(data)
    x, y = zip(*data)
    plt.figure()
    plt.title("Y(x) method 1 for " + str((coord_x, coord_y)))
    plt.scatter(x, y, label="medians")
    
    plt.plot([-0.5, 0.5], [b_vec[1] + b_vec[0] * -0.5, b_vec[1] + b_vec[0] * 0.5], label="Argmax Tol", color='red')
    plt.legend(loc='upper right')
    plt.grid(True)
    print((coord_x, coord_y), 1, b_vec[0], b_vec[1], to_remove)
    
    plt.figure()
    plt.title("Uni and Tol method 2 for " + str((coord_x, coord_y)))
    b_vec2, y_in, y_ex, to_remove, b_uni_vertices, b_tol_vertices = regression_method_2(data)
    print((coord_x, coord_y), 2, b_vec2[0], b_vec2[1], len(to_remove))
    x2 = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    plt.figure()
    plt.title("Y(x) method 2 for " + str((coord_x, coord_y)))
    for i in range(len(x2)):
        plt.plot([x2[i], x2[i]], [y_ex[0][i], y_ex[1][i]], color="gray", zorder=1)
        plt.plot([x2[i], x2[i]], [y_in[0][i], y_in[1][i]], color="blue", zorder=2)
    
    plt.plot([-0.5, 0.5],
             [b_vec2[1] + b_vec2[0] * -0.5, b_vec2[1] + b_vec2[0] * 0.5],
             label="Argmax Tol",
             color="red",
             zorder=1000)
    
    x2_extended = [-3] + x2 + [3]
    for i in range(len(x2_extended) - 1):
        x0 = x2_extended[i]
        x1 = x2_extended[i + 1]
        mid_x = (x0 + x1) / 2
        
        max_idx = 0
        min_idx = 0
        max_val = b_uni_vertices[0][1] + b_uni_vertices[0][0] * mid_x
        min_val = b_uni_vertices[0][1] + b_uni_vertices[0][0] * mid_x
        for j in range(len(b_uni_vertices)):
            val = b_uni_vertices[j][1] + b_uni_vertices[j][0] * mid_x
            if max_val < val:
                max_idx = j
                max_val = val
            if min_val > val:
                min_idx = j
                min_val = val
        
        y0_low = b_uni_vertices[min_idx][1] + b_uni_vertices[min_idx][0] * x0
        y1_low = b_uni_vertices[min_idx][1] + b_uni_vertices[min_idx][0] * x1
        y0_hi = b_uni_vertices[max_idx][1] + b_uni_vertices[max_idx][0] * x0
        y1_hi = b_uni_vertices[max_idx][1] + b_uni_vertices[max_idx][0] * x1
        plt.fill([x0, x1, x1, x0], [y0_low, y1_low, y1_hi, y0_hi], facecolor="green", linewidth=0)
        
       
        max_idx = 0
        min_idx = 0
        max_val = b_tol_vertices[0][1] + b_tol_vertices[0][0] * mid_x
        min_val = b_tol_vertices[0][1] + b_tol_vertices[0][0] * mid_x
        for j in range(len(b_tol_vertices)):
            val = b_tol_vertices[j][1] + b_tol_vertices[j][0] * mid_x
            if max_val < val:
                max_idx = j
                max_val = val
            if min_val > val:
                min_idx = j
                min_val = val
        
        y0_low = b_tol_vertices[min_idx][1] + b_tol_vertices[min_idx][0] * x0
        y1_low = b_tol_vertices[min_idx][1] + b_tol_vertices[min_idx][0] * x1
        y0_hi = b_tol_vertices[max_idx][1] + b_tol_vertices[max_idx][0] * x0
        y1_hi = b_tol_vertices[max_idx][1] + b_tol_vertices[max_idx][0] * x1
        plt.fill([x0, x1, x1, x0], [y0_low, y1_low, y1_hi, y0_hi], facecolor="lightblue", linewidth=0)
    
    plt.xlim((-0.6, 0.6))
    plt.ylim((-0.6, 0.6))
    
    plt.legend(loc='upper right')
    plt.grid(True)

if __name__ == "__main__":
    side_a_data = import_sensor_data(FULL_PATH, "a")
    create_visualizations(side_a_data[2][300], 2, 300)
    create_visualizations(side_a_data[3][300], 3, 300)
    plt.show()