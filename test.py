import math

# Define the function to plot
def f(x):
    return x**2

x_min, x_max = -12, 12
y_min, y_max = 0, 25
scale_x = 2
scale_y = 1

for y in range(y_max, y_min - 1, -scale_y):
    line = f"{y:2d} |"
    for x in range(x_min * scale_x, (x_max + 1) * scale_x):
        real_x = x / scale_x
        if round(f(real_x)) == y:
            line += "*"
        else:
            line += " "
    print(line)

print(" " + "-" * ((x_max - x_min) * scale_x + 1))
print(" " + " ".join(f"{x:2d}" if x % 5 == 0 else " " for x in range(x_min, x_max + 1)))
