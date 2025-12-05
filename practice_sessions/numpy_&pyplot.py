import numpy as np

# Creating 1D and 2D arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[10, 20, 30], [40, 50, 60]])

print("1D Array:", arr1)
print("2D Array:\n", arr2)

print("#________________________________________________________________________#")

arr = np.array([[1, 2, 3], [4, 5, 6]])

print("Shape:", arr.shape)
print("Dimensions:", arr.ndim)
print("Size:", arr.size)
print("Data type:", arr.dtype)

print("#________________________________________________________________________#")

zeros = np.zeros((2, 3))
ones = np.ones((3, 2))
range_arr = np.arange(0, 10, 2)
# in arange...we nee integers btw 0 to 10 with gap of 2
space_arr = np.linspace(0, 1, 5)
# in linespace...the initial number marks the starting of array...the second number is the number with which array ends....and the 3rd number gives the total count of numbers that should be present in our array

print("Zeros:\n", zeros)
print("Ones:\n", ones)
print("Arange:", range_arr)
print("Linspace:", space_arr)

print("#________________________________________________________________________#")

arr = np.array([[10, 20, 30],
                [40, 50, 60],
                [70, 80, 90]])

print("Element at (0,1):", arr[0, 1])
print("2nd Row:", arr[1, :])
print("3rd Column:", arr[:, 2])
print("Slice (2x2):\n", arr[:2, :2])
#arr[num,num]...is used when we wish to retrive an element, row or column from an array...it can also be used to slice the array

print("#________________________________________________________________________#")

a = np.array([10, 20, 30])
b = np.array([1, 2, 3])

print("Addition:", a + b)
print("Subtraction:", a - b)
print("Multiplication:", a * b)
print("Division:", a / b)
print("Square:", a ** 2)
#trying to power by the elements of second array
print("Power of the elements of another array",a**b)


print("#________________________________________________________________________#")

arr = np.array([5, 10, 15, 20])

print("Sum:", np.sum(arr))
print("Mean:", np.mean(arr))
print("Median:", np.median(arr))
print("Standard Deviation:", np.std(arr))
print("Max:", np.max(arr))
print("Min:", np.min(arr))

print("#________________________________________________________________________#")

m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[5, 6], [7, 8]])

print("Element-wise Multiplication:\n", m1 * m2)
print("Matrix Multiplication:\n", np.dot(m1, m2))
print("Transpose:\n", m1.T)

print("#________________________________________________________________________#")

np.random.seed(0)
print("Random Integers:\n", np.random.randint(1, 10, (2, 3)))
#randint gives random integers btw the first and second number ...and it makes an array of size given by programmer
print("Random Floats:\n", np.random.rand(2, 3))
#float values ✅ it includes 0.0, ❌ but excludes 1.0.

print("Normal Distribution:\n", np.random.randn(5))
#np.random.randn() generates numbers from the standard normal (Gaussian) distribution,which has: Mean (μ) = 0 ; Standard deviation (σ) = 1 ; So the values are centered around 0, and most of them lie roughly between -3 and +3.

print("#________________________________________________________________________#")

arr = np.array([5, 10, 15, 20, 25])
mask = arr > 12
print("Array:", arr)
print("Mask:", mask)
print("Filtered:", arr[mask])
print("Filtered_without_naming",arr[arr>12])
print("#________________________________________________________________________#")

matrix = np.array([[1, 2, 3], [4, 5, 6]])

print("Sum (axis=0):", np.sum(matrix, axis=0))
#axis=0 is summing alongside rows and axis=1 is summing alongside columns
print("Sum (axis=1):", np.sum(matrix, axis=1))
print("Mean (axis=0):", np.mean(matrix, axis=0))
print("Cumulative Sum:", np.cumsum(matrix))
#summing the next digit with previous sum

print("#________________________________________________________________________#")

arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
vector = np.array([10, 20, 30])

print("After Broadcasting:\n", arr + vector)
#vector added to each row

print("#________________________________________________________________________#")

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("Vertical Stack:\n", np.vstack((a, b)))
# output...similar to 2,3 array
print("Horizontal Stack:\n", np.hstack((a, b)))
#output:  [1 2 3 4 5 6]

arr = np.arange(10)
split = np.array_split(arr, 3)
#indices_or_section denotes how many sections do we want out random integers to split into
print("Splitted Arrays:", split)

print("#________________________________________________________________________#")

A = np.array([[2, 1], [1, 3]])
B = np.array([8, 18])

X = np.linalg.solve(A, B)

print("Matrix A:\n", A)
print("Vector B:", B)
print("Solution X:", X)

print("#________________________________________________________________________#")

arr = np.array([[10, 20, 30], [40, 50, 60]])

# Save array
np.save("my_array.npy", arr)

# Load array
loaded = np.load("my_array.npy")

print("Original Array:\n", arr)
print("Loaded Array:\n", loaded)

print("#________________________________________________________________________#")

arr = np.array([12, 5, 7, 2, 9])

print("Original:", arr)
print("Sorted:", np.sort(arr))
print("Indices that would sort:", np.argsort(arr))
print("Search position for 6:", np.searchsorted(arr, 6))

print("#________________________________________________________________________#")
import plotly.express as px

data = {
    "Year": [2018, 2019, 2020, 2021, 2022],
    "Sales": [100, 150, 130, 170, 180]
}

fig = px.line(data, x="Year", y="Sales", title="Sales Growth Over Years")
fig.show()

import plotly.express as px

data = {
    "Height": [150, 160, 170, 180, 190],
    "Weight": [55, 60, 70, 80, 90],
    "Gender": ["F", "F", "M", "M", "M"]
}

fig = px.scatter(data, x="Height", y="Weight", color="Gender", title="Height vs Weight by Gender")
fig.show()

import plotly.express as px

data = {
    "Category": ["A", "B", "C", "D"],
    "Revenue": [120, 300, 180, 250]
}

fig = px.bar(data, x="Category", y="Revenue", title="Revenue by Category", color="Category")
fig.show()
import plotly.express as px

data = {
    "Fruit": ["Apple", "Banana", "Grapes", "Orange"],
    "Quantity": [30, 15, 25, 20]
}

fig = px.pie(data, names="Fruit", values="Quantity", title="Fruit Market Share")
fig.show()
import plotly.express as px

data = {
    "Country": ["India", "USA", "Canada", "Germany", "Australia"],
    "GDP": [3.7, 25.5, 2.2, 4.9, 1.8]
}

fig = px.choropleth(data, locations="Country", locationmode="country names",
                    color="GDP", title="World GDP (Trillions USD)",
                    color_continuous_scale="Viridis")
fig.show()
import plotly.express as px
import numpy as np

data = np.random.randn(500)

fig = px.histogram(data, nbins=20, title="Random Data Distribution")
fig.show()
import plotly.express as px

data = {
    "Department": ["HR", "IT", "Finance", "IT", "HR", "Finance"],
    "Salary": [40, 60, 55, 70, 45, 65]
}

fig = px.box(data, x="Department", y="Salary", title="Salary Distribution by Department")
fig.show()

import plotly.graph_objects as go

years = [2018, 2019, 2020, 2021, 2022]
sales = [100, 150, 130, 170, 180]
profit = [20, 30, 25, 35, 40]

fig = go.Figure()
fig.add_trace(go.Scatter(x=years, y=sales, mode='lines+markers', name='Sales'))
fig.add_trace(go.Scatter(x=years, y=profit, mode='lines+markers', name='Profit'))

fig.update_layout(title="Sales and Profit Over Time", xaxis_title="Year", yaxis_title="Value")
fig.show()

import plotly.express as px
import numpy as np

z = np.random.rand(5, 5)
fig = px.imshow(z, text_auto=True, color_continuous_scale="Plasma", title="Random Heatmap")
fig.show()

import plotly.express as px
import numpy as np

df = {
    "x": np.random.rand(50),
    "y": np.random.rand(50),
    "z": np.random.rand(50),
    "color": np.random.randint(0, 10, 50)
}

fig = px.scatter_3d(df, x="x", y="y", z="z", color="color", title="3D Scatter Plot Example")
fig.show()
