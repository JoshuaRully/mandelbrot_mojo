import benchmark
from math import iota
from sys import num_physical_cores
from algorithm import parallelize, vectorize
from complex import ComplexFloat64, ComplexSIMD
from python import Python

alias float_type = DType.float32
alias int_type = DType.int32
alias simd_width = 2 * simdwidthof[float_type]()
alias unit = benchmark.Unit.ms

# parameters
alias width = 960
alias height = 960
alias MAX_ITERS = 200

alias min_x = -2.0
alias max_x = 0.6
alias min_y = -1.5
alias max_y = 1.5

@value
struct Matrix[type: DType, rows: Int, cols: Int]:
    var data: DTypePointer[type]

    fn __init__(inout self):
        self.data = DTypePointer[type].alloc(rows * cols)

    fn __getitem__(self, row: Int, col: Int) -> Scalar[type]:
        return self.data.load(row * cols + col)

    fn store[width: Int = 1](self, row: Int, col: Int, val: SIMD[type, width]):
        self.data.store[width=width](row * cols + col, val)

# https://en.wikipedia.org/wiki/Mandelbrot_set
# compute the number of steps to escape
fn mandelbrot_kernel(c: ComplexFloat64) -> Int:
    var z: ComplexFloat64 = c
    for i in range(MAX_ITERS):
        z = z * z + c
        if z.squared_norm() > 4:
            return i
    return MAX_ITERS

# TODO: clean up implicit typing
fn compute_mandelbrot() -> Matrix[float_type, height, width]:
    # create a matrix; each element corresponds to a pixel
    var matrix = Matrix[float_type, height, width]()

    var dx = (max_x - min_x) / width
    var dy = (max_y - min_y) / height

    var y = min_y
    for row in range(height):
        var x = min_x
        for col in range(width):
            matrix.store(row, col, mandelbrot_kernel(ComplexFloat64(x, y)))
            x += dx
        y += dy
    return matrix

fn show_plot[type: DType](matrix: Matrix[type, height, width]) raises:
    alias scale = 10
    alias dpi = 64

    var np = Python.import_module("numpy")
    var plt = Python.import_module("matplotlib.pyplot")
    var colors = Python.import_module("matplotlib.colors")

    var numpy_array = np.zeros((height, width), np.float64)

    for row in range(height):
        for col in range(width):
            # TODO: itemsets were removed in numpy==2.0.0; update to arr[idx] = val;
            numpy_array.itemset((row, col), matrix[row, col])

    var figure = plt.figure(1, [scale, scale * height // width], dpi)
    var light = colors.LightSource(315, 10, 0, 1, 1, 0)
    figure.add_axes([0.0, 0.0, 1.0, 1.0], False, 1)

    var image = light.shade(numpy_array, plt.cm.hot, colors.PowerNorm(0.3), "hsv", 0, 0, 1.5)
    plt.imshow(image)
    plt.axis("off")
    plt.show()

fn main() raises -> None:
    show_plot(compute_mandelbrot())
    return None