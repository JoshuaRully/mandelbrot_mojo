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

    var image = light.shade(
        numpy_array, plt.cm.hot, colors.PowerNorm(0.3), "hsv", 0, 0, 1.5
    )
    plt.imshow(image)
    plt.axis("off")
    plt.show()


# get your mojo on
fn mandelbrot_kernel_SIMD[
    simd_width: Int
](c: ComplexSIMD[float_type, simd_width]) -> SIMD[int_type, simd_width]:
    """A vectorized implementation of the inner mandelbrot computation."""
    var cx = c.re
    var cy = c.im
    var x = SIMD[float_type, simd_width](0)
    var y = SIMD[float_type, simd_width](0)
    var y2 = SIMD[float_type, simd_width](0)
    var iters = SIMD[int_type, simd_width](0)

    # escapes once all pixels within vector lange are done
    var t: SIMD[DType.bool, simd_width] = True
    for _ in range(MAX_ITERS):
        if not any(t):
            break
        y2 = y * y
        y = x.fma(y + y, cy)
        t = x.fma(x, y2) <= 4
        x = x.fma(x, cx - y2)
        iters = t.select(iters + 1, iters)
    return iters


fn run_mandelbrot(parallel: Bool) raises -> Float64:
    var matrix = Matrix[int_type, height, width]()

    @parameter
    fn worker(row: Int):
        var scale_x = (max_x - min_x) / width
        var scale_y = (max_y - min_y) / height

        @parameter
        fn compute_vector[simd_width: Int](col: Int):
            """Each time we operate on a `simd_width` vector of pixels."""
            var cx = min_x + (col + iota[float_type, simd_width]()) * scale_x
            var cy = min_y + row * scale_y
            var c = ComplexSIMD[float_type, simd_width](cx, cy)
            matrix.store(row, col, mandelbrot_kernel_SIMD[simd_width](c))

        # Vectorize the call to compute_vector where call gets a chunk of pixels.
        vectorize[compute_vector, simd_width](width)

    @parameter
    fn bench():
        for row in range(height):
            worker(row)

    @parameter
    fn bench_parallel():
        parallelize[worker](height, height)

    var time: Float64 = 0
    if parallel:
        time = benchmark.run[bench_parallel](max_runtime_secs=0.5).mean(unit)
    else:
        time = benchmark.run[bench](max_runtime_secs=0.5).mean(unit)

    show_plot(matrix)
    matrix.data.free()
    return time


fn main() raises -> None:
    # TODO: benchmark naive implementation
    # show_plot(compute_mandelbrot())
    var vectorized = run_mandelbrot(parallel=False)
    var parallelized = run_mandelbrot(parallel=True)
    print("Number of physical cores:", num_physical_cores())
    print("Vectorized:", vectorized, "ms")
    print("Parallelized:", parallelized, "ms")
    print("Parallel speedup:", vectorized / parallelized)
    return None
