const std = @import("std");
const mtx = @import("matrix.zig");

const Matrix = mtx.Matrix;

pub fn LayerNorm(comptime T: type) type {
    return struct {
        gamma: []T,
        beta: []T,


        const Self = @This();
        pub fn apply(self: Self, mat: Matrix(T)) void {
            const dim: T = @floatFromInt(mat.height);
            for (0..mat.height) |i| {
                const row = mat.row(i);
                var mean: T = 0;

                for (row) |e| {
                    mean += e;
                }
                mean /= dim;
                for (row) |*e| {
                    e.* -= mean;
                }

                var diviation: T = 0;
                for (row) |e| {
                    diviation += e*e;
                }
                diviation /= dim;
                diviation = @sqrt(diviation);

                for (row) |*e| {
                    e.* /= diviation; 
                }

                for (row, self.gamma, self.beta) |*e, g, b| {
                    e.* = e.* * g + b; 
                }
            }
        }

        test apply {
            var cont = [_]T {
                1, 2, 3, 4,
                2, 4, -1, 2,
            };

            const mat = Matrix(T) {
                .capacity = 8, 
                .height = 2,
                .width = 4,
                .ptr = &cont,
            };

            var gamma = [_]T { 1, -1, 0, 10 };
            var beta =  [_]T { 3, 2, 4, 1 };

            const layer = LayerNorm(T) {
                .gamma = &gamma,
                .beta = &beta,
            };

            layer.apply(mat);

            mat.print();
        }
    };
}

comptime {
    _ = LayerNorm(f32);
}
