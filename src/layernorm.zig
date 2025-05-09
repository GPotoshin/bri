const std = @import("std");
const mtx = @import("matrix.zig");

const Matrix = mtx.Matrix;

pub fn LayerNorm(comptime T: type) type {
    return struct {
        gamma: []T,
        beta: []T,

        const Self = @This();
        pub fn init(allocator: std.mem.Allocator, dim: u32) !Self {
            return LayerNorm(T) {
                .gamma = try allocator.alloc(T, dim),
                .beta = try allocator.alloc(T, dim),
            };
        }

        pub fn isEqualTo(self: Self, expected: Self, delta: T) bool {
            if (
                mtx.compareVectorDelta(T, self.gamma, expected.gamma, delta) and
                mtx.compareVectorDelta(T, self.beta, expected.beta, delta)
            ) {
                return true;
            }
            return false;
        }

        pub fn fillRandom(self: Self, rand: std.Random, k: T) void {
            mtx.fillVecRandom(T, rand, self.gamma, k);
            mtx.fillVecRandom(T, rand, self.beta, k);
        }

        pub fn copyValuesFrom(dest: *Self, source: Self) !void {
            if (dest.gamma.len < source.gamma.len) {
                return error.IncompatibleObjects;
            }
            std.mem.copyBackwards(T, dest.gamma, source.gamma);
            if (dest.beta.len < source.beta.len) {
                return error.IncompatibleObjects;
            }
            std.mem.copyBackwards(T, dest.beta, source.beta);
        }

        pub fn destroy(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.gamma);
            self.gamma.len = 0;
            allocator.free(self.beta);
            self.beta.len = 0;
        }

        pub fn apply(self: Self, mat: Matrix(T)) void {
            const dim: T = @floatFromInt(mat.width);
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

        pub fn writeWeights(self: Self, writer: anytype) !void {
            mtx.writeVector(T, writer, self.gamma) catch |e| {
                std.log.err("Can't write gamma vector\n", .{});
                return e;
            };
            mtx.writeVector(T, writer, self.beta) catch |e| {
                std.log.err("Can't write beta vector\n", .{});
                return e;
            };
        }

        pub fn readWeights(self: Self, reader: anytype) !void {
            mtx.readVector(T, reader, self.gamma) catch |e| {
                std.log.err("Can't read gamma vector\n", .{});
                return e;
            };
            mtx.readVector(T, reader, self.beta) catch |e| {
                std.log.err("Can't read beta vector\n", .{});
                return e;
            };
        }

        test apply {
            var cont = [_]T {
                1, 2, 3, 4,
                2, 4, -1, 2,
            };

            const mat = Matrix(T) {
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

            const expected = [_]T {1.658359214, 2.447213595, 4, 14.41640786,
                3.140028008, 0.7397479244, 4, 2.400280084};
            try std.testing.expect(mtx.compareVectorDelta(T, &expected, mat.toSlice(), 0.0001));

        }
    };
}

comptime {
    _ = LayerNorm(f32);
}
