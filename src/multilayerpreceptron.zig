const std = @import("std");
const mtx = @import("matrix.zig");
const Matrix = mtx.Matrix;

pub const MultilayerPreceptronHeader = struct {
    version: u32,
    type_len: u32,
    in_dim: u32,
    mid_dim: u32,
    out_dim: u32,
    max_seq_len: u32,

    const Self = @This();
    pub fn write(self: Self, writer: anytype) !void {
        writer.writeInt(u32, self.version, .little) catch |e| {
            std.log.err("Can't write version to a file\n", .{});
            return e;
        };
        writer.writeInt(u32, self.type_len, .little) catch |e| {
            std.log.err("Can't write type_len to a file\n", .{});
            return e;
        };
        writer.writeInt(u32, self.in_dim, .little) catch |e| {
            std.log.err("Can't write in_dim to a file\n", .{});
            return e;
        };
        writer.writeInt(u32, self.mid_dim, .little) catch |e| {
            std.log.err("Can't write mid_dim to a file\n", .{});
            return e;
        };
        writer.writeInt(u32, self.out_dim, .little) catch |e| {
            std.log.err("Can't write out_dim to a file\n", .{});
            return e;
        };
        writer.writeInt(u32, self.max_seq_len, .little) catch |e| {
            std.log.err("Can't write max_seq_len to a file\n", .{});
            return e;
        };
    }

    pub fn read(self: Self, reader: anytype) !void {
        reader.readInt(u32, self.version, .little) catch |e| {
            std.log.err("Can't read version to a file\n", .{});
            return e;
        };
        reader.readInt(u32, self.type_len, .little) catch |e| {
            std.log.err("Can't read type_len to a file\n", .{});
            return e;
        };
        reader.readInt(u32, self.in_dim, .little) catch |e| {
            std.log.err("Can't read in_dim to a file\n", .{});
            return e;
        };
        reader.readInt(u32, self.mid_dim, .little) catch |e| {
            std.log.err("Can't read mid_dim to a file\n", .{});
            return e;
        };
        reader.readInt(u32, self.out_dim, .little) catch |e| {
            std.log.err("Can't read out_dim to a file\n", .{});
            return e;
        };
        reader.readInt(u32, self.max_seq_len, .little) catch |e| {
            std.log.err("Can't read max_seq_len to a file\n", .{});
            return e;
        };
    }
};

// @NotTested
// only 2 layers for now
pub fn MultilayerPreceptron(comptime T: type) type {
    return struct {
        header: MultilayerPreceptronHeader,

        mat1: Matrix(T),
        mid: Matrix(T),
        vec1: []T,
        mat2: Matrix(T),
        vec2: []T,
        out: Matrix(T),
        
        const Self = @This();

        pub fn compute(self: *Self, input: Matrix(T)) !void {
            try mtx.matprod(T, self.mat1, input, &self.mid);
            try self.mid.addRow(self.vec1);
            self.mid.reLU();
            try mtx.matprod(T, self.mat2, self.mid, &self.out);
            try self.out.addRow(self.vec2);
        } 

        pub fn init(allocator: std.mem.Allocator, header: MultilayerPreceptronHeader) !Self {
            var retval: Self = undefined;
            retval.header = header;
            try retval.allocateFromHeader(allocator);
            return retval;
        }

        pub fn allocateFromHeader(self: *Self, allocator: std.mem.Allocator) !void {
            self.mat1 = try Matrix(T).init(allocator, self.header.mid_dim, self.header.in_dim);
            self.vec1 = try allocator.alloc(T, self.header.mid_dim);
            self.mid = try Matrix(T).init(allocator, self.header.max_seq_len, self.header.mid_dim);
            self.mat2 = try Matrix(T).init(allocator, self.header.out_dim, self.header.mid_dim);
            self.vec2 = try allocator.alloc(T, self.header.out_dim);
        }

        pub fn destroy(self: *Self, allocator: std.mem.Allocator) void {
            self.mat1.destroy(allocator);
            allocator.free(self.vec1);
            self.mid.destroy(allocator);
            self.mat1.destroy(allocator);
            allocator.free(self.vec2);
            self.mat2.destroy(allocator);
        }

        pub fn readWeights(self: Self, reader: anytype) !void {
            self.mat1.read(reader) catch |e| {
                std.log.err("can't read first matrix ({}x{}) {}\n",
                    .{self.mat1.height, self.mat1.width, T});
                return e;
            };
            mtx.readVector(T, reader, self.vec1) catch |e| {
                std.log.err("can't read first vector ({}) {}\n",
                    .{self.vec1.len, T});
                return e;
            };
            self.mat2.read(reader) catch |e| {
                std.log.err("can't read second matrix ({}x{}) {}\n",
                    .{self.mat2.height, self.mat2.width, T});
                return e;
            };
            mtx.readVector(T, reader, self.vec2) catch |e| {
                std.log.err("can't read second vector ({}) {}\n",
                    .{self.vec2.len, T});
                return e;
            };
        }

        pub fn writeWeights(self: Self, writer: anytype) !void {
            self.mat1.write(writer) catch |e| {
                std.log.err("can't write first matrix ({}x{}) {}\n",
                    .{self.mat1.height, self.mat1.width, T});
                return e;
            };
            mtx.writeVector(T, writer, self.vec1) catch |e| {
                std.log.err("can't write first vector ({}) {}\n",
                    .{self.vec1.len, T});
                return e;
            };
            self.mat2.write(writer) catch |e| {
                std.log.err("can't write second matrix ({}x{}) {}\n",
                    .{self.mat2.height, self.mat2.width, T});
                return e;
            };
            mtx.writeVector(T, writer, self.vec2) catch |e| {
                std.log.err("can't write second vector ({}) {}\n",
                    .{self.vec2.len, T});
                return e;
            };
        }

        test compute {
            var data1 = [_]T {
                1, 2, 3,
                -1, 2, -3,
            };

            var vdata1 = [_]T { 0.1, -0.1 };

            var data2 = [_]T {
                0.1, 0.4,
                0.3, -0.2,
                0.5, -0.1,
            };
            var vdata2 = [_]T { 0.3, -0.2, 0.0};
            var outdata: [30]T = undefined;
            var middata: [20]T = undefined;

            var mlp = MultilayerPreceptron(T) {
                .header = MultilayerPreceptronHeader {
                    .version = 0,
                    .type_len = @sizeOf(T),
                    .in_dim = 3,
                    .mid_dim = 2,
                    .out_dim = 3,
                    .max_seq_len = 10,
                },

                .mat1 = Matrix(T) {
                    .capacity = 6,
                    .height = 2,
                    .width = 3,
                    .ptr = &data1,
                },
                .mat2 = Matrix(T) {
                    .capacity = 6,
                    .height = 3,
                    .width = 2,
                    .ptr = &data2,
                },
                .vec1 = &vdata1,
                .vec2 = &vdata2,

                .mid = Matrix(T) {
                    .capacity = middata.len,
                    .height = 0,
                    .width = 0,
                    .ptr = &middata,
                },

                .out = Matrix(T) {
                    .capacity = outdata.len,
                    .height = 0,
                    .width = 0,
                    .ptr = &outdata,
                },
            };

            var testdata = [_]T {
                9, 8,  7,
                -1, -2, -3,
            };
            const testseq = Matrix(T) {
                .capacity = 6,
                .height = 2,
                .width = 3,
                .ptr = &testdata,
            };

            try mlp.compute(testseq);

            const expected = [_]T {4.91, 13.63, 23.05, 2.66, -1.38, -0.59};
            try @import("test.zig").compare_delta(T, &expected, mlp.out.toSlice(), 0.00001);

        }
    };
}

comptime {
    _ = MultilayerPreceptron(f32);
}
