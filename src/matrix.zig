const std = @import("std");
const parsing = @import("parsing.zig");

const AlgebraicError = error {
    IncompatibleObjects,
};

/// Matrix for now is intended to be used with arenas, or manually free it's
/// pointer but there are no checks for double freeing!
pub fn Matrix(comptime T: type) type {
    return struct {
        ptr: []T,
        height: usize,
        width: usize,

        const Self = @This();

        /// allocates memoty and returns a matrix of given size
        pub fn init(allocator: std.mem.Allocator, height: usize, width: usize) !Self {
            var retval: Self = undefined;
            try retval.allocate(allocator, height*width);
            retval.height = height;
            retval.width = width;
            return retval;
        }

        pub fn copy(self: Self, allocator: std.mem.Allocator) !Self {
            var retval = self;
            const ptr = try allocator.alloc(T, self.height*self.width);
            retval.ptr = ptr;
            std.mem.copyBackwards(T, retval.ptr, self.toSlice());
            return retval;
        }

        /// Copies data to preallocatoed matrix from which it is called.
        /// Returns an error if the underlying slice cannot hold all the data
        pub fn copyValuesFrom(self: *Self, mat: Self) !void {
            if (self.ptr.len < mat.height*mat.width) {
                std.log.err("", .{});
                return AlgebraicError.IncompatibleObjects;
            }
            std.mem.copyBackwards(T, self.ptr, mat.toSlice());
        }

        pub fn isEqualTo(self: Self, expected: Self, delta: T) bool {
            if (self.height != expected.height) {
                std.debug.print("Inequal height\n", .{});
                return false;
            }
            if (self.width != expected.width) {
                std.debug.print("Inequal width\n", .{});
                return false;
            }

            return compareVectorDelta(T, expected.toSlice(), self.toSlice(), delta);
        }

        /// sets height and width to zero. The idea is that we can have an arbitrary
        /// data length and calculus should be coherent
        pub fn allocate(self: *Self, allocator: std.mem.Allocator, capacity: usize) !void {
            self.height = 0;
            self.width = 0;
            self.ptr = try allocator.alloc(T, capacity);
        }

        pub fn destroy(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.ptr);
            self.height = 0;
            self.width = 0;
        }

        /// line-column convention. Not intended for high load. Use direct
        /// poniter instead! Indencing starts with ZERO
        pub inline fn at(self: Self, i: usize, j: usize) ?T {
            if (i > self.height or j > self.width) {
                return null;
            }
            return self.ptr[i*self.width+j];
        }

        pub inline fn row(self: Self, i: usize) []T {
            const start = i*self.width;
            return self.ptr[start..start+self.width];
        }

        /// returns the slice of size of matrix, not its capacity
        pub inline fn toSlice(self: Self) []T {
            return self.ptr[0..self.height*self.width];
        }

        // 
        pub inline fn submatrix(self: Self, start: usize, end: usize) ?Self {
            if (start < 0 or start >= end or end > self.height) {
                std.log.err("wrong indices for sub matrix\n", .{});
                return null;
            }

            return Matrix(T){
                .height = end - start,
                .width = self.width,
                .ptr = self.ptr[self.width*start..],
            };
        }

        /// This overwrites with random data with a uniform distribution over
        /// [-k, k). Parameter k needs to be small as otherwise calculus explods
        pub fn fillRandom(self: Self, rand: std.Random, k: T) void {
            for (0..self.height) |i| {
                const vec = self.row(i);
                fillVecRandom(T, rand, vec, k);
            }
        }

        /// writes matrix numbers as continuous array without any other
        /// inforamtion
        pub fn write(self: Self, writer: anytype) !void {
            if (self.height*self.width > self.ptr.len) {
                return error.OutOfBound;
            }
            const ptr: [*]u8 = @ptrCast(self.ptr.ptr);
            const size: usize = self.height*self.width*@sizeOf(T);
            _ = try writer.write(ptr[0..size]);
        }
        /// reads a matrix as continuous array with length taken from matrix
        pub fn read(self: Self, reader: anytype) !void {
            if (self.height*self.width > self.ptr.len) {
                return error.OutOfBound;
            }
            const ptr: [*]u8 =  @ptrCast(self.ptr.ptr);
            const size: usize = self.height*self.width*@sizeOf(T);
            const buffer: []u8 = ptr[0..size];
            _ = try reader.read(buffer);
        }

        pub fn initFromFile(allocator: std.mem.Allocator, file: std.fs.File) !Matrix(T) {
            const reader = file.reader();

            const typelen = reader.readInt(u32, .little) catch |e| {
                std.log.err("Can't read typelen from a matrix file\n", .{});
                return e;
            };
            if (typelen != @sizeOf(T)) {
                std.log.err("matrix file contains a matrix of a different type\n", .{});
                return error.DataConflict;
            }
            const width = reader.readInt(u32, .little) catch |e| {
                std.log.err("Can't read width from a matrix file\n", .{});
                return e;
            };
            const height = reader.readInt(u32, .little) catch |e| {
                std.log.err("Can't read height from a matrix file\n", .{});
                return e;
            };

            const mat = try Matrix(T).init(allocator, height, width);
            try mat.read(reader);
            return mat;
        }

        pub fn writeToFile(self: Matrix(T), file: std.fs.File) !void {
            try file.seekTo(0);
            const writer = file.writer();

            writer.writeInt(u32, @sizeOf(T), .little) catch |e| {
                std.log.err("Can't write typelen to an embending file\n", .{});
                return e;
            };
            writer.writeInt(u32, @truncate(self.width), .little) catch |e| {
                std.log.err("Can't write width to a matrix file\n", .{});
                return e;
            };
            writer.writeInt(u32, @truncate(self.height), .little) catch |e| {
                std.log.err("Can't write height to a matrix file\n", .{});
                return e;
            };

            try self.write(writer);
            try file.setEndPos(try file.getPos());
        }

        pub fn print(mat: Self) void {
            for (0..mat.height) |i| {
                for (0..mat.width) |j| {
                    std.debug.print("{}\t", .{mat.at(i,j).?});
                }
                std.debug.print("\n", .{});
            }
            std.debug.print("height: {}, width: {}, ptr: {*}, len: {}\n", .{mat.height, mat.width, mat.ptr.ptr, mat.ptr.len});
        }

        pub fn scale(self: Self, k: T) void {
            for(0..self.height) |i| {
                const vec = self.row(i);
                for(vec) |*v| {
                    v.* *= k;
                }
            }
        }

        pub fn add(self: Self, mat: Self) !void {
            if (self.height != mat.height or self.width != mat.width or
                self.ptr.len < self.height*self.width or
                mat.ptr.len < mat.height*mat.width) {
                return error.IncompatibleObjects;
            }

            for (self.toSlice(), mat.toSlice()) |*s, m| {
                s.* += m;
            }
        }

        /// adds to each row of mat the vect
        /// mat: NxM, vec: M
        pub fn addRow(self: Self, vec: []T) !void {
            if (self.width != vec.len) {
                std.log.err("Incoherent mat and vect sizes!\n", .{});
                return error.IncompatibleObjects;
            }

            // dumb implementation
            for (0..self.height) |i| {
                for (0..self.width) |j| {
                    self.ptr[i*self.width+j] += vec[j];
                }
            }
        }

        /// adds to each row of mat the vect
        /// mat: NxM, vec: M
        pub fn addCol(self: Self, vec: []T) !void {
            if (self.height != vec.len) { // @Rq: maybe do < or just to max
                std.log.err("Incoherent self and vect sizes!\n", .{});
                return error.IncompatibleObjects;
            }

            // dumb implementation
            for (0..self.height) |i| {
                for (0..self.width) |j| {
                    self.ptr[i*self.width+j] += vec[i];
                }
            }
        }

        pub fn reLU(self: Self) void {
            for (self.toSlice()) |*e| {
                e.* = @max(0, e.*);
            }
        }

        const empty_arr = &[0]T{};
        pub const empty = Matrix(T) {
            .height = 0,
            .width = 0,
            .ptr = empty_arr,
        };

        pub fn fromSlice(slice: []T, height: u32, width: u32) Self {
            return Matrix(T) {
                .height = height,
                .width = width,
                .ptr = slice[0..@min(slice.len, height*width)],
            };
        }


// ---------------------------- TESTS --------------------------------------

        test init {
            var mat = try Matrix(T).init(std.testing.allocator, 10, 10);
            mat.destroy(std.testing.allocator);
        }

        test allocate {
            var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
            defer arena.deinit();
            const allocator = arena.allocator();

            var mat: Matrix(T) = undefined;
            try mat.allocate(allocator, 10);
        }

        test at {
            var content = [_]T{
                1, 2, 3,
                4, 5, 6,
            };
            const mat = Matrix(T) {
                .height = 2,
                .width = 3,
                .ptr = &content,
            };
            try std.testing.expectEqual(5, mat.at(1,1).?);
        }

        test row {
            var content = [_]T {
                1, 2, 3,
                4, 5, 6
            };
            const mat = Matrix(T) {
                .height = 2,
                .width = 3,
                .ptr = &content,
            };
            try std.testing.expectEqualSlices(T,
                &[_]T{4, 5, 6},
                mat.row(1),
            );
        }

        test submatrix {
            var content = [_]T {
                1, 2,
                3, 4,
                5, 6
            };
            const mat = Matrix(T) {
                .height = 3,
                .width = 2,
                .ptr = &content,
            };
            const sub = mat.submatrix(0, 2);
            try std.testing.expectEqualSlices(T, &[_]T {
                1, 2,
                3, 4,
            },
            sub.?.toSlice());
        }

        test fillRandom {
            var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
            defer arena.deinit();
            const allocator = arena.allocator();

            var mat = try Matrix(T).init(allocator, 3, 3);
            var xoshiro = std.Random.Xoshiro256.init(123);
            const rand = xoshiro.random();

            mat.fillRandom(rand, 1);
        }

        test write {
            var content = [6]T {1, 2, 3, 4, 5, 6};
            const mat = Matrix(T) {
                .height = 2,
                .width = 3,
                .ptr = &content,
            };

            var buffer = [6]T {0, 0, 0, 0, 0, 0};
            const handler: [*]u8 = @ptrCast((buffer[0..]).ptr);
            var stream = std.io.fixedBufferStream(handler[0..6*@sizeOf(T)]);
            const writer = stream.writer();

            try mat.write(writer);

            try std.testing.expectEqual(content, buffer);
        }

        test read {
            var content = [6]T {0, 0, 0, 0, 0, 0};
            const mat = Matrix(T) {
                .height = 2,
                .width = 3,
                .ptr = &content,
            };

            var buffer = [6]T {1, 2, 3, 4, 5, 6};
            const handler: [*]u8 = @ptrCast((buffer[0..]).ptr);
            var stream = std.io.fixedBufferStream(handler[0..6*@sizeOf(T)]);
            const reader = stream.reader();

            try mat.read(reader);
            try std.testing.expectEqual(buffer, content);
        }
        
        test scale {
            var content = [6]T {1, 2, 3, 4, 5, 6};
            const mat = Matrix(T) {
                .height = 2,
                .width = 3,
                .ptr = &content,
            };
            mat.scale(2);
            try std.testing.expectEqualSlices(T, &[_]T {
                2, 4, 6,
                8, 10, 12,
            }, mat.toSlice());
        }

        test add {
            var content = [6]T {1, 2, 3, 4, 5, 6};
            const mat = Matrix(T) {
                .height = 2,
                .width = 3,
                .ptr = &content,
            };
            try mat.add(mat);
            try std.testing.expectEqualSlices(T, &[_]T {
                2, 4, 6,
                8, 10, 12,
            }, mat.toSlice());
        }

        test addRow {
            var content = [6]T {
                1, 2, 3,
                4, 5, 6
            };
            var forme = [3]T {1, 2, 3};
            const mat = Matrix(T) {
                .height = 2,
                .width = 3,
                .ptr = &content,
            };
            try mat.addRow(&forme);
            try std.testing.expectEqualSlices(T, &[_]T {
                2, 4, 6,
                5, 7, 9,
            },
            mat.toSlice());
        }

        test addCol {
            var content = [6]T {
                1, 2, 3,
                4, 5, 6
            };
            var vect = [2]T {1, 2};
            const mat = Matrix(T) {
                .height = 2,
                .width = 3,
                .ptr = &content,
            };
            try mat.addCol(&vect);
            try std.testing.expectEqualSlices(T, &[_]T {
                2, 3, 4,
                6, 7, 8,
            },
            mat.toSlice());
        }
    };
}

comptime {
_ = Matrix(f32);
}


/// computes `(mat1 x mat2^t)^t` because the second argement is trited as an
/// array of vectors a and vetors are lines in this library. The reason is
/// memory locality. mat1: MxN, mat2: LxN, out LxM
pub fn matprod(comptime T: type, mat1: Matrix(T), mat2: Matrix(T), out: *Matrix(T)) !void {
    if (mat1.width != mat2.width) {
        std.log.err("Wrong dimensions in product mat1 and mat2\n", .{});
        return error.IncompatibleObjects; 
    }

    if (out.ptr.len < mat1.height*mat2.height) {
        std.log.err("Can't store the output\n", .{});
        return error.OutOfBound; 
    }

    out.width = mat1.height;
    out.height = mat2.height;

    for (0..mat2.height) |i| {
        const vect = mat2.row(i);
        var out_vect = out.row(i);
        for (0..mat1.height) |j| {
            out_vect[j] = 0;
            const vect2 = mat1.row(j);
            for (0..vect.len) |k| {
                out_vect[j] += vect2[k]*vect[k];
            }
        }
        for (mat1.height..out.width) |j| {
            out_vect[j] = 0;
        }
    }
    for (mat2.height..out.height) |i| {
        const vect = out.row(i);
        for (vect) |*v| {
            v.* = 0;
        }
    }
}

/// computes `(mat1 x mat2^t)^t` and adds it to the out. Can't be used in parallel.
/// mat1: MxN, mat2: LxN, out LxM
pub fn mataddprod(comptime T: type, mat1: Matrix(T), mat2: Matrix(T), out: *Matrix(T)) !void { // @Testit
    if (mat1.width != mat2.width) {
        std.log.err("Wrong dimensions in product mat1 and mat2\n", .{});
        return error.IncompatibleObjects; 
    }

    if (out.ptr.len < mat1.height*mat2.height) {
        std.log.err("Can't store the output\n", .{});
        return error.OutOfBound; 
    }

    out.width = mat1.height;
    out.height = mat2.height;

    for (0..mat2.height) |i| {
        const vect = mat2.row(i);
        var out_vect = out.row(i);
        for (0..mat1.height) |j| {
            const vect2 = mat1.row(j);
            for (0..vect.len) |k| {
                out_vect[j] += vect2[k]*vect[k];
            }
        }
    }
}

/// memory for the output should be preallocated and the sizes should be coherent
/// mat: NxM, input: LxM, vect: N, output: LxN
pub fn affine(comptime T: type, payload: struct {mat: Matrix(T), input: Matrix(T),
    vect: []T, output: *Matrix(T)}) !void {
    // the sympliest algo
    // const in_dim: u32 = payload.mat.width;
    // const out_dim: u32 = payload.mat.height;

    try matprod(T, payload.mat, payload.input, payload.output);
    try payload.output.addRow(payload.vect);
}


/// memory for the output should be preallocated and the sizes should be coherent
/// mat: NxM, input: LxM, vect: N, output: LxN
pub fn affine2(comptime T: type, payload: struct {mat: Matrix(T), input: Matrix(T),
    vect: []T, output: *Matrix(T)}) !void {
    // the sympliest algo
    // const in_dim: u32 = payload.mat.width;
    // const out_dim: u32 = payload.mat.height;

    try matprod(T, payload.mat, payload.input, payload.output);

    try payload.output.addCol(payload.vect);
}

/// funciton overwrites content of mat and does soft max on every row of mat
pub fn softmax(comptime T: type, mat: Matrix(T)) void {
    for (0..mat.height) |i| {
        const vect = mat.row(i);
        var sum: T = 0;

        for (vect) |*v| {
            v.* = @exp(v.*);
            sum += v.*;
        }

        for (vect) |*v| {
            v.* = v.*/sum;
        }
    }

}

pub fn fillVecRandom(comptime T: type, rand: std.Random, vec: []T, k: T) void {
    for (vec) |*v| {
        const f = rand.float(T);
        v.* = k*(2.0*f - 1.0);
    }
}

pub fn readVector(comptime T: type, reader: anytype, vec: []T) !void {
    const ptr: [*]u8 = @ptrCast(vec.ptr);
    const size: usize = vec.len*@sizeOf(T);
    _ = try reader.read(ptr[0..size]);
}


pub fn writeVector(comptime T: type, writer: anytype, vec: []T) !void {
    const ptr: [*]u8 = @ptrCast(vec.ptr);
    const size: usize = vec.len*@sizeOf(T);
    _ = try writer.write(ptr[0..size]);
}

pub fn compareVectorDelta(comptime T: type, expected: []const T, actual: []const T, delta: T) bool {
    if (expected.len != actual.len) {
        return false;
    }
    for (expected, actual) |e, a| {
        if (@abs(e-a) > delta) {
            return false;
        }
    }
    return true;
}

test matprod {
    var cont1 = [_]f32 {
        1, 2, 3,
        4, 5, 6,
    };
    var cont2 = [_]f32 {
        1, 0, 1,
        0, 2, 2,
        4, 2, 3,
        8, 0, 2
    };
    var cont3: [8]f32 = undefined;

    const mat1 = Matrix(f32) {
        .height = 2,
        .width = 3,
        .ptr = &cont1,
    };
    const mat2 = Matrix(f32) {
        .height = 4,
        .width = 3,
        .ptr = &cont2,
    };
    var out = Matrix(f32) {
        .height = 4,
        .width = 2,
        .ptr = &cont3,
    };
    
    try matprod(f32, mat1, mat2, &out);
    
    try std.testing.expectEqualSlices(f32, &[_]f32 {
        4, 10,
        10, 22,
        17, 44, 
        14, 44}, &cont3);
}

test affine {
    var cont1 = [_]f32 {
        1, 2, 3,
        4, 5, 6,
    };
    var cont2 = [_]f32 {
        1, 0, 1,
        0, 2, 2,
        4, 2, 3,
        8, 0, 2
    };
    var cont3: [8]f32 = undefined;

    const mat1 = Matrix(f32) {
        .height = 2,
        .width = 3,
        .ptr = &cont1,
    };
    const mat2 = Matrix(f32) {
        .height = 4,
        .width = 3,
        .ptr = &cont2,
    };
    var out = Matrix(f32) {
        .height = 4,
        .width = 2,
        .ptr = &cont3,
    };
    var vec = [_]f32 {1, 1};
    try affine(f32, .{.mat = mat1, .input = mat2, .vect = &vec, .output = &out});

    try std.testing.expectEqualSlices(f32, &[_]f32 {
        5, 11,
        11, 23,
        18, 45, 
        15, 45}, &cont3);
}

test affine2 {
    var cont1 = [_]f32 {
        1, 2, 3,
        4, 5, 6,
    };
    var cont2 = [_]f32 {
        1, 0, 1,
        0, 2, 2,
        4, 2, 3,
        8, 0, 2
    };
    var cont3: [8]f32 = undefined;

    const mat1 = Matrix(f32) {
        .height = 2,
        .width = 3,
        .ptr = &cont1,
    };
    const mat2 = Matrix(f32) {
        .height = 4,
        .width = 3,
        .ptr = &cont2,
    };
    var out = Matrix(f32) {
        .height = 4,
        .width = 2,
        .ptr = &cont3,
    };
    var vec = [_]f32 {1, 2, 3, 4};
    try affine2(f32, .{.mat = mat1, .input = mat2, .vect = &vec, .output = &out});

    try std.testing.expectEqualSlices(f32, &[_]f32 {
        5, 11,
        12, 24,
        20, 47, 
        18, 48}, &cont3);
}

test softmax {
    var cont = [_]f32 {
        1, 2,
        -1, -2,
        0, 0,
    };
    const mat = Matrix(f32) {
        .height = 3,
        .width = 2,
        .ptr = &cont,
    };

    softmax(f32, mat);

    try std.testing.expect(@abs(cont[0]-@exp(1.0)/(@exp(1.0)+@exp(2.0))) < 0.0001);
    try std.testing.expect(@abs(cont[1]-@exp(2.0)/(@exp(1.0)+@exp(2.0))) < 0.0001);
    try std.testing.expect(@abs(cont[2]-@exp(-1.0)/(@exp(-1.0)+@exp(-2.0))) < 0.0001);
    try std.testing.expect(@abs(cont[3]-@exp(-2.0)/(@exp(-1.0)+@exp(-2.0))) < 0.0001);
    try std.testing.expect(@abs(cont[4]-@exp(0.0)/(@exp(0.0)+@exp(0.0))) < 0.0001);
    try std.testing.expect(@abs(cont[5]-@exp(0.0)/(@exp(0.0)+@exp(0.0))) < 0.0001);
}

test fillVecRandom {
    var xoshiro = std.Random.Xoshiro256.init(123);
    const rand = xoshiro.random();
    var cont: [5]f32 = undefined;
    fillVecRandom(f32, rand, &cont, 1);
}

test readVector {
    var content = [6]f32 {0, 0, 0, 0, 0, 0};
    var buffer = [6]f32 {1, 2, 3, 4, 5, 6};
    const handler: [*]u8 = @ptrCast((buffer[0..]).ptr);
    var stream = std.io.fixedBufferStream(handler[0..6*@sizeOf(f32)]);
    const reader = stream.reader();

    try readVector(f32, reader, &content);
    try std.testing.expectEqual(buffer, content);
}

test writeVector {
    var buffer = [6]f32 {0, 0, 0, 0, 0, 0};
    var content = [6]f32 {1, 2, 3, 4, 5, 6};
    const handler: [*]u8 = @ptrCast((buffer[0..]).ptr);
    var stream = std.io.fixedBufferStream(handler[0..6*@sizeOf(f32)]);
    const writer = stream.writer();

    try writeVector(f32, writer, &content);
    try std.testing.expectEqual(buffer, content);
}

