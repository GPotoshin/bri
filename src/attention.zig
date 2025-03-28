const std = @import("std");
const mtx = @import("matrix.zig");
const Matrix = mtx.Matrix;

pub const AttentionHeader = struct {
    version: u32 = 0,

    type_len: u32,

    seq_dim: u32,
    ctx_dim: u32,
    att_dim: u32,
    out_dim: u32,

    max_seq_len: u32,
    max_ctx_len: u32,


    const Self = @This();
    pub fn write(self: Self, writer: anytype) !void {
        writer.writeInt(u32, self.version, .little) catch |e| {
            std.log.err("Can't write version to a file\n", .{});
            return e;
        };
        writer.writeInt(u32, self.type_len, .little) catch |e| {
            std.log.err("Can't write type_size to a file\n", .{});
            return e;
        };
        writer.writeInt(u32, self.seq_dim, .little) catch |e| {
            std.log.err("Can't write seq_dim from file\n", .{});
            return e;
        };
        writer.writeInt(u32, self.ctx_dim, .little) catch |e| {
            std.log.err("Can't write ctx_dim from file\n", .{});
            return e;
        };
        writer.writeInt(u32, self.att_dim, .little) catch |e| {
            std.log.err("Can't write att_dim from file\n", .{});
            return e;
        };
        writer.writeInt(u32, self.out_dim, .little) catch |e| {
            std.log.err("Can't write out_dim from file\n", .{});
            return e;
        };
        writer.writeInt(u32, self.max_seq_len, .little) catch |e| {
            std.log.err("Can't write max_seq_len from file\n", .{});
            return e;
        };
        writer.writeInt(u32, self.max_ctx_len, .little) catch |e| {
            std.log.err("Can't write seq_dim from file\n", .{});
            return e;
        };
    }

    pub fn read(self: *Self, reader: anytype) !void {
        self.version = reader.readInt(u32, .little) catch |e| {
            std.log.err("Can't read version\n", .{});
            return e;
        };
        self.type_len = reader.readInt(u32, .little) catch |e| {
            std.log.err("Can't read type\n", .{});
            return e;
        };
        self.seq_dim = reader.readInt(u32, .little) catch |e| {
            std.log.err("Can't read seq_dim\n", .{});
            return e;
        };
        self.ctx_dim = reader.readInt(u32, .little) catch |e| {
            std.log.err("Can't read ctx_dim\n", .{});
            return e;
        };
        self.att_dim = reader.readInt(u32, .little) catch |e| {
            std.log.err("Can't read att_dim\n", .{});
            return e;
        };
        self.out_dim = reader.readInt(u32, .little) catch |e| {
            std.log.err("Can't read out_dim\n", .{});
            return e;
        };
        self.max_seq_len = reader.readInt(u32, .little) catch |e| {
            std.log.err("Can't read max_seq_len\n", .{});
            return e;
        };
        self.max_ctx_len = reader.readInt(u32, .little) catch |e| {
            std.log.err("Can't read max_ctx_len\n", .{});
            return e;
        };
    }
};


// @Todo: actually as calculus are happenning on gpu, in the future
// intermediate memory should be allocated for gpu!
pub fn Attention(comptime T: type) type {
    return struct {
        header: AttentionHeader,

        query_matrix: Matrix(T),
        query_vect: []T,

        key_matrix: Matrix(T),
        key_vect: []T,

        value_matrix: Matrix(T),
        value_vect: []T,

        query: Matrix(T),
        key: Matrix(T),
        value: Matrix(T),
        score: Matrix(T),

        out: Matrix(T),

        const Self = @This();

        /// allocates memory for a given header and creates an Attention block
        /// the output matrix should preallocated by caller
        pub fn init(allocator: std.mem.Allocator, header: AttentionHeader, out: Matrix(T)) !Self {
            if (out.height != header.max_seq_len or out.width != header.out_dim) {
                std.log.err("Output matrix is of a wrong size\n", .{});
                return error.DataConflict;
            }

            var retval: Self = undefined;
            retval.header = header;
            try retval.allocateForHeader(allocator);
            retval.out = out;

            return retval;
        }

        /// allocates the memory for all weights with dimensions from it's header
        pub fn allocateForHeader(self: *Self, allocator: std.mem.Allocator) !void {
            const header = self.header;
            self.query_matrix = try Matrix(T).init(allocator, header.att_dim, header.seq_dim);
            self.query_vect = try allocator.alloc(T, header.att_dim);
            self.query = try Matrix(T).init(allocator, header.max_seq_len, header.att_dim);
            
            self.key_matrix = try Matrix(T).init(allocator, header.att_dim, header.ctx_dim);
            self.key_vect = try allocator.alloc(T, header.att_dim);
            self.key = try Matrix(T).init(allocator, header.max_ctx_len, header.att_dim);

            self.score = try Matrix(T).init(allocator, header.max_seq_len, header.max_ctx_len); 

            self.value_matrix = try Matrix(T).init(allocator, header.out_dim, header.ctx_dim);
            self.value_vect = try allocator.alloc(T, header.out_dim);
            self.value = try Matrix(T).init(allocator, header.out_dim, header.max_ctx_len);
        }

        pub fn destroy(self: *Self, allocator: std.mem.Allocator) void {
            self.query_matrix.destroy(allocator);
            allocator.free(self.query_vect);
            self.query.destroy(allocator);

            self.key_matrix.destroy(allocator);
            allocator.free(self.key_vect);
            self.key.destroy(allocator);

            self.score.destroy(allocator);

            self.value_matrix.destroy(allocator);
            allocator.free(self.value_vect);
            self.value.destroy(allocator);
        }

        /// reads from the beginning of the file its header and weights. If you
        /// want more precise action take a look at `readHeader` and `readWeights`
        pub fn initFromFile(allocator: std.mem.Allocator, file: std.fs.File,
            out: Matrix(T)) !Self {
            file.seekTo(0);
            const reader = file.reader();

            var retval: Self = undefined;
            retval.out = out;
            try retval.readHeader(reader);
            try retval.allocateForHeader(allocator);
            try retval.readWeights(reader);

            return retval;
        }

        /// reads all the dimention from a file to structure

        /// memory for weights sould be preallocated
        pub fn readWeights(self: *Self, reader: anytype) !void {
            self.query_matrix.read(reader) catch |e| {
                std.log.err("can't read query matrix ({}x{}) {}\n",
                    .{self.query_matrix.height, self.query_matrix.width, T});
                return e;
            };
            mtx.readVector(T, reader, self.query_vect) catch |e| {
                std.log.err("can't read query vector ({}) {}\n",
                    .{self.query_vect.len, T});
                return e;
            };
            self.key_matrix.read(reader) catch |e| {
                std.log.err("can't read key matrix ({}x{}) {}\n",
                    .{self.key_matrix.height, self.key_matrix.width, T});
                return e;
            };
            mtx.readVector(T, reader, self.key_vect) catch |e| {
                std.log.err("can't read key vector ({}) {}\n",
                    .{self.key_vect.len, T});
                return e;
            };
            self.value_matrix.read(reader) catch |e| {
                std.log.err("can't read value matrix ({}x{}) {}\n",
                    .{self.value_matrix.height, self.value_matrix.width, T});
                return e;
            };
            mtx.readVector(T, reader, self.value_vect) catch |e| {
                std.log.err("can't read key vector ({}) {}\n",
                    .{self.value_vect.len, T});
                return e;
            };
        }

        // this version is tested it does not have division by the dymention
        // because it can be done internaly, by setting correctly the matrix.
        pub fn calculate(self: *Self, seq: Matrix(T), ctx: Matrix(T),
            comptime mask: enum{bidirectional, unidirectional}) !void {
            
            try mtx.affine(T, .{.mat = self.query_matrix, .input = seq,
                .vect = self.query_vect, .output = &self.query});
            try mtx.affine(T, .{.mat = self.key_matrix, .input = ctx,
                .vect = self.key_vect, .output = &self.key});
            try mtx.matprod(T, self.key, self.query, &self.score);

            self.score.scale(1/@sqrt(@as(f32, @floatFromInt(self.header.att_dim))));

            try mtx.affine2(T, .{.mat = ctx, .input = self.value_matrix,
                .vect = self.value_vect, .output = &self.value});

            if (mask == .unidirectional) {
                for (0..self.score.height) |i| {
                    for(self.score.row(i)[i+1..]) |*e| {
                        e.* = -std.math.inf(T);
                    }
                }
            }
            
            mtx.softmax(T, self.score);
            try mtx.matprod(T, self.value, self.score, &self.out);
        }

        pub fn writeWeights(self: Self, writer: anytype) !void {
            self.query_matrix.write(writer) catch |e| {
                std.log.err("can't write query matrix ({}x{}) {}\n",
                    .{self.query_matrix.height, self.query_matrix.width, T});
                return e;
            };
            mtx.writeVector(T, writer, self.query_vect) catch |e| {
                std.log.err("can't write query vector ({}) {}\n",
                    .{self.query_vect.len, T});
                return e;
            };
            self.key_matrix.write(writer) catch |e| {
                std.log.err("can't write key matrix ({}x{}) {}\n",
                    .{self.key_matrix.height, self.key_matrix.width, T});
                return e;
            };
            mtx.writeVector(T, writer, self.key_vect) catch |e| {
                std.log.err("can't write key vector ({}) {}\n",
                    .{self.key_vect.len, T});
                return e;
            };
            self.value_matrix.write(writer) catch |e| {
                std.log.err("can't write value matrix ({}x{}) {}\n",
                    .{self.value_matrix.height, self.value_matrix.width, T});
                return e;
            };
            mtx.writeVector(T, writer, self.value_vect) catch |e| {
                std.log.err("can't write key vector ({}) {}\n",
                    .{self.value_vect.len, T});
                return e;
            };
        }

        pub fn fillRandom(self: Self, rand: std.Random, k: T) void {
            self.query_matrix.fillRandom(rand, k);
            mtx.fillVecRandom(T, rand, self.query_vect, k);
            self.key_matrix.fillRandom(rand, k);
            mtx.fillVecRandom(T, rand, self.key_vect, k);
            self.value_matrix.fillRandom(rand, k);
            mtx.fillVecRandom(T, rand, self.value_vect, k);
        }

        const TestingData = struct {
            pub const header = AttentionHeader {
                .version = 0,
                .type_len = @sizeOf(T),
                .seq_dim = 2,
                .ctx_dim = 3,
                .out_dim = 4,
                .att_dim = 5,
                .max_ctx_len = 6,
                .max_seq_len = 7,
            };
            pub var cont1 = [_]T {
                0, 1,
                2, 3,
                4, 5,
                6, 7,
                8, 9,
            };
            pub var cont2 = [_]T {0, 1, 2, 3, 4};
            pub var cont3 = [_]T {
                0, 1, 2,
                3, 4, 5,
                6, 7, 8,
                9,10,11,
               12,13,14,
            };
            var cont4 = [_]T {-1, -2, -3, -4, -5};
            var cont5 = [_]T {
                9, 8, 7,
                6, 5, 4,
                3, 2, 1,
                0,-1,-2,
            };
            var cont6 = [_]T {2, 4, 6, 8};
        };

        test init {
            const allocator = std.testing.allocator;
            var out = try Matrix(T).init(allocator, TestingData.header.max_seq_len, TestingData.header.out_dim);
            var att = try Attention(T).init(allocator, TestingData.header, out);
            att.destroy(allocator);
            out.destroy(allocator);
        }
        
        test allocateForHeader {
            const allocator = std.testing.allocator;
            var out = try Matrix(T).init(allocator, TestingData.header.max_seq_len, TestingData.header.out_dim);
            var att: Attention(T) = undefined;
            att.header = TestingData.header;
            att.out = out;
            try att.allocateForHeader(allocator);
            att.destroy(allocator);
            out.destroy(allocator);
        }

        test writeWeights {
            const file = try std.fs.cwd().openFile("test_files/test_attention", .{.mode = .write_only});
            const writer = file.writer();

            try TestingData.header.write(writer);

            var att: Attention(T) = undefined;
            att.header = TestingData.header;

            att.query_matrix.capacity = TestingData.header.att_dim*TestingData.header.seq_dim;
            att.query_matrix.height = TestingData.header.att_dim;
            att.query_matrix.width = TestingData.header.seq_dim;
            att.query_matrix.ptr = &TestingData.cont1;

            att.query_vect = &TestingData.cont2;

            att.key_matrix.capacity = TestingData.header.att_dim*TestingData.header.ctx_dim;
            att.key_matrix.height = TestingData.header.att_dim;
            att.key_matrix.width = TestingData.header.ctx_dim;
            att.key_matrix.ptr = &TestingData.cont3;

            att.key_vect = &TestingData.cont4;

            att.value_matrix.capacity = TestingData.header.out_dim*TestingData.header.ctx_dim;
            att.value_matrix.height = TestingData.header.out_dim;
            att.value_matrix.width = TestingData.header.ctx_dim;
            att.value_matrix.ptr = &TestingData.cont5;

            att.value_vect = &TestingData.cont6;

            try att.writeWeights(writer);
            try file.setEndPos(try file.getPos());
            file.close();
        }

        test initFromFile {
            const allocator = std.testing.allocator;
            const file = try std.fs.cwd().openFile("test_files/test_attention", .{.mode = .read_only});
            const reader = file.reader();

            var header: AttentionHeader = undefined;
            try header.read(reader);

            var att: Attention(T) = undefined;
            att.header = header;
            try att.allocateForHeader(allocator);
            try att.readWeights(reader);

            try std.testing.expectEqualSlices(T, &TestingData.cont1, att.query_matrix.toSlice());

            try std.testing.expectEqualSlices(T, &TestingData.cont2, att.query_vect);

            try std.testing.expectEqualSlices(T, &TestingData.cont3, att.key_matrix.toSlice());

            try std.testing.expectEqualSlices(T, &TestingData.cont4, att.key_vect);

            try std.testing.expectEqualSlices(T, &TestingData.cont5, att.value_matrix.toSlice());

            try std.testing.expectEqualSlices(T, &TestingData.cont6, att.value_vect);

            att.destroy(allocator);
            file.close();
        }

        test readWeights {
            const allocator = std.testing.allocator;
            const file = try std.fs.cwd().openFile("test_files/test_attention", .{.mode = .read_only});
            const reader = file.reader();

            var header: AttentionHeader = undefined;
            try header.read(reader);

            var att: Attention(T) = undefined;
            att.header = header;
            try att.allocateForHeader(allocator);
            try att.readWeights(reader);

            try std.testing.expectEqualSlices(T, &TestingData.cont1, att.query_matrix.toSlice());

            try std.testing.expectEqualSlices(T, &TestingData.cont2, att.query_vect);

            try std.testing.expectEqualSlices(T, &TestingData.cont3, att.key_matrix.toSlice());

            try std.testing.expectEqualSlices(T, &TestingData.cont4, att.key_vect);

            try std.testing.expectEqualSlices(T, &TestingData.cont5, att.value_matrix.toSlice());

            try std.testing.expectEqualSlices(T, &TestingData.cont6, att.value_vect);

            att.destroy(allocator);
            file.close();
        }

        test calculate {
            var seq_data = [_]T {
                1, 1,
                1,-1,
                0,-1,
            };
            const seq = Matrix(T) {
                .capacity = 6,
                .height = 3,
                .width = 2,
                .ptr = &seq_data,
            };

            var ctx_data = [_]T {
                1, 2, 3,
                3, 2, 1,
            };

            const ctx = Matrix(T) {
                .capacity = 6,
                .height = 2,
                .width = 3,
                .ptr = &ctx_data,
            };

            const allocator = std.testing.allocator;
            const file = try std.fs.cwd().openFile("test_files/test_attention", .{.mode = .read_only});
            const reader = file.reader();

            var att: Attention(T) = undefined;
            var header: AttentionHeader = undefined;
            try header.read(reader);
            att.header = header;
            try att.allocateForHeader(allocator);
            defer att.destroy(allocator);
            try att.readWeights(reader);

            att.out = try Matrix(T).init(allocator, att.header.max_seq_len, att.header.out_dim);
            defer att.out.destroy(allocator);

            try att.calculate(seq, ctx, .bidirectional);

            try std.testing.expectEqualSlices(T, &[_]T { // hand-calculated
                1,  6, 11, 16, 21,
                -1, 0,  1,  2,  3,
                -1,-2, -3, -4, -5,
            }, att.query.toSlice()[0..seq.height*att.header.att_dim]);

            try std.testing.expectEqualSlices(T, &[_]T {
                7, 24, 41, 58, 75,
                3, 20, 37, 54, 71,
            }, att.key.toSlice()[0..ctx.height*att.header.att_dim]);

            // @Todo: add row!!!
            try std.testing.expectEqualSlices(T, &[_]T {
                48, 52,
                32, 36,
                16, 20,
                0,  4,
            }, att.value.toSlice());

            // Actually it works fine. You should get something close to this
            try std.testing.expect(std.math.isNan(att.score.toSlice()[0])); // you cannot compare nans!
            try std.testing.expect(std.math.isNan(att.score.toSlice()[1]));
            try std.testing.expect(@abs(att.score.toSlice()[2]-0.9998695346) < 0.00001 or
                std.math.isNan(att.score.toSlice()[2]));
            try std.testing.expect(@abs(att.score.toSlice()[3]-0.0001304654165) < 0.00000001 or
                std.math.isNan(att.score.toSlice()[3]));
            try std.testing.expect(@abs(att.score.toSlice()[4]-0) < 0.00000001 or
                std.math.isNan(att.score.toSlice()[4]));
            try std.testing.expect(@abs(att.score.toSlice()[5]-1) < 0.00000001 or
                std.math.isNan(att.score.toSlice()[5]));

            try std.testing.expect(std.math.isNan(att.out.toSlice()[0]));
            try std.testing.expect(std.math.isNan(att.out.toSlice()[1]));
            try std.testing.expect(std.math.isNan(att.out.toSlice()[2]));
            try std.testing.expect(std.math.isNan(att.out.toSlice()[3]));

            try std.testing.expect(@abs(att.out.toSlice()[4]-48.00052186) < 0.0000001 or
                std.math.isNan(att.out.toSlice()[4]));
            try std.testing.expect(@abs(att.out.toSlice()[5]-32.00052186) < 0.0000001 or
                std.math.isNan(att.out.toSlice()[4]));
            try std.testing.expect(@abs(att.out.toSlice()[6]-16.00052186) < 0.0000001 or
                std.math.isNan(att.out.toSlice()[4]));
            try std.testing.expect(@abs(att.out.toSlice()[7]-0.000521861666) < 0.0000001 or
                std.math.isNan(att.out.toSlice()[4]));
            try std.testing.expect(@abs(att.out.toSlice()[8]-52) < 0.0000001 or
                std.math.isNan(att.out.toSlice()[4]));
            try std.testing.expect(@abs(att.out.toSlice()[9]-36) < 0.0000001 or
                std.math.isNan(att.out.toSlice()[4]));
            try std.testing.expect(@abs(att.out.toSlice()[10]-20) < 0.0000001 or
                std.math.isNan(att.out.toSlice()[4]));
            try std.testing.expect(@abs(att.out.toSlice()[11]-4) < 0.0000001 or
                std.math.isNan(att.out.toSlice()[4]));

            file.close();
        }
    };
}

comptime {
    _ = Attention(f32);
}
