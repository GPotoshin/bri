const std = @import("std");
const mtx = @import("matrix.zig");
const Matrix = mtx.Matrix;

const AttentionHeader = struct {
    version: u32,

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
            std.debug.print("Can't write version to a file\n", .{});
            return e;
        };
        writer.writeInt(u32, self.type_len, .little) catch |e| {
            std.debug.print("Can't write type_size to a file\n", .{});
            return e;
        };
        writer.writeInt(u32, self.seq_dim, .little) catch |e| {
            std.debug.print("Can't write seq_dim from file\n", .{});
            return e;
        };
        writer.writeInt(u32, self.ctx_dim, .little) catch |e| {
            std.debug.print("Can't write ctx_dim from file\n", .{});
            return e;
        };
        writer.writeInt(u32, self.att_dim, .little) catch |e| {
            std.debug.print("Can't write att_dim from file\n", .{});
            return e;
        };
        writer.writeInt(u32, self.out_dim, .little) catch |e| {
            std.debug.print("Can't write out_dim from file\n", .{});
            return e;
        };
        writer.writeInt(u32, self.max_seq_len, .little) catch |e| {
            std.debug.print("Can't write max_seq_len from file\n", .{});
            return e;
        };
        writer.writeInt(u32, self.max_ctx_len, .little) catch |e| {
            std.debug.print("Can't write seq_dim from file\n", .{});
            return e;
        };
    }

    pub fn read(self: *Self, reader: anytype) !void {
        self.version = reader.readInt(u32, .little) catch |e| {
            std.debug.print("Can't read version\n", .{});
            return e;
        };
        self.type_len = reader.readInt(u32, .little) catch |e| {
            std.debug.print("Can't read type\n", .{});
            return e;
        };
        self.seq_dim = reader.readInt(u32, .little) catch |e| {
            std.debug.print("Can't read seq_dim\n", .{});
            return e;
        };
        self.ctx_dim = reader.readInt(u32, .little) catch |e| {
            std.debug.print("Can't read ctx_dim\n", .{});
            return e;
        };
        self.att_dim = reader.readInt(u32, .little) catch |e| {
            std.debug.print("Can't read att_dim\n", .{});
            return e;
        };
        self.out_dim = reader.readInt(u32, .little) catch |e| {
            std.debug.print("Can't read out_dim\n", .{});
            return e;
        };
        self.max_seq_len = reader.readInt(u32, .little) catch |e| {
            std.debug.print("Can't read max_seq_len\n", .{});
            return e;
        };
        self.max_ctx_len = reader.readInt(u32, .little) catch |e| {
            std.debug.print("Can't read max_ctx_len\n", .{});
            return e;
        };
    }
};

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
                std.debug.print("Output matrix is of a wrong size\n", .{});
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
            const header = AttentionHeader.read(reader);
            self.query_matrix = try Matrix(T).init(allocator, self.att_dim, self.seq_dim);
            self.query_vect = try allocator.alloc(T, self.att_dim);
            self.query = try Matrix(T).init(allocator, self.max_seq_len, self.att_dim);
            
            self.key_matrix = try Matrix(T).init(allocator, self.att_dim, self.ctx_dim);
            self.key_vect = try allocator.alloc(T, self.att_dim);
            self.key = try Matrix(T).init(allocator, self.max_ctx_len, self.att_dim);

            self.score = try Matrix(T).init(allocator, self.max_seq_len, self.max_ctx_len); 

            self.value_matrix = try Matrix(T).init(allocator, self.out_dim, self.ctx_dim);
            self.value_vect = try allocator.alloc(T, self.out_dim);
            self.value = try Matrix(T).init(allocator, self.out_dim, self.max_ctx_len);
        }

        pub fn free(self: *Self, allocator: std.mem.Allocator) !void {

        }

        /// reads from the beginning of the file its header and weights. If you
        /// want more precise action take a look at `readHeader` and `readWeights`
        pub fn initFromFile(allocator: std.mem.Allocator, file: std.fs.File,
            out: Matrix(T)) !Self {
            file.seekTo(0);
            var reader = file.reader();

            var retval: Self = undefined;
            try retval.readHeader(reader);
            try retval.allocateForHeader(allocator);
            try retval.readWeights(reader);

            return retval;
        }

        /// reads all the dimention from a file to structure

        /// memory for weights sould be preallocated
        pub fn readWeights(self: *Self, reader: anytype) !void {
            self.query_matrix.read(reader) catch |e| {
                std.debug.print("can't read query matrix ({}x{}) {}\n",
                    .{self.query_matrix.height, self.query_matrix.width, T});
                return e;
            };
            readVector(T, reader, self.query_vect) catch |e| {
                std.debug.print("can't read query vector ({}) {}\n",
                    .{self.query_vect.len, T});
                return e;
            };
            self.key_matrix.read(reader) catch |e| {
                std.debug.print("can't read key matrix ({}x{}) {}\n",
                    .{self.key_matrix.height, self.key_matrix.width, T});
                return e;
            };
            readVector(T, reader, self.key_vect) catch |e| {
                std.debug.print("can't read key vector ({}) {}\n",
                    .{self.key_vect.len, T});
                return e;
            };
            self.value_matrix.read(reader) catch |e| {
                std.debug.print("can't read value matrix ({}x{}) {}\n",
                    .{self.value_matrix.height, self.value_matrix.width, T});
                return e;
            };
            readVector(T, reader, self.value_vect) catch |e| {
                std.debug.print("can't read key vector ({}) {}\n",
                    .{self.value_vect.len, T});
                return e;
            };
        }

        // this version is tested it does not have division by the dymention
        // because it can be done internaly, by setting correctly the matrix.
        pub fn calculate(self: Self, seq: Matrix(T), ctx: Matrix(T),
            comptime mask: enum{bidirectional, unidirectional}) !Matrix(T) {
            
            try affine(T, .{.mat = self.query_matrix, .input = seq,
                .vect = self.query_vect, .output = self.query});
            try affine(T, .{.mat = self.key_matrix, .input = ctx,
                .vect = self.key_vect, .output = self.key});
            try matprod(T, self.key, self.query, self.score);

            try affine2(T, .{.mat = ctx, .input = self.value_matrix,
                .vect = self.value_vect, .output = self.value});

            if (mask == .unidirectional) {
                for (0..self.score.height) |i| {
                    for(self.score.row(i)[i+1..]) |*e| {
                        e.* = -std.math.inf(T);
                    }
                }
            }
            
            softmax(T, self.score);
            try matprod(T, self.value, self.score, self.out);

            return self.out;
        }

        pub fn writeWeights(self: Self, writer: anytype) !void {
            self.query_matrix.write(writer) catch |e| {
                std.debug.print("can't write query matrix ({}x{}) {}\n",
                    .{self.query_matrix.height, self.query_matrix.width, T});
                return e;
            };
            writeVector(T, writer, self.query_vect) catch |e| {
                std.debug.print("can't write query vector ({}) {}\n",
                    .{self.query_vect.len, T});
                return e;
            };
            self.key_matrix.write(writer) catch |e| {
                std.debug.print("can't write key matrix ({}x{}) {}\n",
                    .{self.key_matrix.height, self.key_matrix.width, T});
                return e;
            };
            writeVector(T, writer, self.key_vect) catch |e| {
                std.debug.print("can't write key vector ({}) {}\n",
                    .{self.key_vect.len, T});
                return e;
            };
            self.value_matrix.write(writer) catch |e| {
                std.debug.print("can't write value matrix ({}x{}) {}\n",
                    .{self.value_matrix.height, self.value_matrix.width, T});
                return e;
            };
            writeVector(T, writer, self.value_vect) catch |e| {
                std.debug.print("can't write key vector ({}) {}\n",
                    .{self.value_vect.len, T});
                return e;
            };
        }

        pub fn fillRandom(self: Self, rand: std.Random, k: T) void {
            self.query_matrix.fillRandom(rand, k);
            fillVecRandom(T, rand, self.query_vect, k);
            self.key_matrix.fillRandom(rand, k);
            fillVecRandom(T, rand, self.key_vect, k);
            self.value_matrix.fillRandom(rand, k);
            fillVecRandom(T, rand, self.value_vect, k);
        }
    };
}
