/// seq2seq transformer
const std = @import("std");
const parsing = @import("parsing.zig");

fn fillVecRandom(comptime T: type, rand: std.Random, vec: []T, k: T) void {
    for (vec) |*v| {
        const f = rand.float(T);
        v.* = k*(2.0*f - 1.0);
    }
}

pub fn Matrix(comptime T: type) type {
    return struct {
        ptr: [*]T,
        height: usize,
        width: usize,

        const Self = @This();
        pub fn init(allocator: std.mem.Allocator, height: usize, width: usize) !Self {
            var retval: Self = undefined;
            retval.ptr = (try allocator.alloc(T, height*width)).ptr;
            retval.height = height;
            retval.width = width;
            return retval;
        }

        /// line-column convention
        pub inline fn at(self: Self, i: usize, j: usize) ?T {
            if (i > self.height or j > self.width) {
                return null;
            }
            
            return self.ptr[i*self.width+j];
        }

        /// line-column convetion
        pub inline fn at_nocheck(self: Self, i: usize, j: usize) T {
            return self.ptr[i*self.width+j];
        }

        pub inline fn row(self: Self, i: usize) []T {
            const start = i*self.width;
            return self.ptr[start..start+self.width];
        }

        pub inline fn submatrix(self: Self, start: u32, end: u32) ?Self {
            if (start < 0 or start >= end or end > self.height) {
                std.debug.print("wrong indices for sub matrix\n", .{});
                return null;
            }

            return Matrix(T){
                .height = end - start,
                .width = self.width,
                .ptr = self.ptr + self.width*start,
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
    };
}

const AlgebraicError = error {
    IncompatibleObjects,
};

pub fn matprint(comptime T: type, mat: Matrix(T)) void {
    for (0..mat.height) |i| {
        for (0..mat.width) |j| {
            std.debug.print("{}\t", .{mat.at(i,j).?});
        }
        std.debug.print("\n", .{});
    }
}

/// computes `(mat1 x mat2^t)^t` because the second argement is trited as an
/// array of vectors a and vetors are lines in this library. The reason is
/// memory locality. mat1: MxN, mat2: LxN, out LxM
pub fn matprod(comptime T: type, mat1: Matrix(T), mat2: Matrix(T), out: Matrix(T)) !void {
    if (mat1.width != mat2.width) {
        std.debug.print("Wrong dimensions in product mat1 and mat2\n", .{});
        return error.IncompatibleObjects; 
    }

    if (out.width < mat1.height) {
        std.debug.print("Wrong width in out\n", .{});
        return error.IncompatibleObjects; 
    }

    if (out.height < mat2.height) {
        std.debug.print("Wrong height in out\n", .{});
        return error.IncompatibleObjects; 
    }

    for (0..mat2.height) |i| {
        const vect = mat2.row(i);
        var out_vect = out.row(i);
        for (0..mat1.height) |j| {
            out_vect[j] = 0;
            for (0..vect.len) |k| {
                out_vect[j] += mat1.at_nocheck(j, k)*vect[k];
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

/// adds to each row of mat the vect
/// mat: NxM, vec: M
pub fn addMatRow(comptime T: type, mat: Matrix(T), vec: []T) !void {
    if (mat.width != vec.len) {
        std.debug.print("Incoherent mat and vect sizes!\n", .{});
        return error.IncompatibleObjects;
    }

    // dumb implementation
    for (0..mat.height) |i| {
        for (0..mat.width) |j| {
            mat.ptr[i*mat.width+j] += vec[j];
        }
    }
}

/// adds to each row of mat the vect
/// mat: NxM, vec: M
pub fn addMatCol(comptime T: type, mat: Matrix(T), vec: []T) !void {
    if (mat.height != vec.len) {
        std.debug.print("Incoherent mat and vect sizes!\n", .{});
        return error.IncompatibleObjects;
    }

    // dumb implementation
    for (0..mat.height) |i| {
        for (0..mat.width) |j| {
            mat.ptr[i*mat.width+j] += vec[i];
        }
    }
}

/// memory for the output should be preallocated and the sizes should be coherent
/// mat: NxM, input: LxM, vect: N, output: LxN
pub fn affine(comptime T: type, payload: struct {mat: Matrix(T), input: Matrix(T),
    vect: []T, output: Matrix(T)}) !void {
    // the sympliest algo
    // const in_dim: u32 = payload.mat.width;
    // const out_dim: u32 = payload.mat.height;

    try matprod(T, payload.mat, payload.input, payload.output);
    try addMatRow(T, payload.output, payload.vect);
}


/// memory for the output should be preallocated and the sizes should be coherent
/// mat: NxM, input: LxM, vect: N, output: LxN
pub fn affine2(comptime T: type, payload: struct {mat: Matrix(T), input: Matrix(T),
    vect: []T, output: Matrix(T)}) !void {
    // the sympliest algo
    // const in_dim: u32 = payload.mat.width;
    // const out_dim: u32 = payload.mat.height;

    try matprod(T, payload.mat, payload.input, payload.output);
    try addMatCol(T, payload.output, payload.vect);
}

pub fn matscale(comptime T: type, k: T, mat: Matrix(T)) void {
    for(0..mat.height) |i| {
        const vec = mat.row(i);
        for(vec) |*v| {
            v.* *= k;
        }
    }
}

/// funciton overwrites content of mat and does soft max on every row of mat
pub fn softmax(comptime T: type, mat: Matrix(T)) void {
    for (0..mat.height) |i| {
        const vect = mat.row(i);
        var sum: T = 0;

        for (vect) |*v| {
            v.* = @exp(v.*);
            // std.debug.print("exp: {}\n", .{v.*});
            sum += v.*;
        }
        // std.debug.print("sum: {}\n", .{sum});
        for (vect) |*v| {
            v.* = v.*/sum;
        }
    }

}

fn readMatrix(comptime T: type, reader: anytype, mat: Matrix(T)) !void {
    const ptr: [*]u8 =  @ptrCast(mat.ptr);
    const size: usize = mat.height*mat.width*@sizeOf(T);
    const buffer: []u8 = ptr[0..size];
    _ = try reader.read(buffer);           
}

fn readVector(comptime T: type, reader: anytype, vec: []T) !void {
    const ptr: [*]u8 = @ptrCast(vec.ptr);
    const size: usize = vec.len*@sizeOf(T);
    _ = try reader.read(ptr[0..size]);
}

fn writeMatrix(comptime T: type, writer: anytype, mat: Matrix(T)) !void {
    const ptr: [*]u8 = @ptrCast(mat.ptr);
    const size: usize = mat.height*mat.width*@sizeOf(T);
    _ = try writer.write(ptr[0..size]);
}

fn writeVector(comptime T: type, writer: anytype, vec: []T) !void {
    const ptr: [*]u8 = @ptrCast(vec.ptr);
    const size: usize = vec.len*@sizeOf(T);
    _ = try writer.write(ptr[0..size]);
}

pub fn Attention(comptime T: type) type {
    return struct {
        seq_dim: u32,
        ctx_dim: u32,
        attn_dim: u32,
        out_dim: u32,

        max_seq_len: u32,
        max_ctx_len: u32,

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
        pub fn init(allocator: std.mem.Allocator, hyper_param: struct {
            seq_dim: u32, ctx_dim: u32, attn_dim: u32, out_dim: u32,
            max_seq_len: u32, max_ctx_len: u32}, out: Matrix(T)) !Self {

            if (out.height != hyper_param.max_seq_len or out.width != hyper_param.out_dim) {
                std.debug.print("Output matrix is of a wrong size\n", .{});
                return error.DataConflict;
            }

            var retval: Self = undefined;
            retval.seq_dim = hyper_param.seq_dim;
            retval.ctx_dim = hyper_param.ctx_dim;
            retval.attn_dim = hyper_param.attn_dim;
            retval.out_dim = hyper_param.out_dim;
            retval.max_seq_len =hyper_param.max_seq_len;
            retval.max_ctx_len =hyper_param.max_ctx_len;

            retval.query_matrix = try Matrix(T).init(allocator, hyper_param.attn_dim, hyper_param.seq_dim);
            retval.query_vect = try allocator.alloc(T, hyper_param.attn_dim);
            retval.query = try Matrix(T).init(allocator, hyper_param.max_seq_len, hyper_param.attn_dim);
            
            retval.key_matrix = try Matrix(T).init(allocator, hyper_param.attn_dim, hyper_param.ctx_dim);
            retval.key_vect = try allocator.alloc(T, hyper_param.attn_dim);
            retval.key = try Matrix(T).init(allocator, hyper_param.max_ctx_len, hyper_param.attn_dim);

            retval.score = try Matrix(T).init(allocator, hyper_param.max_seq_len, hyper_param.max_ctx_len); 

            retval.value_matrix = try Matrix(T).init(allocator, hyper_param.out_dim, hyper_param.ctx_dim);
            retval.value_vect = try allocator.alloc(T, hyper_param.out_dim);
            retval.value = try Matrix(T).init(allocator, hyper_param.out_dim, hyper_param.max_ctx_len);


            retval.out = out;

            return retval;
        }

        pub fn initFromFile(allocator: std.mem.Allocator, file: std.fs.File,
            out: Matrix(T)) !Self {
            // file format for attention
            var reader = file.reader();

            const type_size = reader.readInt(u32, .little) catch |e| {
                std.debug.print("Can't read type_size from file\n", .{});
                return e;
            };

            if (type_size != @sizeOf(T)) {
                std.debug.print("The type_size in a attention file is wrong\n", .{});
                return error.DataConflict;
            }

            const seq_dim = reader.readInt(u32, .little) catch |e| {
                std.debug.print("Can't read seq_dim from file\n", .{});
                return e;
            };
            const ctx_dim = reader.readInt(u32, .little) catch |e| {
                std.debug.print("Can't read seq_dim from file\n", .{});
                return e;
            };
            const attn_dim = reader.readInt(u32, .little) catch |e| {
                std.debug.print("Can't read seq_dim from file\n", .{});
                return e;
            };
            const out_dim = reader.readInt(u32, .little) catch |e| {
                std.debug.print("Can't read seq_dim from file\n", .{});
                return e;
            };
            const max_seq_len = reader.readInt(u32, .little) catch |e| {
                std.debug.print("Can't read seq_dim from file\n", .{});
                return e;
            };
            const max_ctx_len = reader.readInt(u32, .little) catch |e| {
                std.debug.print("Can't read seq_dim from file\n", .{});
                return e;
            };

            const retval = try init(allocator, .{.seq_dim = seq_dim, .ctx_dim = ctx_dim,
                .attn_dim = attn_dim, .out_dim = out_dim, .max_seq_len =
                max_seq_len, .max_ctx_len = max_ctx_len}, out);
           
            readMatrix(T, reader, retval.query_matrix) catch |e| {
                std.debug.print("can't read query matrix ({}x{}) {}\n",
                    .{retval.query_matrix.height, retval.query_matrix.width, T});
                return e;
            };

            readVector(T, reader, retval.query_vect) catch |e| {
                std.debug.print("can't read query vector ({}) {}\n",
                    .{retval.query_vect.len, T});
                return e;
            };

            readMatrix(T, reader, retval.key_matrix) catch |e| {
                std.debug.print("can't read key matrix ({}x{}) {}\n",
                    .{retval.key_matrix.height, retval.key_matrix.width, T});
                return e;
            };

            readVector(T, reader, retval.key_vect) catch |e| {
                std.debug.print("can't read key vector ({}) {}\n",
                    .{retval.key_vect.len, T});
                return e;
            };

            readMatrix(T, reader, retval.value_matrix) catch |e| {
                std.debug.print("can't read value matrix ({}x{}) {}\n",
                    .{retval.value_matrix.height, retval.value_matrix.width, T});
                return e;
            };

            readVector(T, reader, retval.value_vect) catch |e| {
                std.debug.print("can't read key vector ({}) {}\n",
                    .{retval.value_vect.len, T});
                return e;
            };

            return retval;
        }

        // this vesion is tested it does not have division by the dymention
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

        pub fn writeToFile(self: Self, file: std.fs.File) !void {
            try file.seekTo(0);
            var writer = file.writer();
            
            writer.writeInt(u32, @sizeOf(T), .little) catch |e| {
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
            writer.writeInt(u32, self.attn_dim, .little) catch |e| {
                std.debug.print("Can't write attn_dim from file\n", .{});
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

            writeMatrix(T, writer, self.query_matrix) catch |e| {
                std.debug.print("can't write query matrix ({}x{}) {}\n",
                    .{self.query_matrix.height, self.query_matrix.width, T});
                return e;
            };

            writeVector(T, writer, self.query_vect) catch |e| {
                std.debug.print("can't write query vector ({}) {}\n",
                    .{self.query_vect.len, T});
                return e;
            };

            writeMatrix(T, writer, self.key_matrix) catch |e| {
                std.debug.print("can't write key matrix ({}x{}) {}\n",
                    .{self.key_matrix.height, self.key_matrix.width, T});
                return e;
            };

            writeVector(T, writer, self.key_vect) catch |e| {
                std.debug.print("can't write key vector ({}) {}\n",
                    .{self.key_vect.len, T});
                return e;
            };

            writeMatrix(T, writer, self.value_matrix) catch |e| {
                std.debug.print("can't write value matrix ({}x{}) {}\n",
                    .{self.value_matrix.height, self.value_matrix.width, T});
                return e;
            };

            writeVector(T, writer, self.value_vect) catch |e| {
                std.debug.print("can't write key vector ({}) {}\n",
                    .{self.value_vect.len, T});
                return e;
            };

            try file.setEndPos(try file.getPos());
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

pub fn MHAttention(comptime T: type) type {
    return struct {
        heads: u32,

        seq_dim: u32,
        ctx_dim: u32,
        att_dim: u32,
        mid_dim: u32,
        out_dim: u32,

        att_results: Matrix(T),
        attention: []Attention(T),
        comb_matrix: Matrix(T),
        comb_vect: []T,


        out: Matrix(T),

        const Self = @This();
        pub fn init(allocator: std.mem.Allocator, hyper_param: struct {
            seq_dim: u32, ctx_dim: u32, att_dim:u32, mid_dim: u32, heads: u32,
            out_dim: u32, max_seq_len: u32, max_ctx_len: u32}, out: Matrix(T)) Self {    
            var retval: Self = undefined;

            // maybe change the number of parameters passed?
            if (out.height != hyper_param.max_seq_len or out.width !=
                hyper_param.out_dim) {
                std.debug.print("Wrong dimensions for output matrix for MHAttension\n", .{});
                return error.DataConflict;
            }

            retval.heads = hyper_param.heads;
            retval.seq_dim = hyper_param.seq_dim;
            retval.ctx_dim = hyper_param.ctx_dim;
            retval.mid_dim = hyper_param.mid_dim;
            retval.out_dim = hyper_param.out_dim;

            retval.attention = try allocator.alloc(T, hyper_param.heads);
            retval.att_results = try Matrix(T).init(allocator, hyper_param.max_seq_len
                * hyper_param.heads, hyper_param.mid_dim);
            for (0..hyper_param.heads) |i| {
                retval.attention[i] = try Attention(T).init(allocator, .{
                    .seq_dim = hyper_param.seq_dim,
                    .ctx_dim = hyper_param.ctx_dim,
                    .attn_dim = hyper_param.att_dim,
                    .out_dim = hyper_param.mid_dim,
                    .max_seq_len = hyper_param.max_seq_len,
                    .max_ctx_len = hyper_param.max_ctx_len},
                    retval.att_results.submatrix(i*hyper_param.max_seq_len,
                        (i+1)*hyper_param.max_seq_len),
                );
            }

            retval.comb_matrix = try Matrix(T).init(allocator, hyper_param.out_dim,
                hyper_param.mid_dim*hyper_param.heads);
            retval.comb_vect = try allocator.alloc(T, hyper_param.out_dim);
            retval.out = out;

            return retval;
        }
    };
}


fn EncodeLayer(T: type) type {
    return struct {
        enc_attention_matrix: Matrix(T),
        enc_layer_norm_gamma: [2][]T,
        enc_layer_norm_beta: [2][]T,
        enc_mlp1_matrix: Matrix(T),
        enc_mlp2_matrix: Matrix(T),
        enc_mlp1_vect: []T,
        enc_mlp2_vect: []T,

        const Self = @This();
        pub fn init(allocator: std.mem.Allocator, emb_dim: u32, mpl_dim: u32) Self {
            var retval: Self = undefined;

            retval.enc_attention_matrix = Matrix(T).init(allocator, emb_dim, mpl_dim);

            return retval;
        }
    };
}

pub fn EDTransformer(comptime T: type) type {
    return struct {

        // hyper param
        dictionery_len: u32,
        max_length: u32,
        encode_layers: u32,
        decode_layers: u32,
        H: u32, // ???
        emb_dim: u32,
        mlp_dim: u32,

        embedding_matrix: Matrix(T),
        position_matrix: Matrix(T),
        // encoder param
        encode: []EncodeLayer(T),

        //decoder param
        dec_attention_matrix: []Matrix(T),
        cross_attention_matrix: []Matrix(T),
        unembedding_matrix: Matrix(T),

        allocator: std.mem.Allocator,

        const Self = @This();
        pub fn init(allocator: std.mem.Allocator, comptime hyper_param: struct {
            dict_len: u32,
            max_length: u32,
            encode_layers: u32,
            decode_layers: u32,
            H: u32,
            emb_dim: u32,
            mlp_dim: u32}) Self {
            var retval: Self = undefined;

            retval.allocator = allocator;
            retval.dict_len = hyper_param.dictionery_len;
            retval.max_length = hyper_param.max_length;
            retval.encode_layers = hyper_param.encode_layers;
            retval.decode_layers = hyper_param.decoder_layers;
            retval.H = hyper_param.H;
            retval.emb_dim = hyper_param.emb_dim;
            retval.mlp_dim = hyper_param.mlp_dim;

            retval.embedding_matrix = try Matrix(T).init(allocator, retval.dict_len,
                retval.emb_dim);

            retval.position_matrix = try Matrix(T).init(allocator, retval.max_length,
                retval.emb_dim);

            retval.enc_attention_matrix = try allocator.alloc(Matrix(T), retval.encode_layers);
            retval.enc_layer_norm_beta = try allocator.alloc([2][]T, retval.encode_layers);
            retval.enc_layer_norm_gamma = try allocator.alloc([2][]T, retval.encode_layers);

            for (0..retval.encode_layers) |l| {
                _ = l;
                
            }
            return retval;
        }

    };
}
