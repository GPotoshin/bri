/// seq2seq transformer
const std = @import("std");

const float_t: type = f16;

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

    if (out.width != mat1.height) {
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

pub fn Attention(comptime T: type) type {
    return struct {
        seq_dim: u32,
        ctx_dim: u32,
        attn_dim: u32,
        out_dim: u32,

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
            seq_len: u32, ctx_len: u32}) !Self {

            var retval: Self = undefined;
            retval.seq_dim = hyper_param.seq_dim;
            retval.ctx_dim = hyper_param.ctx_dim;
            retval.attn_dim = hyper_param.attn_dim;
            retval.out_dim = hyper_param.out_dim;

            retval.query_matrix = try Matrix(T).init(allocator, hyper_param.attn_dim, hyper_param.seq_dim);
            retval.query_vect = try allocator.alloc(T, hyper_param.attn_dim);
            retval.query = try Matrix(T).init(allocator, hyper_param.seq_len, hyper_param.attn_dim);
            
            retval.key_matrix = try Matrix(T).init(allocator, hyper_param.attn_dim, hyper_param.ctx_dim);
            retval.key_vect = try allocator.alloc(T, hyper_param.attn_dim);
            retval.key = try Matrix(T).init(allocator, hyper_param.ctx_len, hyper_param.attn_dim);

            retval.value_matrix = try Matrix(T).init(allocator, hyper_param.out_dim, hyper_param.ctx_dim);
            retval.value_vect = try allocator.alloc(T, hyper_param.out_dim);
            retval.value = try Matrix(T).init(allocator, hyper_param.out_dim, hyper_param.ctx_len);

            retval.out = try Matrix(T).init(allocator, hyper_param.seq_len, hyper_param.out_dim);

            return retval;
        }

        pub fn calulate(self: Self, seq: Matrix(T), context: Matrix(T)) !Matrix(T) {
            
            // This can be parallelized
            try affine(T, .{.mat = self.query_matrix, .input = seq,
                .vect = self.query_vect, .output = self.query});
            std.debug.print("got query\n", .{});

            try affine(T, .{.mat = self.key_matrix, .input = context,
                .vect = self.key_vect, .output = self.key});
            std.debug.print("got key\n", .{});

            try affine2(T, .{.mat = context, .input = self.value_matrix,
                .vect = self.value_vect, .output = self.value});
            std.debug.print("got value\n", .{});

            try matprod(T, self.key, self.query, self.score);
            std.debug.print("got score\n", .{});
            
            softmax(T, self.score);
            std.debug.print("softmax of score\n", .{});

            try matprod(T, self.value, self.score, self.out);
            std.debug.print("got out\n", .{});

            return self.out;
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

test "matmult" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var seqv = [_]f16{
        1, 2, 3,
        2, 1, 3,
        4, 3, 1,
        4, 4, 1,
        5, 3, 5,
        2, 4, 3,
        1, 1, 1,
    };
    const seq = Matrix(f16){
        .height = 7,
        .width = 3,
        .ptr = &seqv,
    };

    var ctxv = [_]f16{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 1,
        2, 3,
        4, 5,
        6, 7,
    };
    const ctx = Matrix(f16){
        .height = 8,
        .width = 3,
        .ptr = &ctxv,
    };


    const att = try Attention(f16).init(allocator, .{
        .seq_dim = 3, .ctx_dim = 2, .attn_dim = 4, .out_dim = 5, .seq_len = 7, .ctx_len = 8});
    
    att.query_matrix.ptr[0] = 0;
    att.query_matrix.ptr[1] = 1;
    att.query_matrix.ptr[2] = 2;
    att.query_matrix.ptr[3] = 3;
    att.query_matrix.ptr[4] = 4;
    att.query_matrix.ptr[5] = 5;
    att.query_matrix.ptr[6] = 6;
    att.query_matrix.ptr[7] = 7;
    att.query_matrix.ptr[8] = 8;
    att.query_matrix.ptr[9] = 9;
    att.query_matrix.ptr[10] = 10;
    att.query_matrix.ptr[11] = 11;

    att.query_vect[0] = 3;
    att.query_vect[1] = 1;
    att.query_vect[2] = 2;
    att.query_vect[3] = 4;

    att.key_matrix.ptr[0] = 1; 
    att.key_matrix.ptr[1] = 2; 
    att.key_matrix.ptr[2] = 3; 
    att.key_matrix.ptr[3] = 4; 
    att.key_matrix.ptr[4] = 5; 
    att.key_matrix.ptr[5] = 6; 
    att.key_matrix.ptr[6] = 7; 
    att.key_matrix.ptr[7] = 8; 

    att.key_vect.ptr[0] = 4;
    att.key_vect.ptr[1] = 9;
    att.key_vect.ptr[2] = 6;
    att.key_vect.ptr[3] = 1;

    att.value_matrix.ptr[0] = -9;
    att.value_matrix.ptr[1] = -8;
    att.value_matrix.ptr[2] = -7;
    att.value_matrix.ptr[3] = -6;
    att.value_matrix.ptr[4] = -5;
    att.value_matrix.ptr[5] = -4;
    att.value_matrix.ptr[6] = -3;
    att.value_matrix.ptr[7] = -2;
    att.value_matrix.ptr[8] = -1;
    att.value_matrix.ptr[9] = 0;

    att.value_vect[0] = 3;
    att.value_vect[1] = -1;
    att.value_vect[2] = 10;
    att.value_vect[3] = 11;
    att.value_vect[4] = 3;

    _ = try att.calulate(seq, ctx);
    matprint(f16, att.out);

}
