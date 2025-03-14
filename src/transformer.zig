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

    if (out.height != mat2.height) {
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
pub fn addMatVect(comptime T: type, mat: Matrix(T), vec: []T) !void {
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

/// memory for the output should be preallocated and the sizes should be coherent
/// mat: NxM, input: LxM, vect: N, output: LxN
pub fn affine(comptime T: type, payload: struct {mat: Matrix(T), input: Matrix(T),
    vect: []T, output: Matrix(T)}) !void {
    // the sympliest algo
    // const in_dim: u32 = payload.mat.width;
    // const out_dim: u32 = payload.mat.height;

    try matprod(T, payload.mat, payload.input, payload.output);
    try addMatVect(T, payload.output, payload.vect);
}

pub fn Attention(comptime T: type) type {
    return struct {
        prim_dim: u32,
        cont_dim: u32,
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

        out_value: Matrix(T),

        const Self = @This();
        pub fn init(allocator: std.mem.Allocator, hyper_param: struct {
            prim_dim: u32, cont_dim: u32, attn_dim: u32, out_dim: u32}) !Self {

            const retval: Self = undefined;
            retval.query_matrix = try Matrix(T).init(allocator, hyper_param.attn_dim, hyper_param.prim_dim);
            retval.query_vect = try allocator.alloc(T, hyper_param.attn_dim);
            
            retval.key_matrix = try Matrix(T).init(allocator, hyper_param.attn_dim, hyper_param.cont_dim);
            retval.key_vect = try allocator.alloc(T, hyper_param.attn_dim);

            retval.value_matrix = try Matrix(T).init(allocator, hyper_param.out_dim, hyper_param.cont_dim);
            retval.value_vect = try allocator.alloc(T, hyper_param.out_dim);

            return retval;
        }

        pub fn calulate(self: Self, seq: Matrix(T), context: Matrix(T)) Matrix(T) {
            
            // This can be parallelized
            try affine(T, {.mat = self.query_matrix, .input = seq,
                .vect = self.query_vect, .output = self.query});
            try affine(T, {.mat = self.key_matrix, .input = context,
                .vect = self.key_vect, .output = self.key});
            try affine(T, {.mat = self.value_matrix, .input = context,
                .vect = self.key_vect, .output = self.key});

            return out_value;
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
    
    var m1 = try Matrix(i32).init(allocator, 2, 3);
    var m2 = try Matrix(i32).init(allocator, 4, 3);
    const res = try Matrix(i32).init(allocator, 4, 2);

    var vect = [2]i32{1, 1};

    m1.ptr[0] = 2;
    m1.ptr[1] = 3;
    m1.ptr[2] = 4;
    m1.ptr[3] = 1;
    m1.ptr[4] = 2;
    m1.ptr[5] = 1;

    m2.ptr[0] = 3;
    m2.ptr[1] = 4;
    m2.ptr[2] = 5;
    m2.ptr[3] = 0;
    m2.ptr[4] = 6;
    m2.ptr[5] = 6;
    m2.ptr[6] = 1;
    m2.ptr[7] = -1;
    m2.ptr[8] = 1;
    m2.ptr[9] = 2;
    m2.ptr[10] = 1;
    m2.ptr[11] = 3;

    try affine(i32, .{.mat = m1, .input = m2, .vect = (&vect)[0..], .output = res});
    matprint(i32, res);

}
