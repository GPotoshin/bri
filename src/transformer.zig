/// seq2seq transformer
const std = @import("std");
const parsing = @import("parsing.zig");

// is that a good idea?
usingnamespace @import("matrix.zig");
usingnamespace @import("attention.zig");

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
