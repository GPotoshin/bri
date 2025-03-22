/// seq2seq transformer
const std = @import("std");
const parsing = @import("parsing.zig");

// is that a good idea?


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

        position_matrix: Matrix(T),
        // encoder param
        encode: []EncodeLayer(T),

        //decoder param
        dec_attention_matrix: []Matrix(T),
        cross_attention_matrix: []Matrix(T),
        unembedding_matrix: Matrix(T),

        allocator: std.mem.Allocator,

        const Self = @This();

    };
}
