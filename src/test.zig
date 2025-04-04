const std = @import("std");
const mhatt = @import("mhattention.zig");
const att = @import("attention.zig");
const mtx = @import("matrix.zig"); 
const enc = @import("encoder.zig");

const Attention = att.Attention;
const AttentionHeader = att.AttentionHeader;
const MHAttention = mhatt.MHAttention;
const MHAttentionHeader = mhatt.MHAttentionHeader;
const EncodeLayer = enc.EncodeLayer;
const EncodeLayerHeader = enc.EncodeLayerHeader;
const Matrix = mtx.Matrix;

pub fn mhattData(comptime T: type) type {
    return struct {
        pub const big_mhatt_header = MHAttentionHeader {
            .version = 0,
            .type_len = @sizeOf(T),
            .heads = 4,
            .seq_dim = 1024,
            .ctx_dim = 1024,
            .att_dim = 1024,
            .mid_dim = 1024,
            .out_dim = 1024,
            .max_seq_len = 1024,
            .max_ctx_len = 1024,
        };

        var big_mhatt_out_cont = [_]T {0} **
            (big_mhatt_header.out_dim*big_mhatt_header.max_seq_len);

        pub const big_mhatt_out = Matrix(T) {
            .capacity = big_mhatt_header.out_dim*big_mhatt_header.max_seq_len,
            .height = big_mhatt_header.max_seq_len,
            .width =  big_mhatt_header.out_dim,
            .ptr = &big_mhatt_out_cont,
        };

        pub const test_mhatt_header = MHAttentionHeader {
            .version = 0,
            .type_len = @sizeOf(T),
            .heads = 4,
            .seq_dim = 1,
            .ctx_dim = 2,
            .att_dim = 3,
            .mid_dim = 4,
            .out_dim = 5,
            .max_seq_len = 7,
            .max_ctx_len = 8,
        };

        pub var test_comb_mat_data = [
            test_mhatt_header.heads *
            test_mhatt_header.out_dim *
            test_mhatt_header.mid_dim
        ]T {
            0.1, 0.2, 0.3, 0.4,
            0.1, 0.2, 0.3, 0.4,
            0.1, 0.2, 0.3, 0.4,
            0.1, 0.2, 0.3, 0.4,
            0.1, 0.2, 0.3, 0.4,

            0.3, 0.2, 0.1, 0.4,
            0.3, 0.2, 0.1, 0.4,
            0.3, 0.2, 0.1, 0.4,
            0.3, 0.2, 0.1, 0.4,
            0.3, 0.2, 0.1, 0.4,

            0.9, 0.2, 0.1, 0.4,
            0.9, 0.2, 0.1, 0.4,
            0.9, 0.2, 0.1, 0.7,
            0.3, 0.2, 0.1, 0.7,
            0.3, 0.2, 0.1, 0.7,

            0.9, 0.2, 0.1, 0.4,
            0.9, 0.5, 0.6, 0.4,
            0.9, 0.5, 0.1, 0.7,
            0.3, 0.5, 0.1, 0.7,
            0.3, 0.2, 0.8, 0.7,
        };

        pub var com_vec_data = [5]T {0.1, -0.1, 0, 0.2, -0.2};
        var att_res_data: [7*4*4]T = undefined;
        var out_data: [5*7]T = undefined;
        pub var test_out_matrix = Matrix(T) {
            .capacity = 5*7,
            .width = 5,
            .height = 7,
            .ptr = &out_data,
        };

        pub var test_att1_cont1 = [_]T {
            0,
            2,
            4,
        };
        pub var test_att1_cont2 = [_]T {0, 1, 2};
        pub var test_att1_cont3 = [_]T {
            0, 1,
            3, 4,
            6, 7,
        };
        pub var test_att1_cont4 = [_]T {-1, -2, -3};
        pub var test_att1_cont5 = [_]T {
            9, 8, 
            6, 5, 
            3, 2, 
            0,-1,
        };
        pub var test_att1_cont6 = [_]T {2, 4, 6, 8};

        pub const test_att1_header = test_mhatt_header.toAttentionHeader();

        const atten = Attention(T) {
            .header = test_att1_header,
            .query_matrix = Matrix(T) {
                .capacity = test_att1_header.att_dim*test_att1_header.seq_dim,
                .height = test_att1_header.att_dim,
                .width = test_att1_header.seq_dim,
                .ptr = &test_att1_cont1,
            },

            .query_vect = &test_att1_cont2,

            .key_matrix = Matrix(T) {
                .capacity = test_att1_header.att_dim*test_att1_header.ctx_dim,
                .height = test_att1_header.att_dim,
                .width = test_att1_header.ctx_dim,
                .ptr = &test_att1_cont3,
            },

            .key_vect = &test_att1_cont4,

            .value_matrix = Matrix(T) {
                .capacity = test_att1_header.out_dim*test_att1_header.ctx_dim,
                .height = test_att1_header.out_dim,
                .width = test_att1_header.ctx_dim,
                .ptr = &test_att1_cont5,
            },

            .value_vect = &test_att1_cont6,

            .key = undefined,
            .value = undefined,
            .query = undefined,
            .score = undefined,
            .out = undefined,
        };

        var mid_data: [4][4*7]T = undefined;
        var attentions = [4]Attention(T) {atten, atten, atten, atten};

        pub const test_mhatt = MHAttention(T) {
            .header = test_mhatt_header,

            .attentions = &attentions,
            .att_results = Matrix(T) {
                .capacity = 7*4*4,
                .height = 7*4,
                .width = 4,
                .ptr = &att_res_data,
            },
            .comb_matrix = Matrix(T) {
                .capacity = 5*4*4,
                .height = 5*4,
                .width = 4,
                .ptr = &test_comb_mat_data,
            },
            .comb_vect = &com_vec_data,

            .out = Matrix(T){
                .capacity = out_data.len,
                .height = 7,
                .width = 5,
                .ptr = &out_data,
            },
        };

        var seq_data = [_]T { 0.1, 0.3, 0.2, 0.4 };
        var ctx_data = [_]T { 0.1, 0.3, 0.2, 0.4, 0.3, 0.4, 0.0, -0.1 };

        pub const test_seq = Matrix(T) {
            .capacity = 4,
            .height = 4,
            .width = 1,
            .ptr = &seq_data,
        };
        pub const test_ctx = Matrix(T) {
            .capacity = 8,
            .height = 4,
            .width = 2,
            .ptr = &ctx_data,
        };

        pub const test_att1_answer_data = [_]T {
            39.63546734, 45.51779338, 46.37794993, 37.45404993, 40.09511809,
            39.9070572, 45.82308028, 46.6671428, 37.6305832, 40.27416612,
            39.78800458, 45.68924128, 46.54031778, 37.55304782, 40.19549336,
            40.00224998, 45.93011503, 46.7686183, 37.69276302, 40.33729753,
        };
        
        pub fn prepareTest() void {
            for (&attentions, &mid_data) |*a, *d| {
                a.out.ptr = d;
                a.out.capacity = 28;
            }
        }


        // testing encoder
    };
}

pub fn encoderData(comptime T: type) type {
    return struct {
        pub const big_el_header = EncodeLayerHeader {
            .version = 0,
            .type_len = @sizeOf(T),
            .heads = 16,
            .ctx_dim = 1024,
            .att_dim = 96,
            .mid_dim = 1024,
            .mlp_dim = 4096,
            .max_ctx_len = 1024,
        };

        pub const test_el_header = EncodeLayerHeader {
            .version = 0,
            .type_len = @sizeOf(T),
            .heads = 2,
            .ctx_dim = 2,
            .att_dim = 3,
            .mid_dim = 4,
            .mlp_dim = 5,
            .max_ctx_len = 10,
        };
    };
}

pub fn compare_delta(comptime T: type, expected: []const T, actual: []const T, delta: T) !void {
    if (expected.len != actual.len) {
        return error.IncompatibleObjects;
    }
    for (expected, actual) |e, a| {
        try std.testing.expectApproxEqRel(e, a, delta);
    }
}
