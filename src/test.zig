const std = @import("std");
const mha = @import("mhattention.zig");
const att = @import("attention.zig");
const mtx = @import("matrix.zig"); 
const enc = @import("encoder.zig");
const lan = @import("layernorm.zig");
const mlp = @import("multilayerpreceptron.zig");

const Attention = att.Attention;
const AttentionHeader = att.AttentionHeader;
const MHAttention = mha.MHAttention;
const MHAttentionHeader = mha.MHAttentionHeader;
const EncodeLayer = enc.EncodeLayer;
const EncodeLayerHeader = enc.EncodeLayerHeader;
const Matrix = mtx.Matrix;
const LayerNorm = lan.LayerNorm;
const MLP = mlp.MultilayerPreceptron;
const MLPHeader = mlp.MultilayerPreceptronHeader;

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

        var ctx_data = [10]T {
            2, 3,
            1, 7,
            2, 1,
            4, 0,
            3, 1,
        };

        pub const ctx = Matrix(T) {
            .capacity = 10,
            .width = 2,
            .height = 5,
            .ptr = &ctx_data,
        };

        pub const mhatt_header = test_el_header.toMHAttentionHeader();

        pub const head1 = struct {
            var qmat_data = [_]T {
                0.1, -0.2,
                -0.3, 0.2,
                0.2, -0.1,
            };

            var qvec_data = [_]T {0.0, 0.1, 0.2};

            var kmat_data = [_]T {
                0.1, 0.2,
                -0.3, -0.2,
                0.2, -0.2,
            };

            var kvec_data = [_]T {-0.3, 0.1, 0.2};

            var vmat_data = [_]T {
                0.5, -0.5,
                -0.3, -0.2,
            };

            var vvec_data = [_]T {0.1, 0.0};

            var query: [30]T = undefined;
            var key: [30]T = undefined;
            var value: [20]T = undefined;
            var score: [100]T = undefined;

            pub const att = Attention(T) {
                .header = mhatt_header.toAttentionHeader(),
                .query_matrix = Matrix(T) {
                    .capacity = 6,
                    .width = 2,
                    .height = 3,
                    .ptr = &qmat_data,
                },
                .query_vect = &qvec_data,
                .key_matrix = Matrix(T) {
                    .capacity = 6,
                    .width = 2,
                    .height = 3,
                    .ptr = &kmat_data,
                },
                .key_vect = &kvec_data,
                .value_matrix = Matrix(T) {
                    .capacity = 4,
                    .width = 2,
                    .height = 2,
                    .ptr = &vmat_data,
                },
                .value_vect = &vvec_data,
                .key = Matrix(T) {
                    .capacity = 30,
                    .width = 3,
                    .height = 10,
                    .ptr = &key,
                },
                .query = Matrix(T) {
                    .capacity = 30,
                    .width = 3,
                    .height = 10,
                    .ptr = &query,
                },
                .value = Matrix(T) {
                    .capacity = 20,
                    .width = 2,
                    .height = 10,
                    .ptr = &value
                },
                .score = Matrix(T) {
                    .capacity = 100,
                    .width = 10,
                    .height = 10,
                    .ptr = &score,
                },
                .out = Matrix(T) {
                    .capacity = 20,
                    .width = 2,
                    .height = 10,
                    .ptr = undefined,
                },
            };

            pub var expected_out = [10]T {
                0.4372611895, -1.153090062, 
                0.521449242,  -1.102414468, 
                0.3627843846, -1.176291358, 
                0.402549217,  -1.195240529, 
                0.4014339047, -1.180414241, 
            };

            // // for deeper testing
            //
            // pub const expected_query = [15]T {
            //     -0.4,  0.1,  0.3,
            //     -1.3,  1.2, -0.3,
            //      0,   -0.3,  0.5,
            //      0.4, -1.1,  1 , 
            //      0.1, -0.6,  0.7, 
            // };
            //
            // pub const expected_key = [15]T {
            //     0.5, -1.1, -5.551115123e-17,
            //     1.2, -1.6, -1.000000000e0,
            //     0.1, -0.7,  4.000000000e-1,
            //     0.1, -1.1,  1.000000000e0,
            //     0.2, -1,    6.000000000e-1,
            // };
            //
            // pub const expected_value = [10]T {
            //     -0.4, -2.9,  0.6,  2.1,  1.1, 
            //     -1.2, -1.7, -0.8, -1.2, -1.1, 
            // };
            //
            // pub const expected_score = [25]T {
            //     0.1852475402, 0.1287615593,  0.2228380049, 0.2415977396, 0.2215551559,
            //     0.1816634986, 0.09033836584, 0.3019387867, 0.2062673465, 0.2197920024,
            //     0.1863010496, 0.1522145924,  0.1951077428, 0.2486482,    0.2177284152,
            //     0.1766863961, 0.160168573,   0.1574182591, 0.2869622327, 0.2187645391,
            //     0.1818104975, 0.1502708502,  0.1818104975, 0.2661383004, 0.2199698543,
            // };
        };

        pub const head2 = struct {
            var qmat_data = [_]T {
                0.2, -0.7,
                -0.5, 0.4,
                0.6, -0.18,
            };

            var qvec_data = [_]T {0.012, 0.321, 0.142};

            var kmat_data = [_]T {
                -0.1, 0.322,
                0.553, -0.142,
                -0.522, -0.322,
            };

            var kvec_data = [_]T {-0.33, 0.21, -0.12};

            var vmat_data = [_]T {
                0.53, -0.15,
                -0.32, 0.22,
            };

            var vvec_data = [_]T {0.2, -0.01};

            pub const att = Attention(T) {
                .header = mhatt_header.toAttentionHeader(),
                .query_matrix = Matrix(T) {
                    .capacity = 6,
                    .width = 2,
                    .height = 3,
                    .ptr = &qmat_data,
                },
                .query_vect = &qvec_data,
                .key_matrix = Matrix(T) {
                    .capacity = 6,
                    .width = 2,
                    .height = 3,
                    .ptr = &kmat_data,
                },
                .key_vect = &kvec_data,
                .value_matrix = Matrix(T) {
                    .capacity = 4,
                    .width = 2,
                    .height = 2,
                    .ptr = &vmat_data,
                },
                .value_vect = &vvec_data,
                .key = undefined,
                .query = undefined,
                .value = undefined,
                .score = undefined,
                .out = undefined,
            };

            pub var expected_out = [10]T {
                1.7010108058806597e0,   -8.092955826902519e-1,
                2.2202788468064205e0,   -1.213068110898836e0,
                1.2151169366970755e0,   -3.8070070180296456e-1,
                5.663922428419284e-1,   2.4896039342777504e-1,
                1.049237434657965e0,    -2.367536075651379e-1,

            };
        };

        var comb_mat = [8]T {
            0.2, 0.1,
            -0.2, -0.1,

            -0.2, 0.1,
            0.2, -0.1,
        };
        var comb_vect = [2]T {-0.1, 0.2};
        var out: [20]T = undefined;

        var attentions = [2]Attention(T) {head1.att, head2.att};

        pub const mhatt = MHAttention(T) {
            .header = mhatt_header,
            .attentions = &attentions,
            .att_results = undefined,
            .comb_matrix = Matrix(T) {
                .capacity = 8,
                .width = 2,
                .height = 4,
                .ptr = &comb_mat,
            },
            .comb_vect = &comb_vect,
            .out = Matrix(T) {
                .capacity = 20,
                .height = 10,
                .width = 2,
                .ptr = undefined,
            },
        };

        pub const expected_out = [_]T {
            -5.489884888827613e-1,  6.489884888827613e-1,
            -6.713141800568219e-1,  7.71314180056822e-1,
            -4.26165717295922e-1,   5.26165717295922e-1,
            -2.2739661987151033e-1, 3.2739661987151036e-1,
            -3.712774919032714e-1,  4.712774919032714e-1,
        };

        var beta1 = [_]T {-0.1, 0.1};
        var gamma1 = [_]T {0.0, 0.2};

        var mlp_mat1 = [_]T {
            1.2, 0.4,
            -1.0, 0.8,
            1.1, 2.4,
            0.2, -3.4,
            -1.2, -0.4,
        };

        var mlp_vec1 = [_]T {-0.3, -0.2, -0.1, 0.0, -0.1 };

        var mlp_mat2 = [_] T {
            3.3, 2.0, 0.1, 0.2, -0.2,
            -0.9, -7.7, 2.3, 0.3, 0.0,
        };

        var mlp_vec2 = [_]T {-0.1, -0.1};

        var beta2 = [_]T {0.1, -0.1};
        var gamma2 = [_]T {0.12, 0.21};

        pub var encodeLayer = EncodeLayer(T) {
            .header = test_el_header,
            .mhattention = mhatt,
            .layer_norm1 = LayerNorm(T) {
                .beta = &beta1,
                .gamma = &gamma1,
            },
            .preceptron = MLP(T) {
                .header = test_el_header.toMLPHeader(),
                .mat1 = Matrix(T) {
                    .capacity = 10,
                    .height = 5,
                    .width = 2,
                    .ptr = &mlp_mat1,
                },
                .mat2 = Matrix(T) {
                    .capacity = 10,
                    .height = 2,
                    .width = 5,
                    .ptr = &mlp_mat2,
                },
                .vec1 = &mlp_vec1,
                .vec2 = &mlp_vec2,
                .mid = undefined,
                .out = undefined,
            },
            .layer_norm2 = LayerNorm(T) {
                .beta = &beta2,
                .gamma = &gamma2,
            },
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
