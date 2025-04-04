const std = @import("std");
const mhatt = @import("mhattention.zig");
const ln = @import("layernorm.zig");
const mlp = @import("multilayerpreceptron.zig");
const mtx = @import("matrix.zig");

const Matrix = mtx.Matrix;
const MHAttention = mhatt.MHAttention;
const LayerNorm = ln.LayerNorm;
const MultilayerPreceptron = mlp.MultilayerPreceptron;

pub const EncodeLayerHeader = struct {
    version: u32,
    type_len: u32,
    heads: u32,
    ctx_dim: u32,
    att_dim: u32,
    mid_dim: u32,
    mlp_dim: u32,
    max_ctx_len: u32,

    const Self = @This();
    pub fn toMHAttentionHeader(self: Self) mhatt.MHAttentionHeader {
        return .{
            .version = self.version,
            .type_len = self.type_len,
            .heads = self.heads,
            .seq_dim = self.ctx_dim,
            .ctx_dim = self.ctx_dim,
            .att_dim = self.att_dim,
            .mid_dim = self.mid_dim,
            .out_dim = self.ctx_dim,
            .max_seq_len = self.max_ctx_len,
            .max_ctx_len = self.max_ctx_len,
        };
    }

    pub fn toMLPHeader(self: Self) mlp.MultilayerPreceptronHeader {
        return .{
            .version = self.version,
            .type_len = self.type_len,
            .in_dim = self.ctx_dim,
            .mid_dim = self.mlp_dim,
            .out_dim = self.ctx_dim,
            .max_seq_len = self.max_ctx_len,
        };
    }
};

pub fn EncodeLayer(comptime T: type) type {
    return struct {
        header: EncodeLayerHeader,

        mhattention: MHAttention(T),
        layer_norm1: LayerNorm(T),
        preceptron: MultilayerPreceptron(T),
        layer_norm2: LayerNorm(T),

        const Self = @This();
        pub fn init(self: Self, allocator: std.mem.Allocator, header: EncodeLayerHeader) !void {
            self.header = header;
            self.mhattention = try MHAttention(T).init(allocator, header.toMHAttentionHeader(), Matrix(T).empty);
            self.layer_norm1 = try LayerNorm(T).init(allocator, header.ctx_dim);
            self.preceptron = try MultilayerPreceptron(T).init(allocator, header.toMLPHeader());
            self.layer_norm2 = try LayerNorm(T).init(allocator, header.ctx_dim);
        }

        pub fn compute(self: Self, ctx: Matrix(T)) !void {
            self.mhattention.out = ctx;
            try self.mhattention.compute(ctx, ctx, .bidirectional);   
            try self.layer_norm1.apply(ctx);
            try self.preceptron.compute(ctx); // outputs should be set to them selfs
            try self.layer_norm2.apply(ctx);
        }

        pub fn writeWeights(self: Self, writer: anytype) !void {
            try self.mhattention.writeWeights(writer);
            try self.layer_norm1.writeWeights(writer);
            try self.preceptron.writeWeights(writer);
            try self.layer_norm1.writeWeights(writer);
        }

        pub fn readWeights(self: Self, reader: anytype) !void {
            try self.mhattention.readWeights(reader);
            try self.layer_norm1.readWeights(reader);
            try self.preceptron.readWeights(reader);
            try self.layer_norm1.readWeights(reader);
        }

        const testData = struct {

            pub const header = EncodeLayerHeader {
                .version = 0,
                .type_len = @sizeOf(T),
                .heads = 2,
                .ctx_dim = 2,
                .att_dim = 3,
                .mid_dim = 4,
                .mlp_dim = 5,
                .max_ctx_len = 6,
            };

            var comb_mat_data = [_]T {
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
            };

            var att_res_data: [header.max_ctx_len*header.heads*header.mid_dim]T = undefined;
            var com_vec_data = [5]T {0.1, -0.1, 0, 0.2, -0.2};

            pub var cont1 = [_]T {
                0,
                2,
                4,
            };
            pub var cont2 = [_]T {0, 1, 2};
            pub var cont3 = [_]T {
                0, 1,
                3, 4,
                6, 7,
            };
            var cont4 = [_]T {-1, -2, -3};
            var cont5 = [_]T {
                9, 8, 
                6, 5, 
                3, 2, 
                0,-1,
            };
            var cont6 = [_]T {2, 4, 6, 8};
            
            const atten = Attention(T) {
                    .header = header2.toAttentionHeader(),
                    .query_matrix = Matrix(T) {
                        .capacity = header2.att_dim*header2.seq_dim,
                        .height = header2.att_dim,
                        .width = header2.seq_dim,
                        .ptr = &cont1,
                    },

                    .query_vect = &cont2,

                    .key_matrix = Matrix(T) {
                        .capacity = header2.att_dim*header2.ctx_dim,
                        .height = header2.att_dim,
                        .width = header2.ctx_dim,
                        .ptr = &cont3,
                    },

                    .key_vect = &cont4,

                    .value_matrix = Matrix(T) {
                        .capacity = header2.out_dim*header2.ctx_dim,
                        .height = header2.mid_dim,
                        .width = header2.ctx_dim,
                        .ptr = &cont5,
                    },

                    .value_vect = &cont6,

                    .key = undefined,
                    .value = undefined,
                    .query = undefined,
                    .score = undefined,
                    .out = undefined,
            };

            var mid_data: [4][4*7]T = undefined;
            var attentions = [4]Attention(T) {atten, atten, atten, atten};

            pub const mhatt = MHAttention(T) {
                .header = header2,

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
                    .ptr = &comb_mat_data,
                },
                .comb_vect = &com_vec_data,


                .out = Matrix(T){
                    .capacity = out_data.len,
                    .height = 7,
                    .width = 5,
                    .ptr = &out_data,
                },
            };
        };

        test compute {

        }
    };
}

pub fn Encoder(comptime T: type) type {
    return struct {
        layers: []EncodeLayer(T),

        const Self = @This();
        pub fn compute(self: Self, seq: Matrix(T)) !void {
            for (self.layers) |layer| {
                layer.compute(seq);
            }
        }
    };
}
