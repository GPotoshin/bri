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
        pub fn init(allocator: std.mem.Allocator, header: EncodeLayerHeader) !Self {
            var retval: Self = undefined;
            retval.header = header;
            retval.mhattention = try MHAttention(T).init(allocator, header.toMHAttentionHeader(), Matrix(T).empty);
            retval.layer_norm1 = try LayerNorm(T).init(allocator, header.ctx_dim);
            retval.preceptron = try MultilayerPreceptron(T).init(allocator, header.toMLPHeader());
            retval.layer_norm2 = try LayerNorm(T).init(allocator, header.ctx_dim);
            return retval;
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
        
        pub fn destroy(self: *Self, allocator: std.mem.Allocator) void {
            self.mhattention.destroy(allocator);
            self.layer_norm1.destry(allocator);
            self.preceptron.destroy(allocator);
            self.layer_norm2.destroy(allocator);
        }

        const testData = @import("test.zig").encoderData(T);

        test compute {
            testData.set_mhatt();
            try testData.mhatt.compute(testData.ctx, testData.ctx, .bidirectional);
            testData.mhatt.out.print();
        }
    };
}

comptime {
    _ = EncodeLayer(f64);
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
