const std = @import("std");
const mhatt = @import("mhattention.zig");
const ln = @import("layernorm.zig");
const mlp = @import("multilayerpreceptron.zig");
const mtx = @import("matrix.zig");

const Matrix = mtx.Matrix;
const MHAttention = mhatt.MHAttention;
const LayerNorm = ln.LayerNorm;
const MultilayerPreceptron = mlp.MultilayerPreceptron;

pub fn DecodeLayer(comptime T: type) type {
    return struct {
        mhattention1: MHAttention(T),
        layer_norm1: LayerNorm(T),
        mhattention2: MHAttention(T),
        layer_norm2: LayerNorm(T),
        preceptrom: MultilayerPreceptron(T),
        layer_norm3: LayerNorm(T),

        const Self = @This();
        pub fn compute(self: Self, seq: Matrix(T), ctx: Matrix(T)) !void {
            try self.mhattention1.calculate(seq, seq, .unidirectional);   
            try self.layer_norm1.apply(seq);
            try self.mhattention1.calculate(seq, ctx, .bidirectional);   
            try self.layer_norm2.apply(seq);
            try self.preceptrom.calculate(seq); // outputs should be set to them selfs
            try self.layer_norm3.apply(seq);
        }
    };
}

pub fn Decoder(comptime T: type) type {
    return struct {
        layers: []DecodeLayer(T),

        const Self = @This();
        pub fn compute(self: Self, seq: Matrix(T)) !void {
            for (self.layers) |layer| {
                layer.compute(seq);
            }
        }
    };
}
