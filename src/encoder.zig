const std = @import("std");
const mhatt = @import("mhattention.zig");
const ln = @import("layernorm.zig");
const mlp = @import("multilayerpreceptron.zig");
const mtx = @import("matrix.zig");

const Matrix = mtx.Matrix;
const MHAttention = mhatt.MHAttention;
const LayerNorm = ln.LayerNorm;
const MultilayerPreceptron = mlp.MultilayerPreceptron;

pub fn EncodeLayer(comptime T: type) type {
    return struct {
        mhattention: MHAttention(T),
        layer_norm1: LayerNorm(T),
        preceptron: MultilayerPreceptron(T),
        layer_norm2: LayerNorm(T),

        const Self = @This();
        pub fn compute(self: Self, seq: Matrix(T)) !void {
            try self.mhattention.compute(seq, seq, .bidirectional);   
            try self.layer_norm1.apply(seq);
            try self.preceptron.compute(seq); // outputs should be set to them selfs
            try self.layer_norm2.apply(seq);
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
