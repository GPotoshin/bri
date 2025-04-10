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
    pub fn write(self: Self, writer: anytype) !void {
        writer.writeInt(u32, self.version, .little) catch |e| {
            std.log.err("can't write version\n", .{});
            return e;
        };
        writer.writeInt(u32, self.type_len, .little) catch |e| {
            std.log.err("can't write type_size\n", .{});
            return e;
        };
        writer.writeInt(u32, self.heads, .little) catch |e| {
            std.log.err("can't write heads\n", .{});
            return e;
        };
        writer.writeInt(u32, self.ctx_dim, .little) catch |e| {
            std.log.err("can't write ctx_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.att_dim, .little) catch |e| {
            std.log.err("can't write att_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.mid_dim, .little) catch |e| {
            std.log.err("can't write mid_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.mlp_dim, .little) catch |e| {
            std.log.err("can't write mid_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.max_ctx_len, .little) catch |e| {
            std.log.err("can't write seq_dim\n", .{});
            return e;
        };
    }
    pub fn read(self: *Self, reader: anytype) !void {
        self.version = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read version\n", .{});
            return e;
        };
        self.type_len = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read type_size\n", .{});
            return e;
        };
        self.heads = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read heads\n", .{});
            return e;
        };
        self.ctx_dim = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read ctx_dim\n", .{});
            return e;
        };
        self.att_dim = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read att_dim\n", .{});
            return e;
        };
        self.mid_dim = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read mid_dim\n", .{});
            return e;
        };
        self.mlp_dim = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read mid_dim\n", .{});
            return e;
        };
        self.max_ctx_len = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read seq_dim\n", .{});
            return e;
        };
    }


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
            try retval.allocateForHeader(allocator);
            return retval;
        }

        pub fn allocateForHeader(self: *Self, allocator: std.mem.Allocator) !void {
            self.mhattention = try MHAttention(T).init(allocator, self.header.toMHAttentionHeader(), Matrix(T).empty);
            self.layer_norm1 = try LayerNorm(T).init(allocator, self.header.ctx_dim);
            self.preceptron = try MultilayerPreceptron(T).init(allocator, self.header.toMLPHeader());
            self.layer_norm2 = try LayerNorm(T).init(allocator, self.header.ctx_dim);
        }

        pub fn destroy(self: *Self, allocator: std.mem.Allocator) void {
            self.mhattention.destroy(allocator);
            self.layer_norm1.destroy(allocator);
            self.preceptron.destroy(allocator);
            self.layer_norm2.destroy(allocator);
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

        pub fn readWeights(self: *Self, reader: anytype) !void {
            try self.mhattention.readWeights(reader);
            try self.layer_norm1.readWeights(reader);
            try self.preceptron.readWeights(reader);
            try self.layer_norm1.readWeights(reader);
        }
        
        const testData = @import("test.zig").encoderData(T);

        test writeWeights {
            const file = try std.fs.cwd().openFile("test_files/test_encodelayer", .{.mode = .write_only});
            const writer = file.writer();
            try testData.encodeLayer.header.write(writer);
            try testData.encodeLayer.writeWeights(writer);
            try file.setEndPos(try file.getPos());
        }

        test readWeights {
            const allocator = std.testing.allocator;
            const file = try std.fs.cwd().openFile("test_files/test_encodelayer", .{.mode = .read_only});
            const reader = file.reader();
            var enc_layer: EncodeLayer(T) = undefined;
            try enc_layer.header.read(reader);
            try enc_layer.allocateForHeader(allocator);
            defer enc_layer.destroy(allocator);
            try enc_layer.readWeights(reader);

            const compare_delta = @import("test.zig").compare_delta;
            try compare_delta(T, testData.encodeLayer.mhattention.comb_matrix.toSlice(),
                enc_layer.mhattention.comb_matrix.toSlice(), 0.0001);
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
