const std = @import("std");
const mhatt = @import("mhattention.zig");
const ln = @import("layernorm.zig");
const mlp = @import("multilayerpreceptron.zig");
const mtx = @import("matrix.zig");
const att = @import("attention.zig");

const Attention = att.Attention;
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

        pub fn copyValuesFrom(dest: *Self, source: Self) !void {
            try dest.mhattention.copyValuesFrom(source.mhattention);
            try dest.layer_norm1.copyValuesFrom(source.layer_norm1);
            try dest.preceptron.copyValuesFrom(source.preceptron);
            try dest.layer_norm2.copyValuesFrom(source.layer_norm2);
        }

        pub fn fillRandom(self: Self, rand: std.Random, k: T) void {
            self.mhattention.fillRandom(rand, k);
            self.layer_norm1.fillRandom(rand, k);
            self.preceptron.fillRandom(rand, k);
            self.layer_norm2.fillRandom(rand, k);
        }

        pub fn isEqualTo(self: Self, expected: Self, delta: T) bool {
            if (
                std.meta.eql(self.header, expected.header) and
                self.mhattention.isEqualTo(expected.mhattention, delta) and
                self.layer_norm1.isEqualTo(expected.layer_norm1, delta) and
                self.preceptron.isEqualTo(expected.preceptron, delta) and
                self.layer_norm2.isEqualTo(expected.layer_norm2, delta)
            ) {
                return true;
            }
            return false;
        }

        pub fn allocateForHeader(self: *Self, allocator: std.mem.Allocator) !void {
            self.mhattention = try MHAttention(T).init(allocator, self.header.toMHAttentionHeader(), Matrix(T).empty);
            try self.mhattention.allocateOut(allocator);
            self.layer_norm1 = try LayerNorm(T).init(allocator, self.header.ctx_dim);
            self.preceptron = try MultilayerPreceptron(T).init(allocator, self.header.toMLPHeader());
            try self.preceptron.allocateOut(allocator);
            self.layer_norm2 = try LayerNorm(T).init(allocator, self.header.ctx_dim);
        }

        pub fn destroy(self: *Self, allocator: std.mem.Allocator) void {
            self.mhattention.destroy(allocator);
            self.layer_norm1.destroy(allocator);
            self.preceptron.destroy(allocator);
            self.layer_norm2.destroy(allocator);
        }

        pub fn apply(self: *Self, ctx: Matrix(T)) !void {
            try self.mhattention.compute(ctx, ctx, .bidirectional);   
            try ctx.add(self.mhattention.out);
            self.layer_norm1.apply(ctx);
            try self.preceptron.compute(ctx); // outputs should be set to them selfs
            try ctx.add(self.preceptron.out);
            self.layer_norm2.apply(ctx);
        }

        pub fn writeWeights(self: Self, writer: anytype) !void {
            try self.mhattention.writeWeights(writer);
            try self.layer_norm1.writeWeights(writer);
            try self.preceptron.writeWeights(writer);
            try self.layer_norm2.writeWeights(writer);
        }

        pub fn readWeights(self: *Self, reader: anytype) !void {
            try self.mhattention.readWeights(reader);
            try self.layer_norm1.readWeights(reader);
            try self.preceptron.readWeights(reader);
            try self.layer_norm2.readWeights(reader);
        }
        
        const testData = @import("test.zig").encoderData(T);

        test writeWeights {
            const file = try std.fs.cwd().openFile("test_files/test_encodelayer", .{.mode = .write_only});
            defer file.close();
            const writer = file.writer();
            try testData.encodeLayer.header.write(writer);
            try testData.encodeLayer.writeWeights(writer);
            try file.setEndPos(try file.getPos());
        }

        test readWeights {
            var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
            defer arena.deinit();
            const allocator = arena.allocator();

            const file = try std.fs.cwd().openFile("test_files/test_encodelayer", .{.mode = .read_only});
            defer file.close();
            const reader = file.reader();

            var enc_layer: EncodeLayer(T) = undefined;
            try enc_layer.header.read(reader);
            try enc_layer.allocateForHeader(allocator);
            try enc_layer.readWeights(reader);

            try std.testing.expect(enc_layer.isEqualTo(testData.encodeLayer, 0.000001));
        }

        // Todo: test computed value
        // Actually this test can be removed, but I may still want to use it one day
        test apply {
            var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
            defer arena.deinit();
            const allocator = arena.allocator();

            // separated pipeline
            const ctx = try testData.ctx.copy(allocator);

            var mhattention: MHAttention(T) = undefined;
            mhattention.header = testData.mhatt_header;
            try mhattention.allocateForHeader(allocator);
            try mhattention.allocateOut(allocator);
            try mhattention.compute(ctx, ctx, .bidirectional);
            try ctx.add(mhattention.out);

            var ln1 = try LayerNorm(T).init(allocator, testData.test_el_header.ctx_dim);
            try ln1.copyValuesFrom(testData.encodeLayer.layer_norm1);
            ln1.apply(ctx);

            var pr: MultilayerPreceptron(T) = undefined;
            pr.header = testData.test_el_header.toMLPHeader();
            try pr.allocateFromHeader(allocator);
            try pr.allocateOut(allocator);
            try pr.compute(ctx); // outputs should be set to them selfs
            try ctx.add(pr.out);

            var ln2 = try LayerNorm(T).init(allocator, testData.test_el_header.ctx_dim);
            try ln2.copyValuesFrom(testData.encodeLayer.layer_norm2);
            ln2.apply(ctx);

            // the actual pipeline
            const ctx2 = try testData.ctx.copy(allocator);
            var encodeLayer: EncodeLayer(T) = undefined;
            encodeLayer.header = testData.test_el_header;
            try encodeLayer.allocateForHeader(allocator);
            try encodeLayer.copyValuesFrom(testData.encodeLayer);
            try encodeLayer.apply(ctx2);

            // if every step is coorect then their composition also
            try std.testing.expect(ctx2.isEqualTo(ctx, 0.001));
        }

        test copyValuesFrom {
            var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
            defer arena.deinit();
            const allocator = arena.allocator();

            var encodeLayer: EncodeLayer(T) = undefined;
            var encodeLayerCopy: EncodeLayer(T) = undefined;
            encodeLayer.header = testData.test_el_header;
            encodeLayerCopy.header = testData.test_el_header;

            try encodeLayer.allocateForHeader(allocator);
            try encodeLayerCopy.allocateForHeader(allocator);

            var xoshiro = std.Random.Xoshiro256.init(123);
            const rand = xoshiro.random();
            encodeLayer.fillRandom(rand, 1);
            try encodeLayerCopy.copyValuesFrom(encodeLayer);
            try std.testing.expect(encodeLayerCopy.isEqualTo(encodeLayer, 0.000001));
        }
    };
}

comptime {
    _ = EncodeLayer(f64);
}

pub const EncoderHeader = struct {
    version: u32,
    type_len: u32,
    layers: u32,
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
        writer.writeInt(u32, self.layers, .little) catch |e| {
            std.log.err("can't write layers\n", .{});
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
        self.layers = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read layers\n", .{});
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

    pub fn toELHeader(self: *Self) EncodeLayerHeader {
        return .{
            .version = self.version,
            .type_len = self.type_len,
            .layers = self.layers,
            .heads = self.heads,
            .ctx_dim = self.ctx_dim,
            .att_dim = self.att_dim,
            .mid_dim = self.mid_dim,
            .mlp_dim = self.mlp_dim,
            .max_ctx_len = self.max_ctx_len,
        };
    }
};

pub fn Encoder(comptime T: type) type {
    return struct {
        header: EncoderHeader,
        layers: ?[]EncodeLayer(T),

        const Self = @This();

        pub fn allocateForHeader(self: *Self, allocator: std.mem.Allocator) !void {
            self.layers = try allocator.alloc(EncodeLayer(T), self.header.layers);
            for (self.layers) |*layer| {
                layer.header = self.header.toELHeader();
                layer.allocateForHeader();
            }
        }

        pub fn apply(self: Self, ctx: Matrix(T)) !void {
            for (self.layers) |layer| {
                layer.apply(ctx);
            }
        }

        pub fn init(allocator: std.mem.Allocator, header: EncoderHeader) !Self {
            var retval: Self = undefined;
            retval.header = header;
            try retval.allocateForHeader(allocator);
            return retval;
        }

        pub fn copyValuesFrom(dest: *Self, source: Self) !void {
            if (std.meta.eql(dest.header, source.header)) { // attention! not all fields should be equal
                return error.IncompatibleObjects;
            }
            for (dest.layers, source.layers) |dlayer, slayer| {
                dlayer.copyValuesFrom(slayer);
            }
        }

        pub fn fillRandom(self: Self, rand: std.Random, k: T) void {
            for (self.layers) |layer| {
                layer.fillRandom(rand, k);
            }
        }

        pub fn isEqualTo(self: Self, expected: Self, delta: T) bool {
            if (std.meta.eql(self.header, expected.header) and self.layers.len == expected.layers.len) {
                for (self.layers, expected.layers) |slayer, elayer| {
                    if (!slayer.isEqualTo(elayer, delta)) {
                        return false;
                    }
                }
                return true;
            }
            return false;
        }

        pub fn destroy(self: *Self, allocator: std.mem.Allocator) void {
            for (self.layers) |layer| {
                layer.destroy(allocator);
            }
            allocator.free(self.layers);
            self.layers = null;
        }
        
        pub fn writeWeights(self: Self, writer: anytype) !void {
            if (self.layers.len != self.header.layers) {
                return error.IncompatibleObjects;
            }

            for (self.layers) |layer| {
                try layer.read.writeWeights(writer);
            }
        }

        pub fn readWeights(self: *Self, reader: anytype) !void {
            if (self.layers.len != self.header.layers) {
                return error.IncompatibleObjects;
            }

            for (self.layers) |layer| {
                try layer.read.readWeights(reader);
            }
        }
    };
}
